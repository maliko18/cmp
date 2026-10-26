#!/usr/bin/env python3
"""Campagne PSO/DE avec reprise des runs deja presents dans les CSV."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = [
    "algo", "version", "fonction", "dim", "pop", "run", "seed",
    "minimum", "erreur_vs_bias", "temps_s",
]
POPS = (50, 100, 500)
FUNCTIONS = (1, 2, 3, 4)
STRATEGIES = ("rand1", "best1", "currenttobest1")
REFERENCE_DIM = 100
SEQ_SECONDS = {"pso": 3.5, "de": 2.5}
# Reference timings are preliminary; GPU DE has no measured reference yet.
GPU_SECONDS = {"pso": 0.34, "de": 2.5}
GPU_CPU_GBEST_SECONDS = 1.0


def executable(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = ROOT / f"{name}{suffix}"
    if path.is_file():
        return path
    if os.name == "nt":
        alternate = ROOT / f"{name}.exe"
        if alternate.is_file():
            return alternate
    raise FileNotFoundError(
        f"Executable introuvable: {path}. Compile d'abord avec `make seq` "
        "ou les commandes MSVC documentees dans le README."
    )


def existing_runs(path: Path, config: dict[str, int], algo: str) -> set[int]:
    if not path.exists():
        return set()
    found: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"En-tete CSV inattendu dans {path}")
        for row in reader:
            if (
                row["algo"] == algo
                and int(row["fonction"]) == config["fonction"]
                and int(row["dim"]) == config["dim"]
                and int(row["pop"]) == config["pop"]
            ):
                run = int(row["run"])
                if run <= 0 or run in found:
                    raise ValueError(f"Numero de run invalide ou duplique dans {path}")
                found.add(run)
    return found


def command_for(
    device: str, job: dict[str, object], config: dict[str, int], count: int
) -> list[str]:
    binary_name = "pso_" + device if str(job["family"]) == "pso" else "de_" + device
    command = [str(executable(binary_name))]
    command.extend(
        str(config[key]) for key in ("fonction", "dim", "pop")
    )
    command.append(str(count))
    if str(job["family"]) == "pso":
        command.extend(["--variante", str(job["variant"])])
        if device == "gpu":
            command.extend(["--gbest", str(job["gbest"])])
    else:
        command.extend(["--strategie", str(job["strategy"])])
    return command


def check_required_executables(
    device: str,
    pending: list[tuple[dict[str, object], dict[str, int], Path, int]],
) -> None:
    required = {
        "pso_" + device if str(job["family"]) == "pso" else "de_" + device
        for job, _, _, _ in pending
    }
    missing = [
        ROOT / f"{name}{'.exe' if os.name == 'nt' else ''}"
        for name in sorted(required)
        if not (ROOT / f"{name}{'.exe' if os.name == 'nt' else ''}").is_file()
    ]
    if not missing:
        return

    print("Executables requis manquants:", file=sys.stderr)
    for path in missing:
        print(f"  {path.name}", file=sys.stderr)
        if os.name == "nt":
            source = (
                "pso/seq/pso_seq.cpp"
                if path.stem == "pso_seq"
                else "de/seq/de_seq.cpp"
            )
            print(
                "  cl /nologo /O2 /EHsc /std:c++17 /Icommon "
                f"/Fe:{path.name} {source}",
                file=sys.stderr,
            )
        else:
            print("  make seq", file=sys.stderr)
    print(
        "Compile les executables avant de lancer la campagne. "
        "Les CSV existants sont conserves; relance ensuite exactement "
        "la meme commande pour reprendre les runs manquants.",
        file=sys.stderr,
    )
    raise FileNotFoundError("Preverification des executables non satisfaite.")


def make_job(family: str, variant: str, device: str, gbest: str = "") -> dict[str, object]:
    if family == "pso":
        algo = "pso-ameliore" if variant == "ameliore" else "pso"
        suffix = f"-gpu-gbest-{gbest}" if device == "gpu" else "-seq"
        return {
            "family": family,
            "variant": variant,
            "device": device,
            "gbest": gbest,
            "algo": algo,
            "filename": f"pso-{variant}{suffix}.csv",
            "label": f"PSO {variant} {device}{' gBest ' + gbest if gbest else ''}",
        }
    algo = f"de-{variant}"
    return {
        "family": family,
        "strategy": variant,
        "device": device,
        "algo": algo,
        "filename": f"{algo}-{device}.csv",
        "label": f"DE {variant} {device}",
    }


def build_jobs(device: str, dimensions: tuple[int, ...]) -> list[dict[str, object]]:
    if device == "seq":
        priority = [
            make_job("pso", "ameliore", "seq"),
            make_job("de", "rand1", "seq"),
        ]
        remainder = [
            make_job("pso", "base", "seq"),
            make_job("de", "best1", "seq"),
            make_job("de", "currenttobest1", "seq"),
        ]
    else:
        priority = [
            make_job("pso", "ameliore", "gpu", "gpu"),
            make_job("de", "rand1", "gpu"),
        ]
        remainder = [
            make_job("pso", "base", "gpu", "gpu"),
            make_job("de", "best1", "gpu"),
            make_job("de", "currenttobest1", "gpu"),
        ]
        if 100 in dimensions:
            remainder.extend([
                make_job("pso", "ameliore", "gpu", "cpu"),
                make_job("pso", "base", "gpu", "cpu"),
            ])
    return priority + remainder


def job_configurations(
    job: dict[str, object], dimensions: tuple[int, ...], device: str
) -> list[dict[str, int]]:
    configs = [
        {"fonction": function, "dim": dimension, "pop": population}
        for dimension in dimensions
        for population in POPS
        for function in FUNCTIONS
    ]
    if device == "gpu" and job.get("gbest") == "cpu":
        configs = [
            config for config in configs
            if config["dim"] == 100 and config["pop"] in (100, 500)
        ]
    return configs


def estimate_seconds(
    device: str,
    job: dict[str, object],
    config: dict[str, int],
    runs: int,
) -> float:
    family = str(job["family"])
    scale = (config["dim"] / REFERENCE_DIM) ** 2
    if device == "seq":
        reference = SEQ_SECONDS[family]
    elif job.get("gbest") == "cpu":
        reference = GPU_CPU_GBEST_SECONDS
    else:
        reference = GPU_SECONDS[family]
    return reference * scale * runs


def read_result(stdout: str, expected_runs: list[int]) -> list[dict[str, str]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    if reader.fieldnames != CSV_COLUMNS:
        raise ValueError(f"Sortie CSV invalide: en-tete {reader.fieldnames!r}")
    rows = list(reader)
    if len(rows) != len(expected_runs):
        raise ValueError(
            f"{len(expected_runs)} lignes attendues, {len(rows)} recues."
        )
    seeds: set[str] = set()
    normalized = []
    for index, row in enumerate(rows):
        if set(row) != set(CSV_COLUMNS) or not row["seed"]:
            raise ValueError("Ligne CSV incomplete.")
        if row["seed"] in seeds:
            raise ValueError("L'executable a emis deux seeds identiques dans un meme lot.")
        seeds.add(row["seed"])
        row["run"] = str(expected_runs[index])
        normalized.append(row)
    return normalized


def append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=CSV_COLUMNS, lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_campaign(device: str, quick: bool, results_dir: Path) -> int:
    dimensions = (10, 50) if quick else (10, 50, 100)
    target_runs = 3 if quick else 10
    jobs = build_jobs(device, dimensions)
    pending: list[tuple[dict[str, object], dict[str, int], Path, int]] = []
    completed = 0
    expected = 0

    for job in jobs:
        path = results_dir / str(job["filename"])
        for config in job_configurations(job, dimensions, device):
            expected += target_runs
            runs = existing_runs(path, config, str(job["algo"]))
            if not set(range(1, target_runs + 1)).issubset(runs):
                missing = len(set(range(1, target_runs + 1)) - runs)
                pending.append((job, config, path, missing))
                completed += target_runs - missing
            else:
                completed += target_runs

    check_required_executables(device, pending)

    if device == "gpu":
        print(
            "Estimation GPU: PSO gBest GPU ~0,34 s/run (reference preliminaire); "
            "PSO gBest CPU ~1,0 s/run (sous-ensemble); DE GPU estime a 2,5 s/run "
            "par equivalence conservative CPU, faute de temps DE GPU mesure."
        )
    else:
        print(
            "Estimation sequentielle: PSO ~3,5 s/run et DE ~2,5 s/run a "
            "Dim=100, Pop=100; extrapolation en Dim^2, tous Pop/fonctions."
        )
    remaining_estimate = sum(
        estimate_seconds(device, job, config, missing)
        for job, config, _, missing in pending
    )
    print(
        f"Mode={device}; {target_runs} runs/config; dimensions={dimensions}; "
        f"progression {completed}/{expected}; "
        f"temps restant estime ~{remaining_estimate / 60:.1f} min "
        f"({remaining_estimate:.0f} s)."
    )
    if not pending:
        print("Campagne deja complete: aucun run a ajouter.")
        return 0

    started = time.monotonic()
    for index, (job, config, path, missing) in enumerate(pending, start=1):
        present = existing_runs(path, config, str(job["algo"]))
        missing_runs = sorted(set(range(1, target_runs + 1)) - present)
        missing = len(missing_runs)
        estimate = estimate_seconds(device, job, config, missing)
        print(
            f"[{index}/{len(pending)}] {job['label']}, "
            f"f={config['fonction']} dim={config['dim']} pop={config['pop']}: "
            f"{missing} run(s), ~{estimate:.1f}s",
            flush=True,
        )
        command = command_for(device, job, config, missing)
        completed_process = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        if completed_process.returncode:
            print(
                f"Echec ({completed_process.returncode}): {' '.join(command)}",
                file=sys.stderr,
            )
            if completed_process.stdout:
                print(completed_process.stdout, file=sys.stderr)
            return completed_process.returncode
        try:
            rows = read_result(completed_process.stdout, missing_runs)
        except (ValueError, csv.Error) as error:
            print(f"Sortie inexploitable ({command}): {error}", file=sys.stderr)
            return 1
        append_rows(path, rows)
    elapsed = time.monotonic() - started
    print(
        f"Campagne {device} terminee en {elapsed / 60:.1f} min. "
        f"CSV dans {results_dir}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("seq", "gpu"), required=True)
    parser.add_argument("--rapide", action="store_true", help="3 runs, Dim 10 et 50")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Dossier CSV (defaut: results/)",
    )
    args = parser.parse_args()
    try:
        return run_campaign(
            args.device, args.rapide, args.results_dir.resolve()
        )
    except (FileNotFoundError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Erreur campagne: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
