#!/usr/bin/env python3
"""Cree un jeu GPU/CPU synthetique, explicitement reserve aux tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "algo", "version", "fonction", "dim", "pop", "run", "seed",
    "minimum", "erreur_vs_bias", "temps_s",
]
BIAS = {1: -450.0, 2: -330.0, 3: 390.0, 4: -180.0}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_test")
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    notice = output / "FAKE_DATA_NOTICE.txt"
    notice.write_text(
        "DONNEES SYNTHETIQUES DE TEST UNIQUEMENT.\n"
        "Ces lignes ne sont pas des resultats experimentaux et ne doivent "
        "jamais etre copiees dans results/ ni citees dans l'article.\n",
        encoding="utf-8",
        newline="\n",
    )

    files: dict[str, list[dict[str, object]]] = {}
    configs = ((10, 50, 1), (50, 100, 2), (100, 100, 1), (100, 500, 1))
    for dimension, population, function in configs:
        for run in range(1, 11):
            seed = 987650000 + dimension * 100000 + population * 100 + run
            if dimension == 10 and function == 1:
                pso_offsets = {
                    "pso": BIAS[function] + 0.25,
                    "pso-ameliore": BIAS[function],
                }
            else:
                pso_offsets = {
                    "pso": BIAS[function] + 3.0 + run * 0.1,
                    "pso-ameliore": BIAS[function] + 1.5 + run * 0.08,
                }
            for algo, value in pso_offsets.items():
                for version, seconds in (
                    ("seq", 3.5),
                    ("gpu-gbest-gpu", 0.34),
                ):
                    filename = (
                        f"pso-{'base' if algo == 'pso' else 'ameliore'}-"
                        f"{'seq' if version == 'seq' else 'gpu-gbest-gpu'}.csv"
                    )
                    files.setdefault(filename, []).append({
                        "algo": algo,
                        "version": version,
                        "fonction": function,
                        "dim": dimension,
                        "pop": population,
                        "run": run,
                        "seed": seed,
                        "minimum": value,
                        "erreur_vs_bias": value - BIAS[function],
                        "temps_s": seconds * (dimension / 100) ** 2,
                    })
                if dimension == 100 and population in (100, 500):
                    filename = (
                        f"pso-{'base' if algo == 'pso' else 'ameliore'}-"
                        "gpu-gbest-cpu.csv"
                    )
                    files.setdefault(filename, []).append({
                        "algo": algo,
                        "version": "gpu-gbest-cpu",
                        "fonction": function,
                        "dim": dimension,
                        "pop": population,
                        "run": run,
                        "seed": seed,
                        "minimum": value,
                        "erreur_vs_bias": value - BIAS[function],
                        "temps_s": 1.0,
                    })

            for strategy_index, strategy in enumerate(
                ("rand1", "best1", "currenttobest1")
            ):
                offset = BIAS[function] + 0.5 * strategy_index + run * 0.02
                if dimension == 10 and function == 1:
                    offset = BIAS[function] + 0.5 * strategy_index
                for version, seconds in (("seq", 2.5), ("gpu", 0.25)):
                    filename = f"de-{strategy}-{version}.csv"
                    files.setdefault(filename, []).append({
                        "algo": f"de-{strategy}",
                        "version": version,
                        "fonction": function,
                        "dim": dimension,
                        "pop": population,
                        "run": run,
                        "seed": seed + strategy_index * 1000,
                        "minimum": offset,
                        "erreur_vs_bias": offset - BIAS[function],
                        "temps_s": seconds * (dimension / 100) ** 2,
                    })

    for name, rows in files.items():
        write_csv(output / name, rows)
    print(
        f"{len(files)} CSV synthetiques ecrits dans {output}. "
        "Ne pas utiliser ces donnees comme resultats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
