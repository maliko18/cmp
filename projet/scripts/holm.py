#!/usr/bin/env python3
"""Applique la correction de Holm aux familles de tests de la campagne."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA = 0.05


def _family(row: pd.Series) -> str:
    comparison = str(row["comparaison"])
    version = str(row["version"])
    if comparison.startswith("DE ") and comparison.endswith(": seq vs GPU"):
        return "DE séquentiel vs GPU"
    if comparison == "PSO ameliore vs DE rand1":
        mode = {
            "seq": "séquentiel",
            "gpu-gbest-gpu": "GPU, gBest GPU",
        }.get(version, version)
        return f"PSO amélioré vs DE rand/1 ({mode})"
    if comparison == "PSO base vs ameliore":
        mode = {
            "seq": "séquentiel",
            "gpu-gbest-gpu": "GPU, gBest GPU",
            "gpu-gbest-cpu": "GPU, gBest CPU",
        }.get(version, version)
        return f"PSO base vs amélioré ({mode})"
    if comparison == "Kruskal-Wallis: 3 strategies DE":
        mode = {"seq": "séquentiel", "gpu": "GPU"}.get(version, version)
        return f"Kruskal-Wallis DE ({mode})"
    return "Autres"


def apply_holm(tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne les p-values de Holm par test et un résumé par famille."""
    columns = [
        "famille", "tests_total", "tests_evaluables", "tests_non_testables",
        "significatifs_bruts", "significatifs_holm", "seuil_alpha",
    ]
    adjusted = tests.copy()
    if adjusted.empty:
        adjusted["famille"] = pd.Series(dtype="string")
        adjusted["p_holm"] = pd.Series(dtype="float64")
        adjusted["significatif_holm"] = pd.Series(dtype="boolean")
        return adjusted, pd.DataFrame(columns=columns)

    adjusted["famille"] = adjusted.apply(_family, axis=1)
    adjusted["p_holm"] = np.nan
    adjusted["significatif_holm"] = pd.Series(
        pd.NA, index=adjusted.index, dtype="boolean"
    )
    for _, indices in adjusted.groupby("famille", sort=False).groups.items():
        p_values = adjusted.loc[indices, "p_value"].dropna().astype(float)
        if p_values.empty:
            continue
        ordered = p_values.sort_values()
        count = len(ordered)
        multipliers = np.arange(count, 0, -1, dtype=float)
        corrected = np.minimum(
            1.0,
            np.maximum.accumulate(ordered.to_numpy() * multipliers),
        )
        adjusted.loc[ordered.index, "p_holm"] = corrected
        adjusted.loc[ordered.index, "significatif_holm"] = corrected < ALPHA

    summaries = []
    for family, group in adjusted.groupby("famille", sort=True):
        valid = group["p_value"].notna()
        summaries.append({
            "famille": family,
            "tests_total": len(group),
            "tests_evaluables": int(valid.sum()),
            "tests_non_testables": int((~valid).sum()),
            "significatifs_bruts": int((group.loc[valid, "p_value"] < ALPHA).sum()),
            "significatifs_holm": int(
                group["significatif_holm"].fillna(False).sum()
            ),
            "seuil_alpha": ALPHA,
        })
    return adjusted, pd.DataFrame(summaries, columns=columns)


def _write_summary_tex(summary: pd.DataFrame, path: Path) -> None:
    headers = {
        "famille": "Famille de tests",
        "tests_total": "Tests totaux",
        "tests_evaluables": "Tests évaluables",
        "tests_non_testables": "Tests non testables",
        "significatifs_bruts": "Significatifs avant Holm",
        "significatifs_holm": "Significatifs après Holm",
        "seuil_alpha": "Seuil",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "\\begin{table*}[t]\n\\centering\n"
            "\\caption{Bilan de la correction de Holm par famille de tests}\n"
            "\\label{tab:holm-summary}\n\\scriptsize\n"
            "\\begin{tabular}{p{4.0cm}rrrrrr}\n\\toprule\n"
        )
        stream.write(
            " & ".join(headers[column] for column in summary.columns)
            + " \\\\\n\\midrule\n"
        )
        for row in summary.itertuples(index=False, name=None):
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("--")
                elif isinstance(value, str):
                    values.append(
                        value.replace("\\", r"\textbackslash{}")
                        .replace("&", r"\&")
                        .replace("_", r"\_")
                        .replace("%", r"\%")
                    )
                else:
                    values.append(str(value))
            stream.write(" & ".join(values) + " \\\\\n")
        stream.write(
            "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
        )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "article" / "tables" / "statistical_tests.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "article" / "tables",
    )
    args = parser.parse_args()
    try:
        tests = pd.read_csv(args.input)
        required = {"comparaison", "version", "p_value"}
        missing = required - set(tests.columns)
        if missing:
            raise ValueError(
                "Colonnes manquantes: " + ", ".join(sorted(missing))
            )
        adjusted, summary = apply_holm(tests)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        adjusted.to_csv(
            args.output_dir / "statistical_tests_holm.csv",
            index=False,
            lineterminator="\n",
        )
        summary.to_csv(
            args.output_dir / "holm_summary.csv",
            index=False,
            lineterminator="\n",
        )
        _write_summary_tex(summary, args.output_dir / "holm_summary.tex")
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"Erreur correction de Holm : {error}", file=sys.stderr)
        return 1

    print(
        f"Correction de Holm terminee : {len(tests)} tests, "
        f"{len(summary)} familles; tableaux dans {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
