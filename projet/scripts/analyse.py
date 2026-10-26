#!/usr/bin/env python3
"""Resume les CSV de campagne, tests statistiques, tableaux LaTeX et figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as error:
    raise SystemExit(
        f"Dependance d'analyse absente ({error.name}). "
        "Installer avec: python -m pip install pandas scipy matplotlib"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "algo", "version", "fonction", "dim", "pop", "run", "seed",
    "minimum", "erreur_vs_bias", "temps_s",
}
KNOWN_FILES = {
    "seq": (
        "pso-ameliore-seq.csv", "de-rand1-seq.csv", "pso-base-seq.csv",
        "de-best1-seq.csv", "de-currenttobest1-seq.csv",
    ),
    "gpu": (
        "pso-ameliore-gpu-gbest-gpu.csv", "de-rand1-gpu.csv",
        "pso-base-gpu-gbest-gpu.csv", "de-best1-gpu.csv",
        "de-currenttobest1-gpu.csv", "pso-ameliore-gpu-gbest-cpu.csv",
        "pso-base-gpu-gbest-cpu.csv",
    ),
}
BIAS = {1: -450.0, 2: -330.0, 3: 390.0, 4: -180.0}
PSO_DE_GPU_COLUMNS = [
    "fonction", "dim", "pop",
    "pso_runs", "pso_minimum_mean", "pso_minimum_std", "pso_minimum_median",
    "pso_time_mean_s",
    "de_runs", "de_minimum_mean", "de_minimum_std", "de_minimum_median",
    "de_time_mean_s", "time_ratio_de_over_pso",
    "mann_whitney_p_value", "mann_whitney_conclusion", "quality_winner",
]


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_booktabs(
    frame: pd.DataFrame, path: Path, caption: str, label: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\\begin{table*}[t]\n\\centering\n")
        stream.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        stream.write("\\scriptsize\n")
        if frame.empty:
            stream.write(
                "\\textit{Aucune donnee disponible; executer la campagne.}\n"
            )
            stream.write("\\end{table*}\n")
            return
        columns = list(frame.columns)
        stream.write("\\begin{tabular}{")
        stream.write("l" * len(columns))
        stream.write("}\n\\toprule\n")
        stream.write(" & ".join(tex_escape(column) for column in columns))
        stream.write(" \\\\\n\\midrule\n")
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("--")
                elif isinstance(value, str) and "$\\pm$" in value:
                    values.append(
                        "$\\pm$".join(
                            tex_escape(part) for part in value.split("$\\pm$")
                        )
                    )
                else:
                    values.append(tex_escape(value))
            stream.write(" & ".join(values) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def read_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.csv")) if input_dir.exists() else []
    if not paths:
        print(f"Avertissement: aucun CSV dans {input_dir}", file=sys.stderr)
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Avertissement: CSV ignore {path}: {error}", file=sys.stderr)
            continue
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            print(
                f"Avertissement: {path} ignore; colonnes absentes: "
                f"{', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            continue
        frame = frame[list(sorted(REQUIRED_COLUMNS))].copy()
        for column in (
            "fonction", "dim", "pop", "run", "seed", "minimum",
            "erreur_vs_bias", "temps_s",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(
            subset=[
                "fonction", "dim", "pop", "run", "minimum",
                "erreur_vs_bias", "temps_s",
            ]
        )
        if frame.empty:
            print(f"Avertissement: aucune ligne exploitable dans {path}", file=sys.stderr)
            continue
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[*sorted(REQUIRED_COLUMNS), "source_file"])
    data = pd.concat(frames, ignore_index=True)
    duplicated = data.duplicated(
        subset=["algo", "version", "fonction", "dim", "pop", "run"],
        keep="first",
    )
    if duplicated.any():
        print(
            f"Avertissement: {int(duplicated.sum())} lignes dupliquees "
            "ignorees (meme algo/version/configuration/run).",
            file=sys.stderr,
        )
        data = data.loc[~duplicated].copy()

    for device, filenames in KNOWN_FILES.items():
        present = {path.name for path in paths}
        missing_files = [name for name in filenames if name not in present]
        if missing_files:
            print(
                f"Avertissement: fichiers {device} absents (ignores): "
                + ", ".join(missing_files),
                file=sys.stderr,
            )
    expected_counts = data.groupby(
        ["algo", "version", "fonction", "dim", "pop"],
        observed=True,
    )["run"].nunique()
    incomplete = expected_counts[expected_counts < 10]
    if not incomplete.empty:
        print(
            f"Avertissement: {len(incomplete)} configurations ont moins "
            "de 10 runs (resultats partiels ou mode --rapide).",
            file=sys.stderr,
        )
    return data


def summary_tables(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    grouped = data.groupby(
        ["algo", "version", "fonction", "dim", "pop"],
        observed=True,
        as_index=False,
    )
    result = grouped.agg(
        runs=("run", "nunique"),
        minimum_mean=("minimum", "mean"),
        minimum_std=("minimum", "std"),
        error_mean=("erreur_vs_bias", "mean"),
        error_std=("erreur_vs_bias", "std"),
    )
    result["minimum_mean_pm_std"] = result.apply(
        lambda row: f"{row.minimum_mean:.6g} $\\pm$ {row.minimum_std:.4g}"
        if pd.notna(row.minimum_std) else f"{row.minimum_mean:.6g} (n=1)",
        axis=1,
    )
    result["error_mean_pm_std"] = result.apply(
        lambda row: f"{row.error_mean:.6g} $\\pm$ {row.error_std:.4g}"
        if pd.notna(row.error_std) else f"{row.error_mean:.6g} (n=1)",
        axis=1,
    )
    result = result.sort_values(
        ["fonction", "dim", "pop", "algo", "version"]
    )
    return result


def execution_mode(version: str) -> str:
    lowered = version.lower()
    if lowered == "seq":
        return "seq"
    if "cpu" in lowered:
        return "gpu-gbest-cpu"
    if "gpu" in lowered:
        return "gpu-gbest-gpu"
    return lowered


def timing_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    timing = data.groupby(
        ["algo", "version", "fonction", "dim", "pop"],
        observed=True,
        as_index=False,
    ).agg(
        runs=("run", "nunique"),
        temps_moyen_s=("temps_s", "mean"),
        temps_std_s=("temps_s", "std"),
    )

    data = data.copy()
    data["mode"] = data["version"].map(execution_mode)
    means = data.groupby(
        ["algo", "mode", "fonction", "dim", "pop"],
        observed=True,
    )["temps_s"].mean().unstack("mode")
    speedups = means.reset_index()
    if "seq" not in speedups.columns:
        speedups["seq"] = np.nan
    for mode in ("gpu-gbest-gpu", "gpu-gbest-cpu"):
        if mode not in speedups.columns:
            speedups[mode] = np.nan
    speedups["speedup_gpu_gbest"] = (
        speedups["seq"] / speedups["gpu-gbest-gpu"]
    )
    speedups["speedup_cpu_gbest"] = (
        speedups["seq"] / speedups["gpu-gbest-cpu"]
    )
    speedups = speedups.sort_values(["fonction", "dim", "pop", "algo"])
    return timing, speedups


def rank_sum(first: np.ndarray, second: np.ndarray) -> tuple[str, float | None]:
    if len(first) < 2 or len(second) < 2:
        return "n<2", None
    combined = np.concatenate((first, second))
    if len(np.unique(combined)) == 1:
        return "=", None
    result = stats.mannwhitneyu(
        first, second, alternative="two-sided", method="auto"
    )
    status = "p<0.05" if result.pvalue < 0.05 else "n.s."
    return status, float(result.pvalue)


def statistical_tests(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparaison", "version", "fonction", "dim", "pop", "runs_a",
        "runs_b", "statistique", "p_value", "conclusion",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []

    def add_test(
        name: str,
        version: str,
        function: int,
        dimension: int,
        population: int,
        left: pd.DataFrame,
        right: pd.DataFrame,
    ) -> None:
        a = left["minimum"].to_numpy(dtype=float)
        b = right["minimum"].to_numpy(dtype=float)
        conclusion, pvalue = rank_sum(a, b)
        rows.append({
            "comparaison": name,
            "version": version,
            "fonction": function,
            "dim": dimension,
            "pop": population,
            "runs_a": len(a),
            "runs_b": len(b),
            "statistique": "Mann-Whitney U",
            "p_value": pvalue,
            "conclusion": conclusion,
        })

    keys = ["fonction", "dim", "pop"]
    for key_values, group in data.groupby(keys, observed=True):
        function, dimension, population = map(int, key_values)
        group = group.copy()
        group["mode"] = group["version"].map(execution_mode)

        de = group[group["algo"].astype(str).str.startswith("de-")]
        for strategy in ("rand1", "best1", "currenttobest1"):
            subset = de[de["algo"] == f"de-{strategy}"]
            seq = subset[subset["mode"] == "seq"]
            gpu = subset[subset["mode"] == "gpu-gbest-gpu"]
            if not seq.empty and not gpu.empty:
                add_test(
                    f"DE {strategy}: seq vs GPU", "seq vs gpu",
                    function, dimension, population, seq, gpu,
                )

        # PSO improved vs rand/1, compare only the usual GPU gBest mode.
        for mode in ("seq", "gpu-gbest-gpu"):
            pso = group[group["algo"] == "pso-ameliore"]
            de_rand = group[group["algo"] == "de-rand1"]
            pso = pso[pso["mode"] == mode]
            de_rand = de_rand[de_rand["mode"] == mode]
            if not pso.empty and not de_rand.empty:
                add_test(
                    f"PSO ameliore vs DE rand1", mode,
                    function, dimension, population, pso, de_rand,
                )

        for mode in ("seq", "gpu-gbest-gpu", "gpu-gbest-cpu"):
            base = group[(group["algo"] == "pso") & (group["mode"] == mode)]
            improved = group[
                (group["algo"] == "pso-ameliore")
                & (group["mode"] == mode)
            ]
            if not base.empty and not improved.empty:
                add_test(
                    "PSO base vs ameliore", mode,
                    function, dimension, population, base, improved,
                )

        for mode in ("seq", "gpu-gbest-gpu"):
            samples = [
                de[(de["algo"] == f"de-{strategy}") & (de["mode"] == mode)]
                ["minimum"].to_numpy(dtype=float)
                for strategy in ("rand1", "best1", "currenttobest1")
            ]
            if all(len(sample) >= 2 for sample in samples):
                if len(np.unique(np.concatenate(samples))) == 1:
                    conclusion, pvalue = "=", None
                else:
                    result = stats.kruskal(*samples)
                    conclusion = "p<0.05" if result.pvalue < 0.05 else "n.s."
                    pvalue = float(result.pvalue)
                rows.append({
                    "comparaison": "Kruskal-Wallis: 3 strategies DE",
                    "version": mode,
                    "fonction": function,
                    "dim": dimension,
                    "pop": population,
                    "runs_a": sum(len(sample) for sample in samples),
                    "runs_b": 0,
                    "statistique": "Kruskal-Wallis",
                    "p_value": pvalue,
                    "conclusion": conclusion,
                })

    if not rows:
        print(
            "Avertissement: aucune comparaison statistique complete "
            "(fichiers ou groupes manquants).",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(
        ["comparaison", "fonction", "dim", "pop", "version"]
    )


def pso_vs_de_gpu(
    data: pd.DataFrame, tests: pd.DataFrame
) -> pd.DataFrame:
    """Compare PSO améliore GPU et DE rand/1 GPU par configuration."""
    if data.empty:
        return pd.DataFrame(columns=PSO_DE_GPU_COLUMNS)

    gpu = data[
        (data["version"].map(execution_mode) == "gpu-gbest-gpu")
        & data["algo"].isin(("pso-ameliore", "de-rand1"))
    ]
    if gpu.empty:
        print(
            "Avertissement: aucune ligne pour PSO ameliore GPU gBest GPU "
            "et DE rand1 GPU.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=PSO_DE_GPU_COLUMNS)

    keys = ["fonction", "dim", "pop"]
    summaries = gpu.groupby(
        ["algo", *keys], observed=True
    ).agg(
        runs=("run", "nunique"),
        minimum_mean=("minimum", "mean"),
        minimum_std=("minimum", "std"),
        minimum_median=("minimum", "median"),
        time_mean_s=("temps_s", "mean"),
    ).reset_index()
    pso = summaries[summaries["algo"] == "pso-ameliore"].drop(
        columns="algo"
    ).rename(columns={
        "runs": "pso_runs",
        "minimum_mean": "pso_minimum_mean",
        "minimum_std": "pso_minimum_std",
        "minimum_median": "pso_minimum_median",
        "time_mean_s": "pso_time_mean_s",
    })
    de = summaries[summaries["algo"] == "de-rand1"].drop(
        columns="algo"
    ).rename(columns={
        "runs": "de_runs",
        "minimum_mean": "de_minimum_mean",
        "minimum_std": "de_minimum_std",
        "minimum_median": "de_minimum_median",
        "time_mean_s": "de_time_mean_s",
    })
    comparison = pso.merge(de, on=keys, how="inner", validate="one_to_one")
    if comparison.empty:
        print(
            "Avertissement: aucune configuration commune entre PSO GPU et "
            "DE rand1 GPU.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=PSO_DE_GPU_COLUMNS)

    comparison["time_ratio_de_over_pso"] = (
        comparison["de_time_mean_s"] / comparison["pso_time_mean_s"]
    )

    p_values = tests[
        (tests["comparaison"] == "PSO ameliore vs DE rand1")
        & (tests["version"] == "gpu-gbest-gpu")
    ][[*keys, "p_value", "conclusion"]].rename(columns={
        "p_value": "mann_whitney_p_value",
        "conclusion": "mann_whitney_conclusion",
    })
    comparison = comparison.merge(
        p_values, on=keys, how="left", validate="one_to_one"
    )

    def quality_winner(row: pd.Series) -> str:
        conclusion = row["mann_whitney_conclusion"]
        if conclusion == "=":
            return "Identiques (=)"
        if conclusion != "p<0.05":
            if conclusion == "n.s.":
                return "Pas de difference significative"
            return "Donnees insuffisantes"
        if row["pso_minimum_mean"] < row["de_minimum_mean"]:
            return "PSO ameliore GPU"
        if row["de_minimum_mean"] < row["pso_minimum_mean"]:
            return "DE rand1 GPU"
        return "Moyennes egales (p<0.05)"

    comparison["quality_winner"] = comparison.apply(quality_winner, axis=1)
    comparison["mann_whitney_conclusion"] = comparison[
        "mann_whitney_conclusion"
    ].fillna("n<2")
    return comparison[PSO_DE_GPU_COLUMNS].sort_values(keys).reset_index(drop=True)


def pso_de_latex_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Crée une vue compacte, lisible dans le tableau IEEE de l'article."""
    if comparison.empty:
        return pd.DataFrame(columns=[
            "Fonction", "Dim.", "Pop.",
            "PSO améliore: min. moy. ± ET", "PSO médiane", "PSO temps (s)",
            "DE rand1: min. moy. ± ET", "DE médiane", "DE temps (s)",
            "Temps DE / PSO", "p Mann–Whitney", "Meilleur qualité",
        ])

    def mean_std(mean: float, std: float) -> str:
        std_value = f"{std:.3g}" if pd.notna(std) else "--"
        return f"{mean:.5g} $\\pm$ {std_value}"

    rows = []
    for row in comparison.itertuples(index=False):
        p_value = row.mann_whitney_p_value
        rows.append({
            "Fonction": int(row.fonction),
            "Dim.": int(row.dim),
            "Pop.": int(row.pop),
            "PSO améliore: min. moy. ± ET": mean_std(
                row.pso_minimum_mean, row.pso_minimum_std
            ),
            "PSO médiane": f"{row.pso_minimum_median:.5g}",
            "PSO temps (s)": f"{row.pso_time_mean_s:.5g}",
            "DE rand1: min. moy. ± ET": mean_std(
                row.de_minimum_mean, row.de_minimum_std
            ),
            "DE médiane": f"{row.de_minimum_median:.5g}",
            "DE temps (s)": f"{row.de_time_mean_s:.5g}",
            "Temps DE / PSO": f"{row.time_ratio_de_over_pso:.4g}",
            "p Mann–Whitney": (
                f"{p_value:.4g}"
                if pd.notna(p_value)
                else row.mann_whitney_conclusion
            ),
            "Meilleur qualité": row.quality_winner,
        })
    return pd.DataFrame(rows)


def pso_de_article_latex_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Réduit le tableau Pop=100 aux colonnes nécessaires à la lecture."""
    columns = [
        "Fonction", "Dim.",
        "PSO améliore (moy. ± ET; méd.; s)",
        "DE rand1 (moy. ± ET; méd.; s)",
        "Temps DE / PSO", "p Mann–Whitney", "Meilleur qualité",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    def describe(
        mean: float, std: float, median: float, seconds: float
    ) -> str:
        std_value = f"{std:.3g}" if pd.notna(std) else "--"
        return (
            f"{mean:.5g} $\\pm$ {std_value}; "
            f"{median:.5g}; {seconds:.4g}"
        )

    rows = []
    for row in comparison.itertuples(index=False):
        p_value = row.mann_whitney_p_value
        rows.append({
            "Fonction": int(row.fonction),
            "Dim.": int(row.dim),
            "PSO améliore (moy. ± ET; méd.; s)": describe(
                row.pso_minimum_mean,
                row.pso_minimum_std,
                row.pso_minimum_median,
                row.pso_time_mean_s,
            ),
            "DE rand1 (moy. ± ET; méd.; s)": describe(
                row.de_minimum_mean,
                row.de_minimum_std,
                row.de_minimum_median,
                row.de_time_mean_s,
            ),
            "Temps DE / PSO": f"{row.time_ratio_de_over_pso:.4g}",
            "p Mann–Whitney": (
                f"{p_value:.4g}"
                if pd.notna(p_value)
                else row.mann_whitney_conclusion
            ),
            "Meilleur qualité": row.quality_winner,
        })
    return pd.DataFrame(rows, columns=columns)


def save_placeholder_plot(path: Path, message: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def make_pso_de_gpu_figure(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpu = data[
        (data["version"].map(execution_mode) == "gpu-gbest-gpu")
        & (data["algo"].isin(("pso-ameliore", "de-rand1")))
    ]
    if gpu.empty:
        save_placeholder_plot(
            output_path,
            "Aucune donnee GPU PSO ameliore / DE rand1 disponible.",
        )
        return

    means = gpu.groupby(
        ["algo", "fonction", "dim", "pop"], observed=True
    )["temps_s"].mean().reset_index()
    populations = sorted(means["pop"].astype(int).unique())
    dimensions = sorted(means["dim"].astype(int).unique())
    function_names = {
        1: "Sphere",
        2: "Rastrigin",
        3: "Rosenbrock",
        4: "Griewank",
    }
    function_colors = {
        function: plt.get_cmap("tab10")(function - 1)
        for function in function_names
    }
    algorithm_styles = {
        "pso-ameliore": ("-", "o", "PSO ameliore"),
        "de-rand1": ("--", "s", "DE rand1"),
    }
    fig, axes = plt.subplots(
        1, len(populations),
        figsize=(5.0 * len(populations), 4.2),
        squeeze=False,
    )
    for axis, population in zip(axes[0], populations):
        subset = means[means["pop"] == population]
        for (algo, function), series in subset.groupby(
            ["algo", "fonction"], observed=True
        ):
            series = series.sort_values("dim")
            linestyle, marker, algo_label = algorithm_styles[algo]
            axis.plot(
                series["dim"],
                series["temps_s"],
                linestyle=linestyle,
                marker=marker,
                color=function_colors[int(function)],
                label=f"{function_names[int(function)]}: {algo_label}",
            )
        axis.set_title(f"Population {population}")
        axis.set_xlabel("Dimension")
        axis.set_ylabel("Temps GPU moyen (s)")
        axis.set_xticks(dimensions)
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(output_path)
    plt.close(fig)


def make_figures(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        for name in (
            "runtime_vs_dimension.pdf",
            "speedup_vs_population.pdf",
            "minimum_boxplots.pdf",
        ):
            save_placeholder_plot(output_dir / name, "Aucune donnee disponible.")
        return

    colors = plt.get_cmap("tab10")
    dims = sorted(data["dim"].astype(int).unique())
    pops = sorted(data["pop"].astype(int).unique())
    timed = data.groupby(
        ["algo", "version", "dim", "pop"], observed=True
    )["temps_s"].mean().reset_index()
    fig, axes = plt.subplots(
        1, len(pops), figsize=(5.0 * len(pops), 4.2), squeeze=False
    )
    for axis, population in zip(axes[0], pops):
        plot = timed[timed["pop"] == population]
        for index, ((algo, version), series) in enumerate(
            plot.groupby(["algo", "version"], observed=True)
        ):
            series = series.sort_values("dim")
            axis.plot(
                series["dim"], series["temps_s"], marker="o",
                label=f"{algo} ({version})", color=colors(index % 10),
            )
        axis.set_title(f"Population {population}")
        axis.set_xlabel("Dimension")
        axis.set_ylabel("Temps moyen (s)")
        axis.set_xticks(dims)
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output_dir / "runtime_vs_dimension.pdf")
    plt.close(fig)

    _, speedups = timing_tables(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    has_speedup = False
    for function, axis in zip(sorted(BIAS), axes.flat):
        subset = speedups[speedups["fonction"] == function]
        for algo_index, algo in enumerate(sorted(subset["algo"].unique())):
            algo_rows = subset[subset["algo"] == algo]
            for dimension_index, dimension in enumerate(
                sorted(algo_rows["dim"].unique())
            ):
                series = algo_rows[algo_rows["dim"] == dimension]
                for mode_index, (mode, column) in enumerate((
                    ("GPU gBest", "speedup_gpu_gbest"),
                    ("CPU gBest", "speedup_cpu_gbest"),
                )):
                    valid = series.dropna(subset=[column]).sort_values("pop")
                    if valid.empty:
                        continue
                    axis.plot(
                        valid["pop"], valid[column], marker="o",
                        linestyle="-" if mode_index == 0 else "--",
                        color=colors(algo_index % 10),
                        label=f"{algo}, Dim={dimension}, {mode}",
                    )
                    has_speedup = True
        axis.set_title(f"Fonction {function}")
        axis.set_xlabel("Population")
        axis.set_ylabel("Speedup (seq / GPU)")
        axis.set_xscale("log")
        axis.set_xticks(pops, labels=[str(pop) for pop in pops])
        axis.grid(True, alpha=0.3)
    if has_speedup:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(output_dir / "speedup_vs_population.pdf")
    else:
        plt.close(fig)
        save_placeholder_plot(
            output_dir / "speedup_vs_population.pdf",
            "Speedup indisponible: aucune paire seq/GPU complete.",
        )
    if has_speedup:
        plt.close(fig)

    candidates = ((100, 100, 1), (100, 100, 2), (50, 100, 1), (10, 50, 1))
    chosen: list[tuple[int, int, int]] = []
    for candidate in candidates:
        dimension, population, function = candidate
        subset = data[
            (data["dim"] == dimension)
            & (data["pop"] == population)
            & (data["fonction"] == function)
        ]
        if not subset.empty:
            chosen.append(candidate)
        if len(chosen) == 3:
            break
    if not chosen:
        save_placeholder_plot(
            output_dir / "minimum_boxplots.pdf",
            "Aucune configuration representative disponible.",
        )
        return
    fig, axes = plt.subplots(
        1, len(chosen), figsize=(5 * len(chosen), 4.5), squeeze=False
    )
    for axis, (dimension, population, function) in zip(axes[0], chosen):
        subset = data[
            (data["dim"] == dimension)
            & (data["pop"] == population)
            & (data["fonction"] == function)
        ]
        labels: list[str] = []
        samples: list[np.ndarray] = []
        for (algo, version), group in subset.groupby(
            ["algo", "version"], observed=True
        ):
            labels.append(f"{algo}\n{version}")
            samples.append(group["minimum"].to_numpy(dtype=float))
        axis.boxplot(samples, tick_labels=labels, showfliers=True)
        axis.set_title(
            f"Fonction {function}, Dim={dimension}, Pop={population}"
        )
        axis.set_ylabel("Minimum")
        axis.tick_params(axis="x", labelrotation=35, labelsize=7)
        axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "minimum_boxplots.pdf")
    plt.close(fig)


def run_analysis(input_dir: Path, table_dir: Path, figure_dir: Path) -> None:
    data = read_results(input_dir)
    summary = summary_tables(data)
    timing, speedups = timing_tables(data)
    tests = statistical_tests(data)
    pso_de_comparison = pso_vs_de_gpu(data, tests)
    pso_de_article = pso_de_article_latex_table(
        pso_de_comparison[
            pso_de_comparison["pop"] == 100
        ].copy()
        if not pso_de_comparison.empty
        else pso_de_comparison
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        table_dir / "minimum_summary.csv", index=False, lineterminator="\n"
    )
    timing.to_csv(
        table_dir / "timing_summary.csv", index=False, lineterminator="\n"
    )
    speedups.to_csv(
        table_dir / "speedup_summary.csv", index=False, lineterminator="\n"
    )
    tests.to_csv(
        table_dir / "statistical_tests.csv", index=False, lineterminator="\n"
    )
    pso_de_comparison.to_csv(
        table_dir / "pso_vs_de_gpu.csv", index=False, lineterminator="\n"
    )
    pso_de_comparison[
        pso_de_comparison["pop"] == 100
    ].to_csv(
        table_dir / "pso_vs_de_gpu_pop100.csv",
        index=False,
        lineterminator="\n",
    )
    write_booktabs(
        summary,
        table_dir / "minimum_summary.tex",
        "Moyenne et ecart-type des minima et des erreurs aux biais",
        "tab:generated-minima",
    )
    write_booktabs(
        timing,
        table_dir / "timing_summary.tex",
        "Temps moyens et ecarts-types par configuration",
        "tab:generated-times",
    )
    write_booktabs(
        speedups,
        table_dir / "speedup_summary.tex",
        "Speedup sequentiel/GPU par configuration",
        "tab:generated-speedups",
    )
    test_output = tests.copy()
    if not test_output.empty:
        test_output["p_value"] = test_output["p_value"].map(
            lambda value: "--" if pd.isna(value) else f"{value:.4g}"
        )
    write_booktabs(
        test_output,
        table_dir / "statistical_tests.tex",
        "Tests statistiques non parametriques (seuil 0,05)",
        "tab:generated-tests",
    )
    write_booktabs(
        pso_de_latex_table(pso_de_comparison),
        table_dir / "pso_vs_de_gpu.tex",
        "Comparaison PSO ameliore GPU et DE rand1 GPU par configuration",
        "tab:pso-vs-de-gpu",
    )
    write_booktabs(
        pso_de_article,
        table_dir / "pso_vs_de_gpu_pop100.tex",
        "Comparaison GPU PSO ameliore et DE rand1, Pop=100",
        "tab:pso-vs-de-gpu-pop100",
    )
    make_figures(data, figure_dir)
    make_pso_de_gpu_figure(
        data, figure_dir / "pso_vs_de_gpu_runtime.pdf"
    )
    print(
        f"Analyse terminee: {len(data)} runs, tableaux={table_dir}, "
        f"figures={figure_dir}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "article" / "tables")
    parser.add_argument(
        "--figure-dir", type=Path, default=ROOT / "article" / "figures"
    )
    args = parser.parse_args()
    try:
        run_analysis(
            args.input_dir.resolve(),
            args.table_dir.resolve(),
            args.figure_dir.resolve(),
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Erreur analyse: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
