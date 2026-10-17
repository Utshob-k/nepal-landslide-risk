"""Create report-ready figures from the saved model results.

    python -m src.make_figures

Writes two PNGs under ``results/figures``: one compares random and geographic
validation, and the other shows variation across background draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HEADLINE = {"cutoff_km": 5.0, "block_km": 50, "buffer_km": 10}
MODELS = [
    "Random forest",
    "Random forest + land cover",
    "Gradient boosting",
    "Gradient boosting + land cover",
]


def headline_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Keep the fixed headline setting and the four models shown in the figures."""
    keep = np.ones(len(results), dtype=bool)
    for col, value in HEADLINE.items():
        keep &= results[col].eq(value)
    return results[keep & results["model"].isin(MODELS)].copy()


def _finish(ax, title: str) -> None:
    """Apply the shared styling used by the validation comparison panels."""
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("AUROC")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)


def make_cv_figure(results: pd.DataFrame, output: Path) -> None:
    """Plot random and geographic AUROC for both background schemes."""
    import matplotlib.pyplot as plt

    data = headline_rows(results)
    labels = [
        name.replace("Random forest", "RF").replace("Gradient boosting", "GB")
        for name in MODELS
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True
    )
    for ax, scheme in zip(axes, ["road_weighted", "uniform"], strict=True):
        rows = data[data.abs_set == scheme].set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS))
        ax.barh(
            y + 0.18, rows.random_auroc, height=0.34, color="#9ecae1", label="Random CV"
        )
        ax.barh(
            y - 0.18,
            rows.spatial_auroc,
            height=0.34,
            color="#2171b5",
            label="Spatial CV",
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        _finish(
            ax,
            "Road-matched absences"
            if scheme == "road_weighted"
            else "Uniform absences",
        )
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Spatial validation keeps the model comparison honest",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_absence_figure(results: pd.DataFrame, output: Path) -> None:
    """Show how each fresh background draw changes geographic AUROC."""
    import matplotlib.pyplot as plt

    data = results[results["model"].isin(MODELS)].copy()
    short_labels = ["RF", "RF\n+ land cover", "GB", "GB\n+ land cover"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(bottom=0.27, top=0.84, wspace=0.10)
    for ax, scheme in zip(axes, ["road_weighted", "uniform"], strict=True):
        subset = data[data.abs_set == scheme]
        for i, model in enumerate(MODELS):
            values = subset[subset.model == model].spatial_auroc.to_numpy()
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                s=28,
                color="#2171b5",
                alpha=0.72,
            )
            ax.hlines(values.mean(), i - 0.24, i + 0.24, color="#b2182b", linewidth=2.5)
        ax.set_xticks(range(len(MODELS)), short_labels, fontsize=10)
        ax.set_ylim(0.62, 0.85)
        ax.set_ylabel("Spatial-CV AUROC")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            "Road-matched absences"
            if scheme == "road_weighted"
            else "Uniform absences",
            loc="left",
            weight="bold",
        )
    fig.suptitle("Background sampling moves the score", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.06,
        "Each dot is one absence draw; red line is the mean across 20 draws.",
        ha="center",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create both figures from the committed CSV result tables."""
    baselines = pd.read_csv(ROOT / "results" / "baselines_roads.csv")
    variations = pd.read_csv(ROOT / "results" / "absence_variation.csv")
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    make_cv_figure(baselines, output / "cv_comparison.png")
    make_absence_figure(variations, output / "absence_variation.png")
    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()
