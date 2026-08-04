"""Generate the four claim-boundary figures for the SV-Attention arXiv v2."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
DATA = json.loads((ROOT / "data" / "v2_evidence.json").read_text())

BLUE = "#5b6bd6"
RED = "#ee7b7b"
GREEN = "#5fae7e"
GRAY = "#9aa0a6"
INK = "#263238"
PALE_BLUE = "#eef0ff"
PALE_RED = "#fff0f0"
PALE_GREEN = "#eef8f2"


def _setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    FIGS.mkdir(parents=True, exist_ok=True)


def _token(ax, x, y, kind):
    """One memory token. Figure rule: shapes and color carry the meaning;
    numbers stay in the caption."""
    if kind == "support":
        ax.scatter([x], [y], s=430, facecolors=BLUE, edgecolors=INK,
                   linewidths=0.8, zorder=3)
    elif kind == "reserve":
        ax.scatter([x], [y], s=430, facecolors="white", edgecolors=GRAY,
                   linewidths=1.4, zorder=3)
    elif kind == "removed":
        ax.scatter([x], [y], s=430, facecolors="white", edgecolors=GRAY,
                   linewidths=1.2, alpha=0.4, zorder=3)
        ax.text(x, y, "\u00d7", ha="center", va="center", color=GRAY,
                fontsize=12, zorder=4)
    elif kind == "flipped":
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors=RED,
                   linewidths=1.6, zorder=2)
        ax.scatter([x], [y], s=430, facecolors=BLUE, edgecolors=INK,
                   linewidths=0.8, zorder=3)
    elif kind == "new":
        ax.scatter([x], [y], s=430, facecolors="white", edgecolors=GREEN,
                   linewidths=1.8, zorder=3)
        ax.text(x, y, "+", ha="center", va="center", color=GREEN,
                fontsize=11, weight="bold", zorder=4)


def _readout(ax, y_token, arrows):
    """Readout node plus one weighted arrow per contributing token."""
    ax.add_patch(FancyBboxPatch(
        (0.40, 0.05), 0.20, 0.13,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor="white", edgecolor=INK, linewidth=1.1))
    ax.text(0.50, 0.115, "readout", ha="center", va="center", color=INK,
            fontsize=8.5)
    for x, color in arrows:
        ax.add_patch(FancyArrowPatch(
            (x, y_token - 0.085), (0.50 + (x - 0.50) * 0.16, 0.205),
            arrowstyle="-|>", mutation_scale=7, linewidth=1.4,
            color=color, alpha=0.85, shrinkA=0, shrinkB=0))


def hero_contract():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))
    support = (1, 3, 6)
    reserve_token = 4
    xs = np.linspace(0.08, 0.92, 8)
    y_token = 0.74
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    a, b, c = axes

    a.set_title("(a) Solve once: support carries the readout",
                loc="left", fontsize=9.5)
    for i, x in enumerate(xs):
        _token(a, x, y_token, "support" if i in support else "reserve")
    _readout(a, y_token, [(xs[i], BLUE) for i in support])
    a.scatter([], [], s=120, facecolors=BLUE, edgecolors=INK,
              label="support")
    a.scatter([], [], s=120, facecolors="white", edgecolors=GRAY,
              label="reserve (zero weight)")
    a.legend(loc="lower left", frameon=False, fontsize=7.2,
             bbox_to_anchor=(0.0, 0.32), handletextpad=0.2)

    b.set_title("(b) Remove a reserve token: output unchanged",
                loc="left", fontsize=9.5)
    for i, x in enumerate(xs):
        kind = "removed" if i == reserve_token else (
            "support" if i in support else "reserve")
        _token(b, x, y_token, kind)
    _readout(b, y_token, [(xs[i], BLUE) for i in support])
    b.text(0.63, 0.115, "\u2713 unchanged", ha="left", va="center",
           color=GREEN, fontsize=8.5, weight="bold")

    c.set_title("(c) Admit a new token: an old reserve activates",
                loc="left", fontsize=9.5)
    xs_c = np.linspace(0.07, 0.83, 8)
    x_new = 0.945
    for i, x in enumerate(xs_c):
        kind = "flipped" if i == reserve_token else (
            "support" if i in support else "reserve")
        _token(c, x, y_token, kind)
    _token(c, x_new, y_token, "new")
    c.text(x_new, 0.895, "new", ha="center", color=GREEN, fontsize=7.5)
    c.text(xs_c[reserve_token], 0.895, "was reserve", ha="center",
           color=RED, fontsize=7.5)
    _readout(c, y_token,
             [(xs_c[i], BLUE) for i in support]
             + [(xs_c[reserve_token], RED), (x_new, GREEN)])

    fig.tight_layout()
    fig.savefig(FIGS / "hero_contract.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def numerical_paths():
    fig, ax = plt.subplots(figsize=(11.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lanes = (
        (0.585, PALE_GREEN, GREEN, "fp64\nmaintained",
         ("incremental\nactive-set solve", "rank-one\ndecrement",
          "compare with\nfixed-C refit"),
         "deletion audit +\ncustom backward"),
        (0.075, PALE_BLUE, BLUE, "fp32 / MLX\nbatched",
         ("batched FISTA\npartition", "ridged KKT\nsolve",
          "PyTorch\nautograd"),
         "training only,\nno certificate"),
    )
    node_xs = (0.16, 0.385, 0.61)
    node_w, node_h = 0.165, 0.22
    for y0, bg, edge, lane_label, nodes, outcome in lanes:
        ax.add_patch(FancyBboxPatch(
            (0.015, y0), 0.97, 0.34,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor=bg, edgecolor="none"))
        ax.text(0.03, y0 + 0.17, lane_label, ha="left", va="center",
                color=INK, fontsize=8.5, weight="bold")
        y_mid = y0 + 0.17
        for x, label in zip(node_xs, nodes):
            ax.add_patch(FancyBboxPatch(
                (x, y_mid - node_h / 2), node_w, node_h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                facecolor="white", edgecolor=edge, linewidth=1.2))
            ax.text(x + node_w / 2, y_mid, label, ha="center",
                    va="center", fontsize=8, color=INK)
        for x0, x1 in ((node_xs[0] + node_w, node_xs[1]),
                       (node_xs[1] + node_w, node_xs[2])):
            ax.add_patch(FancyArrowPatch(
                (x0 + 0.005, y_mid), (x1 - 0.005, y_mid),
                arrowstyle="-|>", mutation_scale=10, linewidth=1.3,
                color=INK))
        ax.add_patch(FancyArrowPatch(
            (node_xs[2] + node_w + 0.005, y_mid), (0.845, y_mid),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.3,
            color=edge))
        ax.text(0.855, y_mid, outcome, ha="left", va="center",
                fontsize=8, color=edge, weight="bold")

    ax.plot([0.05, 0.95], [0.5, 0.5], ls=(0, (4, 4)), color=GRAY,
            lw=1.0, zorder=1)
    ax.text(0.5, 0.5, " guarantees do not transfer ", ha="center",
            va="center", fontsize=7.5, color=GRAY,
            backgroundcolor="white", zorder=2)

    fig.tight_layout()
    fig.savefig(FIGS / "two_numerical_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def deletion_evidence():
    regimes = DATA["fixed_c_deletion"]["regimes"]
    names = list(regimes)
    x = np.arange(len(names))
    coverage = [
        100 * regimes[name]["completed"] / regimes[name]["attempted"]
        for name in names
    ]
    median = [regimes[name]["median"] for name in names]
    worst = [regimes[name]["worst"] for name in names]
    decay = [regimes[name]["decay_median"] for name in names]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 3.5))
    left.bar(x, coverage, color=BLUE)
    left.set_ylim(97.5, 100.2)
    left.set_ylabel("completed trials (%)")
    left.set_xticks(x, names, rotation=18, ha="right")
    left.set_title("(a) coverage is part of the result", loc="left", weight="bold")
    for i, name in enumerate(names):
        row = regimes[name]
        left.text(
            i,
            coverage[i] - 0.12,
            f"{row['completed']}/{row['attempted']}",
            ha="center",
            va="top",
            color="white" if coverage[i] > 99 else INK,
            fontsize=8,
            weight="bold",
        )

    width = 0.24
    right.bar(x - width, median, width, color=BLUE, label="decrement median")
    right.bar(x, worst, width, color=GREEN, label="decrement worst")
    right.bar(x + width, decay, width, color=RED, label="decay median")
    right.set_yscale("log")
    right.set_ylabel("decision-function deviation from refit")
    right.set_xticks(x, names, rotation=18, ha="right")
    right.set_title("(b) typical, worst, and decay", loc="left", weight="bold")
    right.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "deletion_evidence_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def selection_scope():
    panels = [
        (
            "Skewed redundancy",
            "rare-item recall",
            DATA["selection"]["skewed_redundancy_rare_recall"],
        ),
        (
            "MIMIC-IV",
            "deterioration-hour retention",
            DATA["selection"]["mimic_deterioration_retention"],
        ),
    ]
    colors = {
        "SV gate": BLUE,
        "H2O": RED,
        "Oracle H2O-style": RED,
        "Random": GRAY,
        "Recency": GREEN,
    }
    display = {"Oracle H2O-style": "H2O oracle"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), sharey=True)
    for index, (title, ylabel, values) in enumerate(panels):
        labels = list(values)
        scores = [values[label] for label in labels]
        axes[index].bar(
            np.arange(len(labels)),
            scores,
            color=[colors[label] for label in labels],
        )
        axes[index].set_xticks(
            np.arange(len(labels)),
            [display.get(label, label) for label in labels],
            rotation=18,
            ha="right",
        )
        axes[index].set_ylim(0, 1.0)
        axes[index].set_title(
            f"({chr(97 + index)}) {title}", loc="left", weight="bold"
        )
        axes[index].set_ylabel(ylabel if index == 0 else "")
        for x_pos, score in enumerate(scores):
            axes[index].text(
                x_pos, score + 0.025, f"{score:.2f}", ha="center", fontsize=8
            )
    fig.suptitle(
        "Selection helps when informative records are atypical; it is not a universal importance score.",
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "selection_scope_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    _setup()
    hero_contract()
    numerical_paths()
    deletion_evidence()
    selection_scope()
    print(f"wrote four figures to {FIGS}")


if __name__ == "__main__":
    main()
