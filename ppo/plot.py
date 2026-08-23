"""Figures from sweep output.

Reads every runs/<grid>/cl{L}_ch{H}/seed{N}/metrics.jsonl under --runs and makes:

  <prefix>_trajectories.png  — entropy and return over training, one line per
                               (clip_low, clip_high) cell, mean +/- sd over seeds.
                               Color encodes clip_low, linestyle encodes clip_high.
  <prefix>_heatmap.png       — final entropy and final return over the grid,
                               single-hue sequential, values printed in cells.

Usage:
    uv run python -m ppo.plot --runs runs/pilot-breakout --out figures --prefix pilot_breakout
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (fixed slot order — do not cycle or reorder).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LINESTYLES = ["-", "--", ":", "-."]
INK = "#333333"

CELL_RE = re.compile(r"cl(?P<cl>[0-9.]+)_ch(?P<ch>[0-9.]+)")


def load_runs(root: Path) -> dict[tuple[float, float], list[list[dict]]]:
    """{(clip_low, clip_high): [metrics-rows per seed, ...]}"""
    cells: dict[tuple[float, float], list[list[dict]]] = defaultdict(list)
    for metrics in sorted(root.glob("cl*_ch*/seed*/metrics.jsonl")):
        m = CELL_RE.search(metrics.parent.parent.name)
        if not m:
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if rows:
            cells[(float(m["cl"]), float(m["ch"]))].append(rows)
    return dict(cells)


def _series(seed_rows: list[list[dict]], key: str, smooth: int = 5):
    """Mean and sd across seeds of a metric, lightly smoothed."""
    n = min(len(rows) for rows in seed_rows)
    steps = np.array([r["global_step"] for r in seed_rows[0][:n]])
    vals = np.array(
        [
            [r[key] if r[key] is not None else np.nan for r in rows[:n]]
            for rows in seed_rows
        ]
    )
    if smooth > 1:
        # NaN-safe moving average normalized by the true window overlap, so
        # the ends don't dive toward zero-padding.
        kernel = np.ones(smooth)
        smoothed = []
        for v in vals:
            ok = ~np.isnan(v)
            num = np.convolve(np.where(ok, v, 0.0), kernel, mode="same")
            den = np.convolve(ok.astype(float), kernel, mode="same")
            smoothed.append(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
        vals = np.array(smoothed)
    return steps, np.nanmean(vals, axis=0), np.nanstd(vals, axis=0)


def _style(cl: float, ch: float, cls: list[float], chs: list[float]):
    return PALETTE[cls.index(cl) % len(PALETTE)], LINESTYLES[
        chs.index(ch) % len(LINESTYLES)
    ]


def _recessive_axes(ax):
    ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors=INK, labelsize=9)


def traj_figure(cells, out_path: Path, title: str, max_entropy: float | None):
    cls = sorted({cl for cl, _ in cells})
    chs = sorted({ch for _, ch in cells})
    fig, (ax_e, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)

    for (cl, ch), seed_rows in sorted(cells.items()):
        color, ls = _style(cl, ch, cls, chs)
        for ax, key in ((ax_e, "entropy"), (ax_r, "episodic_return")):
            steps, mean, sd = _series(seed_rows, key)
            ax.plot(
                steps,
                mean,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=f"clip_low={cl}, clip_high={ch}",
            )
            ax.fill_between(
                steps, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0
            )

    if max_entropy is not None:
        ax_e.axhline(max_entropy, color="#999999", linewidth=1, linestyle=(0, (2, 3)))
        ax_e.annotate(
            "uniform policy",
            (0.99, max_entropy),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#777777",
        )

    ax_e.set_title("Behavior-policy entropy (nats)", fontsize=11, color=INK)
    ax_r.set_title("Episodic return", fontsize=11, color=INK)
    for ax in (ax_e, ax_r):
        ax.set_xlabel("environment steps", fontsize=10, color=INK)
        _recessive_axes(ax)
    ax_e.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _final(seed_rows, key, tail_frac=0.1):
    """Mean of the last tail_frac of training, then mean across seeds."""
    per_seed = []
    for rows in seed_rows:
        tail = rows[-max(1, int(len(rows) * tail_frac)) :]
        vals = [r[key] for r in tail if r[key] is not None]
        if vals:
            per_seed.append(float(np.mean(vals)))
    return float(np.mean(per_seed)) if per_seed else np.nan


def heatmap_figure(cells, out_path: Path, title: str):
    cls = sorted({cl for cl, _ in cells})
    chs = sorted({ch for _, ch in cells})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=160)

    for ax, key, label in zip(
        axes, ("entropy", "episodic_return"), ("final entropy (nats)", "final return")
    ):
        grid = np.full((len(cls), len(chs)), np.nan)
        for (cl, ch), seed_rows in cells.items():
            grid[cls.index(cl), chs.index(ch)] = _final(seed_rows, key)
        im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
        for i in range(len(cls)):
            for j in range(len(chs)):
                if not np.isnan(grid[i, j]):
                    # readable ink on light cells, white on the darkest
                    lo, hi = np.nanmin(grid), np.nanmax(grid)
                    frac = 0.0 if hi == lo else (grid[i, j] - lo) / (hi - lo)
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if frac > 0.6 else INK,
                    )
        ax.set_xticks(range(len(chs)), [str(c) for c in chs])
        ax.set_yticks(range(len(cls)), [str(c) for c in cls])
        ax.set_xlabel("clip_high", fontsize=10, color=INK)
        ax.set_ylabel("clip_low", fontsize=10, color=INK)
        ax.set_title(label, fontsize=11, color=INK)
        ax.tick_params(colors=INK, labelsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="figures")
    p.add_argument("--prefix", required=True)
    p.add_argument("--title", default=None)
    p.add_argument(
        "--max-entropy",
        type=float,
        default=np.log(6),
        help="reference line for the entropy panel (default ln 6, MinAtar)",
    )
    args = p.parse_args()

    cells = load_runs(Path(args.runs))
    if not cells:
        raise SystemExit(f"no runs found under {args.runs}")
    seeds = min(len(v) for v in cells.values())
    title = args.title or f"{Path(args.runs).name} — {seeds} seeds, ent_coef=0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    traj_figure(cells, out / f"{args.prefix}_trajectories.png", title, args.max_entropy)
    if len(cells) >= 4:
        heatmap_figure(cells, out / f"{args.prefix}_heatmap.png", title)
    print(f"wrote figures to {out}/ ({len(cells)} cells, >= {seeds} seeds each)")


if __name__ == "__main__":
    main()
