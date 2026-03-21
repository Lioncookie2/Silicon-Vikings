"""
Visualisering: initialkart, observerte viewports (rektangler), P(S)/P(P)/P(R).

  PYTHONPATH=. python -m astar.visualize --explore-dir astar/analysis/explore/<session>
  PYTHONPATH=. python -m astar.visualize --round-dir ... --runs path/to/simulate_runs.json

PNG lagres under astar/analysis/plots/ som standard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .baseline import apply_floor_and_renorm, build_terrain_prior, load_initial_state_for_seed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _grid_rgb(grid: list[list[int]]) -> np.ndarray:
    """Map terrain codes til RGB-bilde (H, W, 3)."""
    g = np.asarray(grid, dtype=np.int32)
    h, w = g.shape
    rgb = np.ones((h, w, 3), dtype=np.float32) * 0.9
    rgb[g == 10] = (0.15, 0.35, 0.65)
    rgb[g == 11] = (0.75, 0.72, 0.55)
    rgb[g == 0] = (0.85, 0.85, 0.82)
    rgb[g == 5] = (0.45, 0.45, 0.5)
    rgb[g == 4] = (0.2, 0.55, 0.25)
    rgb[g == 1] = (0.85, 0.2, 0.15)
    rgb[g == 2] = (0.9, 0.65, 0.2)
    rgb[g == 3] = (0.5, 0.35, 0.25)
    return rgb


def _load_run_records(explore_dir: Path | None, runs_path: Path | None) -> list[dict]:
    if explore_dir:
        rd = sorted((explore_dir / "runs").glob("run_*.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in rd]
    if runs_path and runs_path.is_file():
        coll = json.loads(runs_path.read_text(encoding="utf-8"))
        return list(coll.get("runs", []))
    return []


def _plot_seed(
    *,
    detail: dict,
    seed: int,
    run_records: list[dict],
    plots_subdir: Path,
    eps: float,
) -> None:
    initial_states = detail["initial_states"]
    grid, settlements = load_initial_state_for_seed(initial_states, seed)
    p = build_terrain_prior(grid, settlements)
    p = apply_floor_and_renorm(p, eps=eps)

    h, w = p.shape[:2]
    base = _grid_rgb(grid)
    seed_runs = [r for r in run_records if int(r.get("seed_index", -1)) == seed]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].imshow(base)
    axes[0, 0].set_title(f"Initial grid (seed {seed})")
    axes[0, 0].axis("off")

    titles = ["P(settlement)", "P(port)", "P(ruin)"]
    for i, ax in enumerate([axes[0, 1], axes[1, 0], axes[1, 1]]):
        im = ax.imshow(p[:, :, i + 1], vmin=0, vmax=1, cmap="magma")
        ax.set_title(titles[i])
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for r in seed_runs:
        vp = r.get("viewport") or {}
        vx = int(vp.get("x", 0))
        vy = int(vp.get("y", 0))
        vw = int(vp.get("w", 0))
        vh = int(vp.get("h", 0))
        for ax in axes.ravel():
            ax.add_patch(
                Rectangle(
                    (vx - 0.5, vy - 0.5),
                    vw,
                    vh,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                )
            )

    plt.suptitle(f"Astar baseline (terrain prior) — {len(seed_runs)} viewport(er) for seed {seed}")
    plt.tight_layout()
    out = plots_subdir / f"seed{seed}_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(base)
    ax2.set_title(f"Initial map only (seed {seed})")
    ax2.axis("off")
    for r in seed_runs:
        vp = r.get("viewport") or {}
        vx, vy = int(vp.get("x", 0)), int(vp.get("y", 0))
        vw, vh = int(vp.get("w", 0)), int(vp.get("h", 0))
        ax2.add_patch(
            Rectangle(
                (vx - 0.5, vy - 0.5),
                vw,
                vh,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
    p2 = plots_subdir / f"seed{seed}_initial_viewports.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {p2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualiser Astar baseline / prediksjoner.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--round-dir", type=Path, help="Mappe med round_detail.json")
    g.add_argument("--explore-dir", type=Path, help="Exploration-mappe (round_detail + runs/)")
    p.add_argument("--seed", type=int, default=None, help="Kun denne seed (default: alle med data)")
    p.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Valgfri simulate_runs.json (kun med --round-dir)",
    )
    p.add_argument(
        "--prediction-json",
        type=Path,
        default=None,
        help="H×W×6 JSON — overstyrer baseline for heatmaps",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Rot for PNG (default: astar/analysis/plots/<session_navn>)",
    )
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--show", action="store_true", help="Vis også interaktivt (i tillegg til filer)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()

    if args.explore_dir:
        rd = args.explore_dir.resolve()
        detail_path = rd / "round_detail.json"
    else:
        rd = args.round_dir.resolve()
        detail_path = rd / "round_detail.json"

    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    run_records = _load_run_records(args.explore_dir, args.runs)

    session_name = rd.name
    plots_base = args.plots_dir or (root / "astar" / "analysis" / "plots" / session_name)
    plots_base.mkdir(parents=True, exist_ok=True)

    if args.prediction_json:
        p = np.asarray(json.loads(Path(args.prediction_json).read_text(encoding="utf-8")), dtype=np.float64)
        plots_base.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        seed = args.seed or 0
        initial_states = detail["initial_states"]
        grid, _ = load_initial_state_for_seed(initial_states, seed)
        base = _grid_rgb(grid)
        axes[0, 0].imshow(base)
        axes[0, 0].set_title(f"Initial (seed {seed})")
        axes[0, 0].axis("off")
        for i, ax in enumerate([axes[0, 1], axes[1, 0], axes[1, 1]]):
            im = ax.imshow(p[:, :, i + 1], vmin=0, vmax=1, cmap="magma")
            ax.set_title(["P(settlement)", "P(port)", "P(ruin)"][i])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        out = plots_base / f"seed{seed}_from_prediction_json.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"wrote {out}")
        return

    seeds_in_data = sorted({int(r.get("seed_index", 0)) for r in run_records})
    if args.seed is not None:
        seeds_to_plot = [args.seed]
    elif run_records:
        seeds_to_plot = seeds_in_data
    else:
        seeds_to_plot = [0]

    for s in seeds_to_plot:
        _plot_seed(
            detail=detail,
            seed=s,
            run_records=run_records,
            plots_subdir=plots_base,
            eps=args.eps,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
