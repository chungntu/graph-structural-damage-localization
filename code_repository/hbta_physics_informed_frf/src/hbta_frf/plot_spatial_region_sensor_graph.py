"""Plot the HBTA sensor graph with drawing-locked and inferred locations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .run_spatial_region_localization import locked_ground_truth


def read_graph(path: Path):
    nodes: dict[str, tuple[float, float]] = {}
    edges: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left, right = row["sensor_i"], row["sensor_j"]
            nodes.setdefault(left, (float(row["x_i_m"]), float(row["z_i_m"])))
            nodes.setdefault(right, (float(row["x_j_m"]), float(row["z_j_m"])))
            edges.append((left, right, row["relation"]))
    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes, edges = read_graph(args.edge_list)
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    states = [f"DS{i}" for i in range(1, 9)]
    truth = locked_ground_truth()
    inferred = result["methods"]["graph_propagated"]["primary"]["states"]
    colors = {
        "truss_chord": "#496a8d",
        "truss_diagonal": "#7b9dbb",
        "deck_truss_interface": "#a87945",
        "deck_longitudinal": "#6d9567",
        "deck_transverse": "#8c6aa8",
        "lateral_bracing": "#b35b5b",
    }
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharex=True, sharey=True)
    for axis, state in zip(axes.flat, states):
        for left, right, relation in edges:
            x0, z0 = nodes[left]
            x1, z1 = nodes[right]
            axis.plot([x0, x1], [z0, z1], color=colors.get(relation, "#b8b8b8"),
                      alpha=0.20, linewidth=0.55, zorder=1)
        for name, (x, z) in nodes.items():
            axis.scatter(x, z, s=7, marker="s" if name.startswith("AL") else "o",
                         color="#4d4d4d", zorder=2)
        actual = np.asarray(truth[state], dtype=float)
        estimate = np.asarray(inferred[state]["predicted_x_m"], dtype=float)
        for x in actual:
            axis.axvline(x, color="#c33b3b", linewidth=1.0, alpha=0.65, zorder=3)
        for x in estimate:
            axis.axvline(x, color="#1f6f9f", linewidth=1.0, linestyle="--",
                         alpha=0.80, zorder=3)
        axis.scatter(actual, np.full(actual.size, 5.65), marker="*", s=75,
                     color="#c33b3b", edgecolor="white", linewidth=0.35, zorder=5)
        axis.scatter(estimate, np.full(estimate.size, 5.35), marker="o", s=32,
                     color="#1f6f9f", edgecolor="white", linewidth=0.35, zorder=5)
        ok = inferred[state]["all_targets_within_one_panel"]
        axis.set_title(f"{state} {'within panel' if ok else 'outside panel'}", fontsize=10)
        axis.set_xlim(-14.8, 14.8)
        axis.set_ylim(-0.2, 6.0)
        axis.grid(alpha=0.18)
        axis.set_yticks([])
    for axis in axes[1, :]:
        axis.set_xlabel("Longitudinal coordinate x (m)")
    figure.suptitle("HBTA sensor graph with actual and inferred damage locations\n"
                    "(graph-propagated score after P1/P2 modal-response fusion)", fontsize=13)
    figure.text(0.5, 0.895, "star/solid red: drawing-locked damage; circle/dashed blue: inferred region",
                ha="center", va="center", fontsize=9, color="#333333")
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
