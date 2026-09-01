"""Full 122-candidate-edge discrete score spectrum for the DS1 confirmatory figure.

``run_ds1_sweep_validation.py`` reports only the target edge's rank and the
top-10 runner-ups for each excitation position. This module recomputes the
same ``endpoint_contrast_alignment_matrix`` score (Eq. 8, s_e) for every one
of the 122 candidate edges at both P1 and P2, so the full-network damage
spectrum figure can be plotted from real data instead of a placeholder
background distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .graph_damage_gp import (
    aligned_residuals,
    endpoint_contrast_alignment_matrix,
    fit_mode_shape_template,
    load_physical_graph,
)


def full_spectrum(features: dict, graph, position: str) -> dict:
    positions = features["positions"].astype(str)
    states = features["states"].astype(str)
    shapes = features["mode_shapes"].astype(np.complex128)
    current = positions == position
    healthy = current & (states == "UDS")
    damaged = current & (states == "DS1")

    template = fit_mode_shape_template(shapes[healthy])
    damaged_residuals = aligned_residuals(shapes[damaged], template)
    damaged_matrix = endpoint_contrast_alignment_matrix(damaged_residuals, graph)
    median_scores = np.median(damaged_matrix, axis=0)

    target_index = next(
        i for i, edge in enumerate(graph.edge_names) if set(edge) == {"AL14", "AL18"}
    )
    order = np.argsort(-median_scores)
    target_rank = int(np.where(order == target_index)[0][0]) + 1

    return {
        "edge_names": [f"{i}-{j}" for i, j in graph.edge_names],
        "median_scores": median_scores.tolist(),
        "target_index": target_index,
        "target_rank": target_rank,
        "target_score": float(median_scores[target_index]),
        "ranked_edge_names": [
            f"{graph.edge_names[idx][0]}-{graph.edge_names[idx][1]}" for idx in order
        ],
        "ranked_scores": median_scores[order].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-features", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    cached = np.load(args.sweep_features, allow_pickle=False)
    features = {key: cached[key] for key in cached.files}
    graph = load_physical_graph(args.edge_list, features["response_sensor_names"])

    result = {
        "status": "ds1_full_edge_spectrum",
        "definition": (
            "Median endpoint-contrast score s_e (Eq. 8) across the two DS1 "
            "sweep passes, for every one of the 122 candidate edges, at each "
            "excitation position."
        ),
        "positions": {
            position: full_spectrum(features, graph, position)
            for position in ("P1", "P2")
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")
    for position, data in result["positions"].items():
        print(f"{position}: target_rank={data['target_rank']} target_score={data['target_score']:.4f}")


if __name__ == "__main__":
    main()
