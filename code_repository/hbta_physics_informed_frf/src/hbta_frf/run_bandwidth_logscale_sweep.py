"""Log-scale bandwidth sweep of the top-K continuous profile (Figure 13).

Grid: 55 log-spaced bandwidths between 0.02 m and 35 m plus ten reference
values (fractions and multiples of the 3.5-m bay). The pool size is fixed at
K = 8 and every other setting follows run_topk_spatial_localization.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .graph_damage_gp import load_physical_graph
from .run_graph_damage_gp_conformal import twofold_tune
from .run_multireference_spatial_localization import make_edge_scores
from .run_spatial_region_localization import load_edge_midpoints, locked_ground_truth
from .run_topk_spatial_localization import PRIMARY_K, PRIMARY_METHOD, summarize

REFERENCE_BANDWIDTHS_M = (0.05, 0.1, 0.25, 0.5, 0.875, 1.75, 2.625, 3.5, 5.25, 8.75)
BAY_M = 3.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-features", type=Path, required=True)
    parser.add_argument("--noise-features", type=Path, required=True)
    parser.add_argument("--frf-dataset", type=Path, required=True)
    parser.add_argument("--edge-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    cached = np.load(args.sweep_features, allow_pickle=False)
    features = {key: cached[key] for key in cached.files}
    noise = np.load(args.noise_features, allow_pickle=False)
    frf = np.load(args.frf_dataset, allow_pickle=False)
    graph = load_physical_graph(args.edge_list, frf["response_sensor_names"])
    selected = twofold_tune(graph, noise["mode_shapes"][noise["recording_ids"] == 0])["selected"]
    scores = make_edge_scores(features, graph, selected)[PRIMARY_METHOD]
    edge_x = load_edge_midpoints(args.edge_list, graph)
    truth = locked_ground_truth()

    grid = sorted(set(np.geomspace(0.02, 35.0, 55).tolist()) | set(REFERENCE_BANDWIDTHS_M))
    result = {}
    for h in grid:
        summary = summarize(scores, edge_x, truth, h, PRIMARY_K)
        result[f"{h:.4f}"] = {
            "bandwidth_m": h,
            "bandwidth_ratio_Lp": h / BAY_M,
            "mean_absolute_error_m": summary["mean_absolute_error_m"],
            "state_fraction_within_one_panel": summary["state_fraction_within_one_panel"],
            "target_fraction_within_one_panel": summary["target_fraction_within_one_panel"],
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json} ({len(result)} bandwidths)")


if __name__ == "__main__":
    main()
