"""Shared constants and data loaders for the R1/R2 manuscript figures."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REVISION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REVISION_DIR.parents[1]
CODE_ROOT = PROJECT_ROOT / "code_repository" / "hbta_physics_informed_frf"
sys.path.insert(0, str(CODE_ROOT / "src"))

from hbta_frf.run_spatial_region_localization import locked_ground_truth  # noqa: E402,F401

FIG_DIR = REVISION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EDGE_LIST = CODE_ROOT / "topology" / "hbta_physical_edge_list.csv"
OUTPUTS = CODE_ROOT / "outputs"

L_BAY = 3.5
# Shaker (MVS) positions read from the official HBTA sensor-layout drawing
# (Zenodo record 14028239, "Sensor layout.pdf"): P2 at the midspan panel point,
# P1 at the panel point two bays (7.0 m) from midspan.
SHAKER_X = {"P1": 7.0, "P2": 0.0}

# Damage types from the benchmark description (Svendsen et al. 2022).
DAMAGE_TYPE = {
    "DS1": "Stringer-to-floor-beam connection",
    "DS2": "Stringer-to-floor-beam connection",
    "DS3": "Stringer cross beam",
    "DS4": "Stringer cross beam",
    "DS5": "Lateral-bracing connection",
    "DS6": "Lateral-bracing connection",
    "DS7": "Lateral-bracing connection",
    "DS8": "Floor-beam-to-main-member connection",
}
TYPE_COLOR = {
    "Stringer-to-floor-beam connection": "#C62828",
    "Stringer cross beam": "#00838F",
    "Lateral-bracing connection": "#EF6C00",
    "Floor-beam-to-main-member connection": "#6A1B9A",
}
RELATION_STYLE = {
    "truss_chord": ("#1E3A5F", "-", 2.0, "Truss chord"),
    "truss_diagonal": ("#4A90C2", "-", 1.5, "Truss diagonal"),
    "deck_longitudinal": ("#E6A100", "-", 1.6, "Deck longitudinal (stringer) path"),
    "deck_transverse": ("#2A9D8F", "-", 1.3, "Deck transverse (cross-girder) path"),
    "lateral_bracing": ("#C0392B", "-", 1.8, "Lateral bracing"),
    "deck_truss_interface": ("#8E6BB0", "--", 1.1, "Deck--truss interface"),
}


def sensor_coordinates() -> dict[str, tuple[float, float, float]]:
    coords: dict[str, tuple[float, float, float]] = {}
    for row in edge_rows():
        coords[row["sensor_i"]] = (float(row["x_i_m"]), float(row["y_i_m"]), float(row["z_i_m"]))
        coords[row["sensor_j"]] = (float(row["x_j_m"]), float(row["y_j_m"]), float(row["z_j_m"]))
    return coords


def edge_rows() -> list[dict]:
    with EDGE_LIST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _discrete_setup():
    from hbta_frf.graph_damage_gp import load_physical_graph
    from hbta_frf.run_ds1_localization_audit import GROUND_TRUTH_EDGE, locate_edge_index
    from hbta_frf.run_spatial_region_localization import load_edge_midpoints

    cache = np.load(CODE_ROOT / "cache" / "all_damage_sweep_modal_features.npz", allow_pickle=False)
    graph = load_physical_graph(EDGE_LIST, cache["response_sensor_names"])
    return cache, graph, load_edge_midpoints(EDGE_LIST, graph), locate_edge_index(graph.edge_names, GROUND_TRUTH_EDGE)


def discrete_benchmark() -> dict:
    """Top-ranked candidate connection per excitation position (Section 4.2).

    For each position, s_e (Eq. 11) is computed against that position's healthy
    template, the median is taken over the two sweep passes, and the midpoint of
    the top-ranked connection is the inferred position. For the two-site state
    DS7, the highest-ranked connection at least one bay away from the first is
    added. A state passes when all errors are within L_bay at both positions.
    """
    from scipy.optimize import linear_sum_assignment
    from hbta_frf.graph_damage_gp import aligned_residuals, endpoint_contrast_alignment_matrix, fit_mode_shape_template

    cache, graph, edge_x, _ = _discrete_setup()
    positions = cache["positions"].astype(str)
    states = cache["states"].astype(str)
    shapes = cache["mode_shapes"].astype(np.complex128)
    truth = locked_ground_truth()
    out = {}
    for state, values in truth.items():
        target = np.asarray(values, dtype=float)
        out[state] = {"truth": target.tolist()}
        for pos in ("P1", "P2"):
            template = fit_mode_shape_template(shapes[(positions == pos) & (states == "UDS")])
            residual = aligned_residuals(shapes[(positions == pos) & (states == state)], template)
            scores = np.median(endpoint_contrast_alignment_matrix(residual, graph), axis=0)
            order = np.argsort(-scores)
            picks = [int(order[0])]
            if len(target) == 2:
                picks.append(next(int(i) for i in order[1:] if abs(edge_x[i] - edge_x[picks[0]]) >= L_BAY))
            predicted = edge_x[picks]
            cost = np.abs(predicted[:, None] - target[None, :])
            rows, cols = linear_sum_assignment(cost)
            by_site = sorted(zip(cols, predicted[rows], cost[rows, cols]))  # ordered by target site
            out[state][pos] = {
                "predicted_x_m": [float(p) for _, p, _ in by_site],
                "errors_m": [float(e) for _, _, e in by_site],
                "edges": ["-".join(graph.edge_names[picks[r]]) for r in rows[np.argsort(cols)]],
            }
        out[state]["pass"] = bool(all(e <= L_BAY + 1e-9 for pos in ("P1", "P2") for e in out[state][pos]["errors_m"]))
    return out


def fused_ds1_spectrum() -> dict:
    """Fused (P1+P2) s_e spectrum for DS1, as in the multi-reference pipeline."""
    from hbta_frf.graph_damage_gp import aligned_residuals, endpoint_contrast_alignment_matrix, fit_mode_shape_template

    cache, graph, _, target = _discrete_setup()
    s = cache["states"].astype(str)
    d = cache["sweep_directions"].astype(str)
    r = cache["recordings"].astype(str)
    shapes = cache["mode_shapes"].astype(np.complex128)
    healthy = np.asarray([
        fit_mode_shape_template(shapes[(s == "UDS") & (d == dd) & np.char.endswith(r, suffix)])
        for suffix in ("01", "02") for dd in ("up", "down")
    ])
    template = fit_mode_shape_template(healthy)
    damaged = np.asarray([fit_mode_shape_template(shapes[(s == "DS1") & (d == dd)]) for dd in ("up", "down")])
    scores = np.median(endpoint_contrast_alignment_matrix(aligned_residuals(damaged, template), graph), axis=0)
    return {
        "scores": scores,
        "target_index": target,
        "target_rank": int(1 + np.sum(scores > scores[target])),
        "median_score": float(np.median(scores)),
    }


def load_permutation_runs() -> dict:
    sweep = json.loads((OUTPUTS / "ds1_sweep_validation.json").read_text(encoding="utf-8"))
    out = {}
    for position in ("P1", "P2"):
        pv = sweep["primary_validations"][position]
        out[position] = {
            "observed_rank": pv["ds1_localization"]["target_aggregate_rank"],
            "null_ranks": np.asarray([run["target_aggregate_rank"] for run in pv["permutation_control"]["runs"]]),
        }
    return out
