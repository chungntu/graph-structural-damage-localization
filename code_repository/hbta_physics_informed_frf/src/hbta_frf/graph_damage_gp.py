"""Damage-conditioned Gaussian fields from rank-one graph stiffness loss."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class PhysicalGraph:
    sensor_names: tuple[str, ...]
    edge_names: tuple[tuple[str, str], ...]
    relations: tuple[str, ...]
    incidence: np.ndarray
    weights: np.ndarray
    laplacian: np.ndarray


@dataclass(frozen=True)
class HealthyGraphModel:
    alpha: float
    alpha_factor: float
    nugget_ratio: float
    resolvent: np.ndarray
    covariance: np.ndarray
    covariance_scale: float
    cholesky: tuple[np.ndarray, bool]
    log_determinant: float
    mode_scales: np.ndarray


def load_physical_graph(
    edge_csv: Path,
    sensor_names: np.ndarray,
    skip_unknown: bool = False,
) -> PhysicalGraph:
    """Load a frozen edge-list artifact without importing prior-paper code.

    ``skip_unknown`` constructs the induced measured subgraph when a directional
    channel is unavailable at some sensors.  The default remains strict so a
    misspelled or unexpectedly absent sensor cannot pass silently.
    """
    names = tuple(map(str, sensor_names))
    name_to_index = {name: index for index, name in enumerate(names)}
    rows = []
    with edge_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left = row["sensor_i"]
            right = row["sensor_j"]
            if left not in name_to_index or right not in name_to_index:
                if skip_unknown:
                    continue
                raise ValueError(f"Unknown graph sensor in edge {left}-{right}")
            rows.append(
                (
                    left,
                    right,
                    row["relation"],
                    float(row["structural_path_length_m"]),
                )
            )
    if not rows:
        raise ValueError("Physical edge list is empty")

    incidence = np.zeros((len(rows), len(names)), dtype=np.float64)
    relations = np.asarray([row[2] for row in rows])
    lengths = np.asarray([row[3] for row in rows], dtype=np.float64)
    for edge_index, (left, right, _, _) in enumerate(rows):
        incidence[edge_index, name_to_index[left]] = 1.0
        incidence[edge_index, name_to_index[right]] = -1.0

    weights = np.empty(len(rows), dtype=np.float64)
    for relation in sorted(set(relations)):
        mask = relations == relation
        inverse_length = 1.0 / np.maximum(lengths[mask], 1e-12)
        inverse_length /= inverse_length.mean()
        weights[mask] = inverse_length / mask.sum()
    weights *= len(rows) / len(set(relations))
    laplacian = incidence.T @ (weights[:, None] * incidence)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if np.sum(eigenvalues < 1e-9) != 1:
        raise ValueError("Expected one connected physical graph")
    return PhysicalGraph(
        sensor_names=names,
        edge_names=tuple((row[0], row[1]) for row in rows),
        relations=tuple(relations.tolist()),
        incidence=incidence,
        weights=weights,
        laplacian=laplacian,
    )


def normalize_mode_shapes(mode_shapes: np.ndarray) -> np.ndarray:
    """Remove arbitrary complex modal scale, retaining spatial shape."""
    norms = np.linalg.norm(mode_shapes, axis=-1, keepdims=True)
    return mode_shapes / np.maximum(norms, np.finfo(float).eps)


def fit_mode_shape_template(training_shapes: np.ndarray) -> np.ndarray:
    """Fit a phase-aligned unit-norm template for each mode."""
    normalized = normalize_mode_shapes(training_shapes.astype(np.complex128))
    template = normalized[0].copy()
    for _ in range(5):
        aligned = align_mode_shapes(normalized, template)
        template = normalize_mode_shapes(np.mean(aligned, axis=0))
    return template


def align_mode_shapes(mode_shapes: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Align arbitrary complex phase of every block and mode to a template."""
    normalized = normalize_mode_shapes(mode_shapes.astype(np.complex128))
    inner = np.sum(np.conj(template[None, :, :]) * normalized, axis=-1)
    phase = np.exp(-1j * np.angle(inner))
    return normalized * phase[:, :, None]


def aligned_residuals(mode_shapes: np.ndarray, template: np.ndarray) -> np.ndarray:
    return align_mode_shapes(mode_shapes, template) - template[None, :, :]


def graph_covariance(
    laplacian: np.ndarray,
    alpha_factor: float,
    nugget_ratio: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return alpha, resolvent, normalized covariance, and resolvent scale."""
    eigenvalues = np.linalg.eigvalsh(laplacian)
    positive = eigenvalues[eigenvalues > 1e-9]
    alpha = float(alpha_factor * np.median(positive))
    resolvent = np.linalg.inv(laplacian + alpha * np.eye(len(laplacian)))
    covariance_scale = float(np.mean(np.diag(resolvent)))
    covariance = resolvent / covariance_scale
    covariance += nugget_ratio * np.eye(len(laplacian))
    return alpha, resolvent, covariance, covariance_scale


def estimate_mode_scales(
    residuals: np.ndarray,
    covariance_factor: tuple[np.ndarray, bool],
) -> np.ndarray:
    """Maximum-likelihood complex variance for each modal residual field."""
    node_count = residuals.shape[-1]
    values = []
    for mode_index in range(residuals.shape[1]):
        current = residuals[:, mode_index, :]
        solved = cho_solve(covariance_factor, current.T, check_finite=False).T
        quadratic = np.real(np.sum(np.conj(current) * solved, axis=1))
        values.append(max(float(np.mean(quadratic) / node_count), 1e-10))
    return np.asarray(values)


def fit_healthy_graph_model(
    graph: PhysicalGraph,
    training_residuals: np.ndarray,
    alpha_factor: float,
    nugget_ratio: float,
) -> HealthyGraphModel:
    alpha, resolvent, covariance, covariance_scale = graph_covariance(
        graph.laplacian, alpha_factor, nugget_ratio
    )
    factor = cho_factor(covariance, lower=True, check_finite=False)
    log_determinant = float(2.0 * np.sum(np.log(np.diag(factor[0]))))
    scales = estimate_mode_scales(training_residuals, factor)
    return HealthyGraphModel(
        alpha=alpha,
        alpha_factor=alpha_factor,
        nugget_ratio=nugget_ratio,
        resolvent=resolvent,
        covariance=covariance,
        covariance_scale=covariance_scale,
        cholesky=factor,
        log_determinant=log_determinant,
        mode_scales=scales,
    )


def healthy_nll(residuals: np.ndarray, model: HealthyGraphModel) -> np.ndarray:
    """Complex-Gaussian negative log likelihood up to a shared constant."""
    node_count = residuals.shape[-1]
    score = np.zeros(len(residuals), dtype=np.float64)
    for mode_index, scale in enumerate(model.mode_scales):
        current = residuals[:, mode_index, :]
        solved = cho_solve(model.cholesky, current.T, check_finite=False).T
        quadratic = np.real(np.sum(np.conj(current) * solved, axis=1)) / scale
        score += model.log_determinant + node_count * np.log(scale) + quadratic
    return score


def isotropic_nll(
    training_residuals: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    node_count = residuals.shape[-1]
    training_energy = np.mean(np.abs(training_residuals) ** 2, axis=(0, 2))
    scales = np.maximum(training_energy, 1e-10)
    return np.sum(
        node_count * np.log(scales)[None, :]
        + np.sum(np.abs(residuals) ** 2, axis=2) / scales[None, :],
        axis=1,
    )


def damage_log_likelihood_ratio(
    residuals: np.ndarray,
    graph: PhysicalGraph,
    model: HealthyGraphModel,
    damage_reductions: tuple[float, ...] = (0.1, 0.3, 0.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max exact rank-one covariance likelihood gain over damage edges/severity."""
    solved_residuals = np.empty_like(residuals)
    for mode_index in range(residuals.shape[1]):
        solved_residuals[:, mode_index, :] = cho_solve(
            model.cholesky,
            residuals[:, mode_index, :].T,
            check_finite=False,
        ).T

    best_score = np.full(len(residuals), -np.inf, dtype=np.float64)
    best_edge = np.full(len(residuals), -1, dtype=np.int64)
    best_reduction = np.full(len(residuals), np.nan, dtype=np.float64)
    for edge_index, edge_vector in enumerate(graph.incidence):
        direction = model.resolvent @ edge_vector
        solved_direction = cho_solve(
            model.cholesky, direction, check_finite=False
        )
        direction_precision_energy = float(direction @ solved_direction)
        compliance = float(edge_vector @ direction)
        projections = np.einsum(
            "n,bmn->bm", direction, solved_residuals, optimize=True
        )
        for reduction in damage_reductions:
            tau = float(reduction * graph.weights[edge_index])
            denominator = 1.0 - tau * compliance
            if denominator <= 1e-10:
                continue
            update_scale = tau / (denominator * model.covariance_scale)
            determinant_term = np.log1p(
                update_scale * direction_precision_energy
            )
            quadratic_gain = np.sum(
                update_scale
                * np.abs(projections) ** 2
                / (
                    model.mode_scales[None, :]
                    * (1.0 + update_scale * direction_precision_energy)
                ),
                axis=1,
            )
            score = quadratic_gain - residuals.shape[1] * determinant_term
            improved = score > best_score
            best_score[improved] = score[improved]
            best_edge[improved] = edge_index
            best_reduction[improved] = reduction
    return best_score, best_edge, best_reduction


def damage_alignment_score(
    residuals: np.ndarray,
    graph: PhysicalGraph,
    model: HealthyGraphModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale-invariant, graph-GP-whitened alignment with rank-one damage fields.

    This statistic conditions out total residual energy and tests only whether
    the spatial residual direction agrees with a physically admissible local
    stiffness-loss direction. It is bounded between zero and one.
    """
    all_alignment = damage_alignment_matrix(residuals, graph, model)
    best_edge = np.argmax(all_alignment, axis=1)
    return np.max(all_alignment, axis=1), best_edge


def damage_alignment_matrix(
    residuals: np.ndarray,
    graph: PhysicalGraph,
    model: HealthyGraphModel,
) -> np.ndarray:
    """Alignment of every residual block with every admissible damage edge.

    Unlike :func:`damage_alignment_score`, this function retains all candidate
    edge scores.  That is essential for blind localization audits, where the
    rank of a prespecified ground-truth edge is the test statistic.
    """
    solved_residuals = np.empty_like(residuals)
    for mode_index in range(residuals.shape[1]):
        solved_residuals[:, mode_index, :] = cho_solve(
            model.cholesky,
            residuals[:, mode_index, :].T,
            check_finite=False,
        ).T
    residual_energy = np.real(
        np.sum(np.conj(residuals) * solved_residuals, axis=2)
    )
    residual_energy = np.maximum(residual_energy, np.finfo(float).eps)

    all_alignment = np.empty(
        (len(residuals), len(graph.edge_names)), dtype=np.float64
    )
    for edge_index, edge_vector in enumerate(graph.incidence):
        direction = model.resolvent @ edge_vector
        solved_direction = cho_solve(
            model.cholesky, direction, check_finite=False
        )
        direction_energy = max(
            float(direction @ solved_direction), np.finfo(float).eps
        )
        projection = np.einsum(
            "n,bmn->bm", direction, solved_residuals, optimize=True
        )
        per_mode = (
            np.abs(projection) ** 2
            / (direction_energy * residual_energy)
        )
        all_alignment[:, edge_index] = np.mean(per_mode, axis=1)
    all_alignment = np.clip(all_alignment, 0.0, 1.0)
    return all_alignment


def endpoint_contrast_alignment_matrix(
    residuals: np.ndarray,
    graph: PhysicalGraph,
) -> np.ndarray:
    """Non-propagating baseline using only each edge's endpoint contrast.

    The baseline shares the same candidate edge set but replaces the physical
    resolvent field ``R b_e`` and graph-GP whitening by the raw incidence vector
    ``b_e`` and an isotropic metric.  It therefore tests whether localization
    merely follows a large difference between the two endpoint sensors.
    """
    residual_energy = np.sum(np.abs(residuals) ** 2, axis=2)
    residual_energy = np.maximum(residual_energy, np.finfo(float).eps)
    direction_energy = np.sum(graph.incidence**2, axis=1)
    projection = np.einsum(
        "en,bmn->bme", graph.incidence, residuals, optimize=True
    )
    per_mode = np.abs(projection) ** 2 / (
        residual_energy[:, :, None] * direction_energy[None, None, :]
    )
    return np.clip(np.mean(per_mode, axis=1), 0.0, 1.0)


def endpoint_node_energy_matrix(
    residuals: np.ndarray,
    graph: PhysicalGraph,
) -> np.ndarray:
    """Mean modal-residual energy at the two endpoints of every edge.

    This comparator retains the physical candidate-edge set but discards the
    relative-motion sign and cross term. For residual block ``b`` and edge
    ``(i, j)``, it returns
    ``(1 / (2J)) * sum_m (|r[b,m,i]|**2 + |r[b,m,j]|**2)``.
    """
    node_energy = np.mean(np.abs(residuals) ** 2, axis=1)
    return 0.5 * (np.abs(graph.incidence) @ node_energy.T).T


def normalized_node_energy_matrix(residuals: np.ndarray) -> np.ndarray:
    """Per-node modal-residual energy after normalizing each modal field."""
    energy = np.abs(residuals) ** 2
    total = np.sum(energy, axis=2, keepdims=True)
    normalized = np.divide(
        energy,
        total,
        out=np.zeros_like(energy, dtype=np.float64),
        where=total > np.finfo(float).eps,
    )
    return np.mean(normalized, axis=1)


def normalized_endpoint_node_energy_matrix(
    residuals: np.ndarray,
    graph: PhysicalGraph,
) -> np.ndarray:
    """Normalized node energy averaged over the endpoints of each edge."""
    node_energy = normalized_node_energy_matrix(residuals)
    return 0.5 * (np.abs(graph.incidence) @ node_energy.T).T


def shuffled_graph(graph: PhysicalGraph, seed: int) -> PhysicalGraph:
    """Permute graph labels within AG/AL families while keeping data fixed."""
    generator = np.random.default_rng(seed)
    permutation = np.arange(len(graph.sensor_names))
    for prefix in ("AG", "AL"):
        indices = np.asarray(
            [i for i, name in enumerate(graph.sensor_names) if name.startswith(prefix)]
        )
        permutation[indices] = generator.permutation(indices)
    incidence = graph.incidence[:, permutation]
    laplacian = incidence.T @ (graph.weights[:, None] * incidence)
    return PhysicalGraph(
        sensor_names=graph.sensor_names,
        edge_names=graph.edge_names,
        relations=graph.relations,
        incidence=incidence,
        weights=graph.weights,
        laplacian=laplacian,
    )
