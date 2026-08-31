import numpy as np

from hbta_frf.graph_damage_gp import (
    PhysicalGraph,
    damage_alignment_matrix,
    damage_alignment_score,
    endpoint_contrast_alignment_matrix,
    endpoint_node_energy_matrix,
    normalized_endpoint_node_energy_matrix,
    normalized_node_energy_matrix,
    fit_healthy_graph_model,
    graph_covariance,
)


def test_sherman_morrison_damaged_resolvent_update() -> None:
    incidence = np.asarray([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    weights = np.asarray([1.2, 0.8])
    laplacian = incidence.T @ (weights[:, None] * incidence)
    alpha, resolvent, _, scale = graph_covariance(laplacian, 0.2, 0.05)
    edge = incidence[0]
    tau = 0.3 * weights[0]
    direction = resolvent @ edge
    coefficient = tau / (1.0 - tau * edge @ direction)
    direct = np.linalg.inv(
        laplacian - tau * np.outer(edge, edge) + alpha * np.eye(3)
    )
    updated = resolvent + coefficient * np.outer(direction, direction)
    np.testing.assert_allclose(updated, direct, rtol=1e-10, atol=1e-10)
    assert scale > 0.0


def test_damage_alignment_is_bounded() -> None:
    incidence = np.asarray([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    weights = np.ones(2)
    laplacian = incidence.T @ incidence
    graph = PhysicalGraph(
        sensor_names=("A", "B", "C"),
        edge_names=(("A", "B"), ("B", "C")),
        relations=("x", "x"),
        incidence=incidence,
        weights=weights,
        laplacian=laplacian,
    )
    generator = np.random.default_rng(2)
    training = generator.normal(size=(5, 2, 3)).astype(np.complex128)
    model = fit_healthy_graph_model(graph, training, 0.1, 0.1)
    score, edge = damage_alignment_score(training, graph, model)
    matrix = damage_alignment_matrix(training, graph, model)
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert np.all((edge >= 0) & (edge < 2))
    np.testing.assert_allclose(score, np.max(matrix, axis=1))
    np.testing.assert_array_equal(edge, np.argmax(matrix, axis=1))

    endpoint = endpoint_contrast_alignment_matrix(training, graph)
    assert endpoint.shape == (len(training), len(graph.edge_names))
    assert np.all((endpoint >= 0.0) & (endpoint <= 1.0))

    node = endpoint_node_energy_matrix(training, graph)
    expected = np.empty_like(node)
    expected[:, 0] = np.mean(
        0.5 * (np.abs(training[:, :, 0]) ** 2 + np.abs(training[:, :, 1]) ** 2),
        axis=1,
    )
    expected[:, 1] = np.mean(
        0.5 * (np.abs(training[:, :, 1]) ** 2 + np.abs(training[:, :, 2]) ** 2),
        axis=1,
    )
    np.testing.assert_allclose(node, expected)

    normalized_nodes = normalized_node_energy_matrix(training)
    np.testing.assert_allclose(np.sum(normalized_nodes, axis=1), 1.0)
    normalized_endpoints = normalized_endpoint_node_energy_matrix(training, graph)
    expected_endpoints = 0.5 * (
        np.abs(graph.incidence) @ normalized_nodes.T
    ).T
    np.testing.assert_allclose(normalized_endpoints, expected_endpoints)
