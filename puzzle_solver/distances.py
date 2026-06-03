import numpy as np


MGC_DUMMY_GRADIENTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ]
)


def euclideanDistance(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff.reshape(diff.shape[0], -1)
    return float(np.linalg.norm(diff, axis=1).sum())


def euclideanDistances(edge, compare_edges):
    diff = edge[np.newaxis, :, :] - compare_edges
    return np.sqrt(np.sum(diff * diff, axis=2)).sum(axis=1)


def predictionDistance(source_edge, target_edge):
    predicted = source_edge.edge + (source_edge.edge - source_edge.adjacent_edge)
    diff = predicted - target_edge.edge
    return float(np.sqrt(np.sum(diff * diff, axis=1)).sum())


def predictionDistances(edge, adjacent_edge, compare_edges):
    predicted = edge + (edge - adjacent_edge)
    diff = predicted[np.newaxis, :, :] - compare_edges
    return np.sqrt(np.sum(diff * diff, axis=2)).sum(axis=1)


def predictionEdgeDistance(own_edge, compare_edge):
    return (
        predictionDistance(own_edge, compare_edge)
        + predictionDistance(compare_edge, own_edge)
    )


def predictionEdgeDistances(
        edge,
        adjacent_edge,
        compare_edges,
        compare_adjacent_edges):
    return (
        predictionDistances(edge, adjacent_edge, compare_edges)
        + predictionDistancesForTargets(
            compare_edges,
            compare_adjacent_edges,
            edge,
        )
    )


def predictionDistancesForTargets(edges, adjacent_edges, target_edge):
    predicted = edges + (edges - adjacent_edges)
    diff = predicted - target_edge[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).sum(axis=1)


def mahalanobisEdgeDistance(own_edge, compare_edge):
    matrix = (own_edge.edge - compare_edge.edge) - own_edge.average_delta
    matrix2 = (compare_edge.edge - own_edge.edge) - compare_edge.average_delta

    scores = np.einsum(
        "ij,jk,ik->i", matrix, compare_edge.inverse_covariance, matrix)
    scores2 = np.einsum(
        "ij,jk,ik->i", matrix2, own_edge.inverse_covariance, matrix2)
    return float(np.sqrt(np.abs(scores)).sum() + np.sqrt(np.abs(scores2)).sum())


def mahalanobisEdgeDistances(
        edge,
        average_delta,
        inverse_covariance,
        compare_edges,
        compare_average_deltas,
        compare_inverse_covariances):
    matrix = (edge[np.newaxis, :, :] - compare_edges) - average_delta
    matrix2 = (
        compare_edges
        - edge[np.newaxis, :, :]
        - compare_average_deltas[:, np.newaxis, :]
    )

    scores = np.einsum(
        "mij,mjk,mik->mi",
        matrix,
        compare_inverse_covariances,
        matrix,
    )
    scores2 = np.einsum(
        "mij,jk,mik->mi",
        matrix2,
        inverse_covariance,
        matrix2,
    )
    return (
        np.sqrt(np.abs(scores)).sum(axis=1)
        + np.sqrt(np.abs(scores2)).sum(axis=1)
    )


def mgcDirectionalDistance(source_edge, target_edge, sqrt=False):
    seam_gradient = target_edge.edge - source_edge.edge
    centered = seam_gradient - source_edge.gradient_average
    scores = np.einsum(
        "ij,jk,ik->i",
        centered,
        source_edge.gradient_inverse_covariance,
        centered,
    )
    scores = np.maximum(scores, 0.0)
    if sqrt:
        scores = np.sqrt(scores)
    return float(scores.sum())


def mgcEdgeDistance(own_edge, compare_edge):
    return (
        mgcDirectionalDistance(own_edge, compare_edge)
        + mgcDirectionalDistance(compare_edge, own_edge)
    )


def mgcEdgeMahalanobisDistance(own_edge, compare_edge):
    return (
        mgcDirectionalDistance(own_edge, compare_edge, sqrt=True)
        + mgcDirectionalDistance(compare_edge, own_edge, sqrt=True)
    )


def mgcEdgeDistances(
        edge,
        gradient_average,
        gradient_inverse_covariance,
        compare_edges,
        compare_gradient_averages,
        compare_gradient_inverse_covariances,
        sqrt=False):
    centered = compare_edges - edge[np.newaxis, :, :] - gradient_average
    scores = np.einsum(
        "mij,jk,mik->mi",
        centered,
        gradient_inverse_covariance,
        centered,
    )

    centered2 = (
        edge[np.newaxis, :, :]
        - compare_edges
        - compare_gradient_averages[:, np.newaxis, :]
    )
    scores2 = np.einsum(
        "mij,mjk,mik->mi",
        centered2,
        compare_gradient_inverse_covariances,
        centered2,
    )
    scores = np.maximum(scores, 0.0)
    scores2 = np.maximum(scores2, 0.0)
    if sqrt:
        scores = np.sqrt(scores)
        scores2 = np.sqrt(scores2)
    return scores.sum(axis=1) + scores2.sum(axis=1)
