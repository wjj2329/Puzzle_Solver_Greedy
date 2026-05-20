import numpy as np


def euclideanDistance(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff.reshape(diff.shape[0], -1)
    return float(np.linalg.norm(diff, axis=1).sum())


def mahalanobisEdgeDistance(own_edge, compare_edge):
    matrix = (own_edge.edge - compare_edge.edge) - own_edge.average_delta
    matrix2 = (compare_edge.edge - own_edge.edge) - compare_edge.average_delta

    scores = np.einsum(
        "ij,jk,ik->i", matrix, compare_edge.inverse_covariance, matrix)
    scores2 = np.einsum(
        "ij,jk,ik->i", matrix2, own_edge.inverse_covariance, matrix2)
    return float(np.sqrt(np.abs(scores)).sum() + np.sqrt(np.abs(scores2)).sum())
