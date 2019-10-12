
# Reference implementation of the floyd warshall aglorithim. Useful for generating test cases for the parallel
# implementation.

from itertools import chain
from math import log10, floor, inf, isinf

def floyd_warshall_solve(adjacency_matrix):
    """Solve in-place the all-pairs shortest path for a 2D adjacency matrix."""
    n_vertices = len(adjacency_matrix)
    for k in range(n_vertices):
        for i in range(n_vertices):
            for j in range(n_vertices):
                if adjacency_matrix[i][j] > adjacency_matrix[i][k] + adjacency_matrix[k][j]:
                    adjacency_matrix[i][j] = adjacency_matrix[i][k] + adjacency_matrix[k][j]
    return adjacency_matrix

def make_adjacency_matrix(values, n):
    """Make a 2D adjacency matrix of dimension `n` from the 1D values vector."""
    return [values[i:i+n] for i in range(0, n ** 2, n)]

def print_adjacency_matrix(adjacency_matrix):
    """Pretty print an adjacency matrix."""
    max_distance = max(1 if isinf(d) else d for d in chain.from_iterable(adjacency_matrix))
    max_distance_digits = int(floor(log10(max_distance)) + 1)
    print("\n".join(
        " ".join("∞" if isinf(d) else f"{{:{max_distance_digits}}}".format(d) for d in row)
        for row in adjacency_matrix
    ))

print_adjacency_matrix(floyd_warshall_solve(make_adjacency_matrix(
    [0, 1, 8, inf, 7,
     inf, 0, inf, 2, 2,
     inf, inf, 0, inf, 1,
     inf, inf, 3, 0, 9,
     6, inf, 3, 0, 9],
    5
)))
