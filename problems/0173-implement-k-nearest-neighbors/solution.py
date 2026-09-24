import numpy as np

def k_nearest_neighbors(points, query_point, k):
    points = np.array(points)
    raw = points - query_point
    diff = np.sqrt(np.sum(raw ** 2, axis = 1))
    order = np.argsort(diff, kind = "stable")
    return list(map(tuple, points[order][0:k]))
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    pass