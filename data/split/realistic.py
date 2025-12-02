import numpy as np


def split(x, y, N, p, seed=None):

    # Labels as integers
    y = np.int8(y)

    # Create sampling number
    split_values = []
    for val in list(np.unique(y)):
        split_value = round(N * p[val])
        split_values.append(split_value)

    # Retrieve indexes
    a_idx, u_idx = [], []
    for val in list(np.unique(y)):
        idx = np.where(y == val)[0]
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx, )
        a_idx.extend(idx[:split_values[val]])
        u_idx.extend(idx[split_values[val]:])

    # Create partitions
    x_a, y_a, x_u, y_u = x[a_idx], y[a_idx], x[u_idx], y[u_idx]
    return x_a, y_a, x_u, y_u