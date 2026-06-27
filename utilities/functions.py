import numpy as np


# Vector normalization
def norm(vector):
    vector = np.asarray(vector, dtype=float)
    return np.linalg.norm(vector)

# Unit vector
def unit(vector):
    vector = np.asarray(vector, dtype=float)
    vector_norm = norm(vector)

    # If the norm is too small, returns a zero vector to avoid division errors
    if vector_norm < 1e-12:
        return np.zeros_like(vector)

    return vector / vector_norm

# Saturation function to limit the magnitude of a vector
def saturate(vector, max_norm=np.inf):
    vector = np.asarray(vector, dtype=float)

    # If max_norm is None or infinity, there is no saturation
    if max_norm is None or np.isinf(max_norm):
        return vector

    vector_norm = norm(vector)

    # If the norm is too small, returns a zero vector to avoid division errors
    if vector_norm < 1e-12:
        return np.zeros_like(vector)

    if vector_norm > max_norm:
        return vector * max_norm / vector_norm

    return vector