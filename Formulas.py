import numpy as np
from scipy.spatial.distance import cosine, cityblock


def cosine_similarity(vector_a, vector_b):
    return 1 - cosine(vector_a, vector_b)  # scipy's cosine returns 1 - cosine similarity


def dice_similarity(vector_a, vector_b):
    intersection = np.minimum(vector_a, vector_b).sum()
    return 2 * intersection / (np.sum(vector_a) + np.sum(vector_b))


def jaccard_similarity(vector_a, vector_b):
    intersection = np.minimum(vector_a, vector_b).sum()
    union = np.maximum(vector_a, vector_b).sum()
    return intersection / union


def euclidean_distance(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a, vector_b):
    return cityblock(vector_a, vector_b)  # using scipy for manhattan distance
