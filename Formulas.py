import numpy as np
from scipy.spatial.distance import cosine, cityblock

def cosine_similarity(vectorA, vectorB):
    return 1 - cosine(vectorA, vectorB)  # scipy's cosine returns 1 - cosine similarity

def dice_similarity(vectorA, vectorB):
    intersection = np.minimum(vectorA, vectorB).sum()
    return 2 * intersection / (np.sum(vectorA) + np.sum(vectorB))

def jaccard_similarity(vectorA, vectorB):
    intersection = np.minimum(vectorA, vectorB).sum()
    union = np.maximum(vectorA, vectorB).sum()
    return intersection / union

def euclidean_distance(vectorA, vectorB):
    return np.linalg.norm(vectorA - vectorB)

def manhattan_distance(vectorA, vectorB):
    return cityblock(vectorA, vectorB)  # using scipy for manhattan distance
