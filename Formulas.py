import numpy as np
from scipy.spatial.distance import cosine, cityblock


def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    :param vector_a: The first vector.
    :param vector_b: The second vector.
    :return: The cosine similarity between the two vectors.
    """
    return 1 - cosine(vector_a, vector_b)  # scipy's cosine returns 1 - cosine similarity


def dice_similarity(vector_a, vector_b):
    """
    Compute the Dice similarity coefficient between two vectors.

    :param vector_a: The first vector.
    :param vector_b: The second vector.
    :return: The Dice similarity coefficient between the two vectors.
    """
    intersection = np.minimum(vector_a, vector_b).sum()
    return 2 * intersection / (np.sum(vector_a) + np.sum(vector_b))


def jaccard_similarity(vector_a, vector_b):
    """
    Calculates the Jaccard similarity between two vectors.

    :param vector_a: First vector.
    :param vector_b: Second vector.
    :return: Jaccard similarity between vector_a and vector_b.

    """
    intersection = np.minimum(vector_a, vector_b).sum()
    union = np.maximum(vector_a, vector_b).sum()
    return intersection / union


def euclidean_distance(vector_a, vector_b):
    """
    Calculate the Euclidean distance between two vectors.

    :param vector_a: A numpy array representing the first vector.
    :param vector_b: A numpy array representing the second vector.
    :return: The Euclidean distance between the two vectors.
    """
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a, vector_b):
    """
    Calculate the Manhattan distance between two vectors.

    :param vector_a: The first vector.
    :param vector_b: The second vector.
    :return: The Manhattan distance between the two vectors.
    """
    return cityblock(vector_a, vector_b)  # using scipy for manhattan distance
