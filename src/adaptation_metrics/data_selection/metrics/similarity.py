"""Similiarity metrics for data selection introduced by Ruder and Plank.

The functions here were adapted and vectorized
from those in the authors' `repo <https://github.com/sebastianruder/learn-to-select-data/blob/master/similarity.py>`_.
"""
from typing import Callable, Dict, Literal

import numpy as np
import pandas as pd

import scipy.spatial.distance
import scipy.stats


# SIMILARITY MEASURES
def _check_input_sanity(repr1:np.ndarray, repr2:np.ndarray):
    """Check that the input representations are not empty."""
    if repr1 is None or repr2 is None:
        raise ValueError("Input representations must not be None.")
    if pd.isna(repr1).any() or pd.isna(repr2).any():
        raise ValueError("Input representations must not be NaN.")
    if np.isinf(repr1).any() or np.isinf(repr2).any():
        raise ValueError("Input representations must not be inf.")
    if repr1.size == 0 or repr2.size == 0:
        raise ValueError("Input representations must not be empty.")
    if repr1.size == 1 and repr2.size > 1:
        repr1 = np.repeat(repr1, repr2.size, axis=0)
    elif repr2.size == 1 and repr1.size > 1:
        repr2 = np.repeat(repr2, repr1.size, axis=0)
    return repr1, repr2


def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def renyi_divergence(repr1, repr2, alpha=0.99):
    """Calculates Renyi divergence (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence)."""
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
    sim = 1 / (alpha - 1) * np.log(log_sum)
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim


def cosine_similarity(repr1, repr2):
    """Calculates cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    sim = 1 - scipy.spatial.distance.cosine(repr1, repr2)
    if np.isnan(sim):
        # the similarity is nan if no term in the document is in the vocabulary
        return 0
    return sim


def euclidean_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate similarity based on Euclidean distance.

    https://en.wikipedia.org/wiki/Euclidean_distance
    """
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
    return sim


def manhattan(repr1, repr2):
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
    return sim

def kl_divergence(repr1, repr2):
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    """Calculates Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)."""
    sim = scipy.stats.entropy(repr1, repr2)
    return sim

def bhattacharyya_similarity(repr1, repr2):
    repr1, repr2 = _check_input_sanity(repr1, repr2)
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim

############################
##### Function factory #####
############################
SimilarityMetric = Literal[
    "jensen-shannon",
    "renyi",
    "cosine",
    "euclidean",
    "variational",
    "bhattacharyya",
    "entropy",
]
SimilarityFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
SIMILARITY_FEATURES = {
    "jensen-shannon",
    "renyi",
    "cosine",
    "euclidean",
    "variational",
    "bhattacharyya",
    "entropy",
}


def similarity_func_factory(metric: SimilarityMetric) -> SimilarityFunc:
    """Return the corresponding similarity function based on the provided metric.

    Args:
        metric (str): Similarity metric

    Raises:
        ValueError: If `metric` does not exist in SIMILARITY_FEATURES
    """
    if metric not in SIMILARITY_FEATURES:
        raise ValueError(f'"{metric}" is not a valid similarity metric.')

    mapping: Dict[SimilarityMetric, SimilarityFunc] = {
        "jensen-shannon": jensen_shannon_divergence,
        "renyi": renyi_divergence,
        "cosine": cosine_similarity,
        "euclidean": euclidean_similarity,
        "variational": manhattan,
        "bhattacharyya": bhattacharyya_similarity,
        "entropy": kl_divergence,
    }

    similarity_function = mapping[metric]
    return similarity_function
