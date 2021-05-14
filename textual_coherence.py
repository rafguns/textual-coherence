"""
This is an implementation of textual coherence for the evaluation of cluster quality,
as previously used by Boyack and colleagues (`study 1`_, `study 2`_).

.. _study 1: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0018029#s2
.. _study 2: https://onlinelibrary.wiley.com/doi/pdf/10.1002/asi.21419

"""
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm.auto import trange


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Jensen-Shannon divergence between vectors p and q"""
    # jensenshannon returns sqrt of J-S divergence
    return jensenshannon(p, q) ** 2


def cluster_jsd_value(doc_probs: np.ndarray, cluster_prob: np.ndarray) -> float:
    """Calculate JSD value for cluster

    Here, `doc_probs` is an n × m array (n documents, m words), where each element
    gives the probability of a word in a document. `cluster_prob` is a vector of
    length m, giving the probability of a word in the whole cluster. Use function
    `probabilities` to calculate these for a document-word array.

    """
    doc_jsd_values = np.apply_along_axis(
        jensen_shannon_divergence, 1, doc_probs, cluster_prob
    )
    return np.mean(doc_jsd_values)


def probabilities(docs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Document probabilities and cluster probability of words in array `docs`

    Given a document-word array, this calculates two probabilities:
    * the probability of a word in a document as an n × m array (n documents, m
      words)
    * the probability of a word in the whole cluster as a vector of length m

    """
    doc_probs = docs / docs.sum(axis=1, keepdims=True)
    cluster_prob = docs.sum(axis=0) / docs.sum()

    return (doc_probs, cluster_prob)


def jsd_samples(
    all_docs: np.ndarray, cluster_size: int, num_samples: int = 5000, seed=None
) -> np.ndarray:
    """Determine JSD for random clusters of given size"""
    if seed is None:
        seed = 5  # Hack but needed to make the two RNGs below work

    rng = np.random.default_rng(seed=seed)
    all_doc_probs = all_docs / all_docs.sum(axis=1, keepdims=True)
    idx = np.arange(len(all_docs))

    idx_samples = rng.choice(idx, size=(num_samples, cluster_size))

    doc_samples = all_docs[idx_samples]
    doc_probs_samples = all_doc_probs[idx_samples]
    cluster_prob_samples = doc_samples.sum(axis=1) / doc_samples.sum(axis=(2)).sum(
        axis=1, keepdims=True
    )

    return np.array(
        [
            cluster_jsd_value(doc_probs, cluster_prob)
            for doc_probs, cluster_prob in zip(doc_probs_samples, cluster_prob_samples)
        ]
    )


def single_cluster_coherence(
    clu: np.ndarray, all_docs: np.ndarray, num_samples: int = 5000, seed=None
) -> float:
    """Normalized coherence of a single cluster `clu`

    Coherence is affected by cluster size, and needs to be normalized.
    Normalization happens by comparing the coherence of `clu` to that of random
    clusters of similar size. More specifically, we calculate the average JSD
    value for `num_samples` random clusters of the same size.

    """
    # Calculate actual JSD value for cluster
    doc_probs, cluster_prob = probabilities(clu)
    jsd_actual = cluster_jsd_value(doc_probs, cluster_prob)

    # Calculate average JSD value for random clusters of same size
    jsd_rand = np.mean(jsd_samples(all_docs, len(clu), num_samples, seed))

    return jsd_rand - jsd_actual


def coherence(
    clustering_solution: List[np.ndarray], num_samples: int = 5000, seed=None
) -> float:
    r"""Average coherence for entire cluster solution

    This is calculated as a weighted average of coherences over all clusters

    .. math::

        Coh = \frac{\sum{n_i Coh_i}}{\sum{n_i}}

    where $n_i$ is the size of cluster $i$.

    """
    # Basic checks to make sure clustering_solution is in the right format
    for clu in clustering_solution:
        if len(clu.shape) != 2:
            raise ValueError("Clustering solution contains non-2D arrays.")
        if clu.shape[1] != clustering_solution[0].shape[1]:
            raise ValueError("Not all clusters have same amount of terms.")

    all_docs = np.concatenate(clustering_solution)
    return sum(
        single_cluster_coherence(clu, all_docs, num_samples, seed) * len(clu)
        for clu in clustering_solution
    ) / len(all_docs)
