from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm.auto import trange


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # jensenshannon returns sqrt of J-S divergence
    return jensenshannon(p, q) ** 2


def cluster_jsd_value(doc_probs: np.ndarray, cluster_prob: np.ndarray) -> float:
    doc_jsd_values = np.apply_along_axis(
        jensen_shannon_divergence, 1, doc_probs, cluster_prob
    )
    return np.mean(doc_jsd_values)


def probabilities(docs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    doc_probs = docs / docs.sum(axis=1, keepdims=True)
    cluster_prob = docs.sum(axis=0) / docs.sum()

    return (doc_probs, cluster_prob)


def random_cluster(all_docs: np.ndarray, cluster_size: int) -> np.ndarray:
    indexes = np.random.choice(len(all_docs), cluster_size)
    return all_docs[indexes]


def jsd_samples(
    all_docs: np.ndarray, cluster_size: int, num_samples: int = 5000
) -> np.ndarray:
    samples = np.empty(num_samples)

    for i in trange(num_samples):
        docs = random_cluster(all_docs, cluster_size)
        doc_probs, cluster_prob = probabilities(docs)
        samples[i] = cluster_jsd_value(doc_probs, cluster_prob)

    return samples


def single_cluster_coherence(
    clu: np.ndarray, all_docs: np.ndarray, num_samples: int = 5000
) -> float:
    # Calculate actual JSD value for cluster
    doc_probs, cluster_prob = probabilities(clu)
    jsd_actual = cluster_jsd_value(doc_probs, cluster_prob)

    # Calculate average JSD value for random clusters of same size
    jsd_rand = np.mean(jsd_samples(all_docs, len(clu), num_samples))

    return jsd_rand - jsd_actual


def coherence(clustering_solution: List[np.ndarray], num_samples: int = 5000) -> float:
    # Basic checks to make sure clustering_solution is in the right format
    for clu in clustering_solution:
        if len(clu.shape) != 2:
            raise ValueError("Clustering solution contains non-2D arrays.")
        if clu.shape[1] != clustering_solution[0].shape[1]:
            raise ValueError("Not all clusters have same amount of terms.")

    all_docs = np.concatenate(clustering_solution)
    return sum(
        single_cluster_coherence(clu, all_docs, num_samples) * len(clu)
        for clu in clustering_solution
    ) / len(all_docs)
