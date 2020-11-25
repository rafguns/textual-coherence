import numpy as np
import pytest

import textual_coherence


@pytest.fixture
def single_cluster():
    """Returns probability tuple for single cluster"""
    # 3 documents, 7 words
    docs = np.array(
        [
            [0, 0, 1, 0, 5, 2, 0, 0, 0],
            [1, 0, 2, 3, 1, 0, 0, 0, 0],
            [1, 1, 0, 4, 0, 0, 0, 2, 0],
        ]
    )
    return textual_coherence.probabilities(docs)


def test_cluster_jsd_value(single_cluster):
    assert textual_coherence.cluster_jsd_value(
        single_cluster[0], single_cluster[1]
    ) == pytest.approx(0.18599437)


def test_coherence_inconsistent_number_terms():
    clustering_solution_inconsistent_terms = [
        np.array([[0, 1, 0], [1, 2, 0], [7, 8, 0]]),
        np.array([[3, 4], [5, 6]]),
    ]
    with pytest.raises(ValueError):
        textual_coherence.coherence(clustering_solution_inconsistent_terms)


def test_coherence_more_than_2_dimensions():
    clustering_solution_wrong_dimensions = [
        np.array([[[0, 1], [1, 2]], [[7, 8], [9, 10]]]),
        np.array([[3, 4], [5, 6]]),
    ]
    with pytest.raises(ValueError):
        textual_coherence.coherence(clustering_solution_wrong_dimensions)
