from typing import List, Tuple

import pytest

from nsr.graph.crf_builder import SeqTagCRFBuilder


@pytest.fixture()
def document():
    return [
        ["Nook", "'s", "Cranny", "opens", "today"],
        ["President", "Nook", "gave", "a", "speech"],
        ["Timmy", "and", "Tommy", "Nook", "are", "the", "employees"],
        ["Tommy", "likes", "the", "President"]
    ]


def check_unary_factors(factors: List[Tuple[Tuple[int]]]) -> None:
    assert factors == [
        ((0, 0),), ((0, 1),), ((0, 2),), ((0, 3),), ((0, 4),),
        ((1, 0),), ((1, 1),), ((1, 2),), ((1, 3),), ((1, 4),),
        ((2, 0),), ((2, 1),), ((2, 2),), ((2, 3),), ((2, 4),), ((2, 5),),
        ((2, 6),),
        ((3, 0),), ((3, 1),), ((3, 2),), ((3, 3),)
    ]


def check_transition_factors(factors: List[Tuple[Tuple[int]]]) -> None:
    assert factors == [
        ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 3), (0, 4)),
        ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (1, 4)),
        ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (2, 3)), ((2, 3), (2, 4)),
        ((2, 4), (2, 5)), ((2, 5), (2, 6)),
        ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (3, 3))
    ]


def test_linear_chain_crf(document):
    """Linear chain CRF, with only emission and transition scores.
    """
    crf_builder = SeqTagCRFBuilder(skip_chain_enabled=False)
    graph = crf_builder(document)

    assert len(graph) == 2
    check_unary_factors(graph["unary"])
    check_transition_factors(graph["transition"])


def test_skip_chain_crf(document):
    """CRF with skip chain connections.
    """
    crf_builder = SeqTagCRFBuilder(skip_chain_enabled=True)
    graph = crf_builder(document)

    assert len(graph) == 3
    check_unary_factors(graph["unary"])
    check_transition_factors(graph["transition"])
    
    # Flaky test: In Python 3.6+, dictionary keeps insertion order
    assert graph["skip"] == [
        ((0, 0), (1, 1)), ((0, 0), (2, 3)), ((1, 1), (2, 3)),  # Nook
        ((1, 0), (3, 3)),  # President
        ((2, 2), (3, 0))  # Tommy
    ]
