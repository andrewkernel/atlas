"""Finite edge and coverage properties for the original preprocessing rules."""
import pytest
from hypothesis import given, settings, strategies as st

from benchmarks.prepare_qa_features import answer_tokens, window_positions, window_starts


@settings(max_examples=40)
@given(length=st.integers(1, 2048), capacity=st.integers(1, 384), fraction=st.integers(1, 384))
def test_every_context_token_remains_covered(length, capacity, fraction):
    advance = min(capacity, fraction)
    starts = window_starts(length, capacity, advance)
    covered = {i for start in starts for i in range(start, min(start + capacity, length))}
    assert covered == set(range(length))
    assert starts[0] == 0 and starts[-1] + capacity >= length


def test_window_labels_refer_to_context_and_closed_span():
    assert window_positions(129, 132, 128, 300, 64) == (67, 70)
    assert window_positions(127, 132, 128, 300, 64) == (0, 0)
    assert window_positions(427, 428, 128, 300, 64) == (0, 0)
    assert window_positions(427, 427, 128, 300, 64) == (365, 365)


def test_character_endpoint_mapping_does_not_count_whitespace_as_tokens():
    offsets = [(0, 1), (2, 6), (7, 10), (11, 15)]
    assert answer_tokens(offsets, 2, 10) == (1, 2)
    with pytest.raises(ValueError):
        answer_tokens(offsets, 1, 2)


def test_advance_and_overlap_are_distinct_quantities():
    assert window_starts(700, 318, 128) == [0, 128, 256, 384]
    assert window_starts(700, 318, 318 - 128) == [0, 190, 380, 570]


@pytest.mark.parametrize("length,capacity,advance", [(0, 12, 3), (5, 0, 1), (5, 2, 3), (5, 2, 0)])
def test_reject_gapped_or_empty_window_recipe(length, capacity, advance):
    with pytest.raises(ValueError):
        window_starts(length, capacity, advance)
