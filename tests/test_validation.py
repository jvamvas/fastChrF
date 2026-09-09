import pytest

from fastchrf import aggregate_chrf, pairwise_chrf

FUNCTIONS = [pairwise_chrf, aggregate_chrf]


@pytest.mark.parametrize("func", FUNCTIONS)
def test_empty_batch_raises(func):
    with pytest.raises(ValueError, match="non-empty"):
        func([], [["a"]])
    with pytest.raises(ValueError, match="non-empty"):
        func([["a"]], [])


@pytest.mark.parametrize("func", FUNCTIONS)
def test_fewer_reference_rows_raises(func):
    # Used to panic with an index-out-of-bounds inside a rayon worker.
    with pytest.raises(ValueError, match="same batch size"):
        func([["a"], ["b"]], [["a"]])


@pytest.mark.parametrize("func", FUNCTIONS)
def test_more_reference_rows_raises(func):
    # Used to silently discard the surplus reference rows.
    with pytest.raises(ValueError, match="same batch size"):
        func([["a"]], [["a"], ["b"], ["c"]])
