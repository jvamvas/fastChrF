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
    with pytest.raises(ValueError, match="same batch size"):
        func([["a"], ["b"]], [["a"]])


@pytest.mark.parametrize("func", FUNCTIONS)
def test_more_reference_rows_raises(func):
    with pytest.raises(ValueError, match="same batch size"):
        func([["a"]], [["a"], ["b"], ["c"]])


@pytest.mark.parametrize("func", FUNCTIONS)
def test_zero_char_order_raises(func):
    with pytest.raises(ValueError, match="char_order"):
        func([["a"]], [["a"]], char_order=0)
    with pytest.raises(ValueError, match="char_order"):
        func([["a"]], [["a"]], char_order=0, eps_smoothing=True)
