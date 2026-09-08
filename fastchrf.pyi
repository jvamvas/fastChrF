from collections.abc import Sequence
from typing import TypeAlias

_Batch: TypeAlias = Sequence[Sequence[str]]


def pairwise_chrf(
    hypotheses: _Batch,
    references: _Batch,
    char_order: int = 6,
    beta: float = 2.0,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> list[list[list[float]]]:
    """Compute pairwise ChrF scores for each batch of hypotheses and references."""


def aggregate_chrf(
    hypotheses: _Batch,
    references: _Batch,
    char_order: int = 6,
    beta: float = 2.0,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> list[list[float]]:
    """Compute reference-aggregated ChrF scores for each batch of hypotheses."""
