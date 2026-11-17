# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Pure functions on arrays shared by the co-clustering modules."""

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray, IntArray

__all__ = ["count_states", "logsumexp", "one_hot"]


def one_hot(labels: IntArray, n: int) -> FloatArray:
    """Return the indicator matrix of ``labels``, of shape (len(labels), n)."""
    indicator: FloatArray = np.zeros((labels.shape[0], n), dtype=np.float64)
    indicator[np.arange(labels.shape[0]), labels] = 1.0
    return indicator


def count_states(
    states: FloatArray, rows: IntArray, cols: IntArray, shape: tuple[int, int]
) -> FloatArray:
    """Count the cells of each co-cluster in each state, for hard labels.

    ``counts[q, l, s]`` is the number of cells of the co-cluster ``(q, l)`` in the
    state ``s``. It is the sufficient statistic of every block model.

    Args:
        states: Indicator of the state of each cell, of shape (M, P, S).
        rows: Label of each row, of shape (M,), in ``0..Q-1``.
        cols: Label of each column, of shape (P,), in ``0..L-1``.
        shape: ``(Q, L)``.

    Returns:
        An array of shape (Q, L, S).
    """
    n_groups, n_blocks = shape
    counts: FloatArray = np.einsum(
        "iq,jl,ijs->qls", one_hot(rows, n_groups), one_hot(cols, n_blocks), states
    )
    return counts


def logsumexp(a: FloatArray, axis: int) -> FloatArray:
    """Return ``log(sum(exp(a)))`` along ``axis``, computed without overflow.

    A slice that holds only ``-inf`` gives ``-inf``.
    """
    highest = np.max(a, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(highest), highest, 0.0)
    with np.errstate(divide="ignore"):
        total: FloatArray = np.log(np.sum(np.exp(a - shift), axis=axis))
    total += np.squeeze(shift, axis=axis)
    return total
