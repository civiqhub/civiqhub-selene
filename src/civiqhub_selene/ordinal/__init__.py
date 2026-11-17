# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Ordinal reading of collective responses. Standard library only.

This subpackage computes descriptive statistics on ordinal responses to proposals,
with their uncertainty. It works on explicit counts only, without network access or
randomness, and does not depend on the semantic reading.

It holds :class:`VoteCounts`, the five counts of votes of one item; the dissensus
index and the collective regime are added with their first tested implementation.
"""

from civiqhub_selene.ordinal.counts import SCORES, VoteCounts

__all__ = ["SCORES", "VoteCounts"]
