# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Semantic reading of proposal texts. Reserved namespace, empty for now.

This subpackage will relate proposals to one another from their texts, using embeddings
supplied by the caller or an encoder the caller selects explicitly. No model is
downloaded implicitly, and heavy dependencies will live in the ``semantic`` extra. It
does not depend on the ordinal reading.
"""
