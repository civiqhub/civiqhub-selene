# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""HTTP service wrapper of the SELENE engine: early scaffold, not functional yet.

The service imports ``civiqhub_selene`` and adds transport, authentication, limits and
job tracking around it. It never reimplements a computation: a change to a method is
made in the engine and reaches the local and service modes together.
"""
