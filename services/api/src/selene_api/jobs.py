# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Execution and tracking of long computations. Stub, empty for now.

This module will implement the asynchronous job contract of the service: submitting a
computation returns a job identifier; the caller then reads the job state and, once it
has succeeded, retrieves the result. No request is held open for an unbounded time.

The API and the worker may run as two processes of the same image. The choice of a
queue, of storage or of a GPU depends on measurements and is not made by this scaffold.
Stored inputs and results stay isolated per environment and access scope, with an
explicit retention period.
"""
