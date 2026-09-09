"""Command-line interface for XAKER.

This package exposes the user-facing entry points:

* :mod:`xaker.cli.train`     — :func:`main` for training entry.
* :mod:`xaker.cli.eval`      — :func:`main` for checkpoint evaluation.
* :mod:`xaker.cli.bench`     — :func:`main` for runtime benchmarks.
* :mod:`xaker.cli.validate`  — :func:`main` for rubric grading.

Each ``main`` function is the target of a ``python -m xaker.cli.<name>``
invocation and uses ``argparse`` to parse CLI options (calling
``argparse.ArgumentParser.parse_args`` which calls :class:`SystemExit` on
errors and on ``--help``).
"""

from __future__ import annotations
