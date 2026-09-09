"""Typed bench driver.

Public surface:
- :class:`Spec`, :class:`Result`, :class:`Metrics` — dataclasses.
- :func:`tick`, :func:`peak`, :func:`converge` — measurement helpers.
- :func:`run` — main driver.
- :func:`write` — JSON serialization.
- :func:`gitsha` — current git HEAD SHA.
"""

from xaker.bench.bench import (
    Metrics,
    Result,
    Spec,
    converge,
    gitsha,
    peak,
    run,
    tick,
    write,
)

__all__ = [
    "Metrics",
    "Result",
    "Spec",
    "converge",
    "gitsha",
    "peak",
    "run",
    "tick",
    "write",
]
