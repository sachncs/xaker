"""Paper-worthiness rubric for XAKER.

Six dimensions (each scored 0-3):

- :class:`Novelty` — distinct method beyond baseline.
- :class:`Repro` — reproducible from code alone.
- :class:`Correctness` — invariant tests cover math.
- :class:`Efficiency` — benchmarked vs baselines.
- :class:`Stability` — stable across seeds/dtypes/lengths.
- :class:`Usability` — clean CLI/API/docs.

Total max = 18. Pass threshold = 14 with no dim < 2 (novelty may be 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass(frozen=True)
class Score:
    """Score for a single rubric dimension (0-3)."""

    value: int
    evidence: str

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 3:
            raise ValueError(f"Score value must be in [0, 3]; got {self.value}")


@dataclass(frozen=True)
class Dimension:
    """One rubric dimension."""

    name: str
    score: Score
    weight: int = 1


@dataclass(frozen=True)
class Rubric:
    """Aggregate of all dimensions."""

    dims: Dict[str, Dimension]

    @property
    def total(self) -> int:
        return sum(d.score.value * d.weight for d in self.dims.values())

    @property
    def passed(self) -> bool:
        if self.total < 14:
            return False
        for d in self.dims.values():
            if d.name != "novelty" and d.score.value < 2:
                return False
        return True


class Grader(Protocol):
    """Protocol for individual dimension graders."""

    name: str

    def grade(self, repo_root: str) -> Dimension: ...