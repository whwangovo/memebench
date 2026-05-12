"""Evaluation pipelines - judge responses and calculate scores."""

from .judge import main as judge
from .score import main as score

__all__ = [
    "judge",  # python pipelines/evaluation/judge.py
    "score",  # python pipelines/evaluation/score.py <results_file>
]
