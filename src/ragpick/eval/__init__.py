"""Evaluation harness: datasets, metrics, and the runner."""

from ragpick.eval.dataset import EvalExample, load_dataset
from ragpick.eval.metrics import JudgeScore, LLMJudge
from ragpick.eval.runner import EvalReport, ExampleResult, evaluate, evaluate_configs, write_reports

__all__ = [
    "EvalExample",
    "EvalReport",
    "ExampleResult",
    "JudgeScore",
    "LLMJudge",
    "evaluate",
    "evaluate_configs",
    "load_dataset",
    "write_reports",
]
