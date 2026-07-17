"""The per-task RFT contract.

Everything that differs between RFT tasks is captured by one
:class:`RftTaskSpec`. The generic engine (:mod:`RFT._common.engine`) and CLI
(:mod:`RFT.cli`) consume a spec and provide the *entire* Azure lifecycle --
``prepare / upload / submit / status / cancel / supervise / deploy / delete /
run / models`` -- so a new version is a ~30-line spec module, not a fresh
launcher.

A task varies in exactly these dimensions:

  * the trainee output schema (:attr:`RftTaskSpec.response_format`);
  * the grader (:attr:`RftTaskSpec.build_grader`, selected by reward variant);
  * the data splits + defaults (model, suffix, hyperparameters);
  * optionally, how to parse + score the deployed model's eval predictions
    (:attr:`RftTaskSpec.prediction_parser` / :attr:`RftTaskSpec.scorer`).

The data splits themselves (``train.jsonl`` / ``valid.jsonl`` / ``eval.jsonl``
with the trainee developer+user messages already baked in) live in
``RFT/<task>/data/`` and are produced by the task's ``build_splits``
script -- the spec only points at the directory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# A grader builder: ``(reward_variant, grader_model) -> Azure grader dict``.
GraderBuilder = Callable[[str, str], dict[str, Any]]
# A trainee output schema builder: ``() -> Azure response_format dict``.
ResponseFormatBuilder = Callable[[], dict[str, Any]]
# Parse one raw trainee reply into the set of predicted (normalised) values.
PredictionParser = Callable[[str | None], "set[str]"]
# Score parsed predictions against the eval rows; returns a metrics dict.
Scorer = Callable[["list[dict[str, Any]]", "list[set[str]]"], dict[str, Any]]


@dataclass(frozen=True, kw_only=True)
class Hparams:
    """RFT ``method.reinforcement.hyperparameters``.

    ``seed`` is omitted from the payload when ``None`` (Azure defaults it).
    """

    n_epochs: int = 2
    batch_size: int = 4
    eval_interval: int = 5
    eval_samples: int = 2
    reasoning_effort: str = "medium"
    learning_rate_multiplier: float = 1.0
    compute_multiplier: float = 1.0
    seed: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "eval_interval": self.eval_interval,
            "eval_samples": self.eval_samples,
            "reasoning_effort": self.reasoning_effort,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "compute_multiplier": self.compute_multiplier,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def with_overrides(self, **kwargs: Any) -> "Hparams":
        """Return a copy with the given fields replaced (drops ``None`` values)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean) if clean else self


@dataclass(frozen=True, kw_only=True)
class RftTaskSpec:
    """Declarative description of one RFT task."""

    # Identity --------------------------------------------------------------
    name: str  # unique registry key, e.g. "typo"
    error_type: str  # taxonomy key, e.g. "typo"

    # Data + defaults -------------------------------------------------------
    data_dir: Path  # holds train/valid/eval jsonl
    base_model: str = "o4-mini-2025-04-16"
    default_suffix: str = ""
    default_grader_model: str = ""
    grader_model_whitelist: tuple[str, ...] = ()
    reward_variants: tuple[str, ...] = ("default",)
    default_reward_variant: str = "default"
    hparams: Hparams = field(default_factory=Hparams)
    train_filename: str = "train.jsonl"
    valid_filename: str = "valid.jsonl"
    eval_filename: str = "eval.jsonl"

    # Version-specific behaviour (the actual fork) --------------------------
    response_format: ResponseFormatBuilder
    build_grader: GraderBuilder

    # Optional eval hooks for the ``run`` verb ------------------------------
    prediction_parser: PredictionParser | None = None
    scorer: Scorer | None = None
    run_reasoning_effort: str = "low"

    def __post_init__(self) -> None:
        if not self.default_suffix:
            object.__setattr__(self, "default_suffix", self.name.replace("_", "-"))
        if self.default_reward_variant not in self.reward_variants:
            object.__setattr__(
                self, "default_reward_variant", self.reward_variants[0]
            )

    @property
    def default_output_dir(self) -> Path:
        return Path("outputs") / f"rft_{self.name}"

    @property
    def grader(self) -> GraderBuilder:  # readable alias
        return self.build_grader


def make_spec(
    *,
    name: str,
    error_type: str,
    data_dir: Path,
    response_format: ResponseFormatBuilder,
    build_grader: GraderBuilder,
    findings_key: str | None = None,
    gold_field: str | None = None,
    ambiguous_field: str | None = None,
    base_model: str = "o4-mini-2025-04-16",
    default_suffix: str = "",
    default_grader_model: str = "gpt-4o-2024-08-06",
    grader_model_whitelist: tuple[str, ...] = (
        "gpt-4o-2024-08-06",
        "o3-mini-2025-01-31",
    ),
    reward_variants: tuple[str, ...] = ("default",),
    default_reward_variant: str = "default",
    hparams: Hparams | None = None,
    run_reasoning_effort: str = "low",
) -> "RftTaskSpec":
    """Build a spec with the run/eval hooks wired from a couple of field names.

    The boilerplate-free path for a new task: give ``findings_key`` (the JSON
    array the trainee emits) to get a prediction parser; give ``gold_field``
    (+ optional ``ambiguous_field``) to also get a cell-level micro-F1 scorer
    for ``rft run``. Omit ``gold_field`` for gold-free eval splits.
    """
    from RFT._common.scoring import (  # local import: scoring has no spec dep
        make_cell_micro_f1_scorer,
        make_json_array_parser,
    )

    parser = make_json_array_parser(findings_key) if findings_key else None
    scorer = (
        make_cell_micro_f1_scorer(gold_field=gold_field, ambiguous_field=ambiguous_field)
        if gold_field
        else None
    )
    return RftTaskSpec(
        name=name,
        error_type=error_type,
        data_dir=data_dir,
        base_model=base_model,
        default_suffix=default_suffix,
        default_grader_model=default_grader_model,
        grader_model_whitelist=grader_model_whitelist,
        reward_variants=reward_variants,
        default_reward_variant=default_reward_variant,
        hparams=hparams or Hparams(),
        response_format=response_format,
        build_grader=build_grader,
        prediction_parser=parser,
        scorer=scorer,
        run_reasoning_effort=run_reasoning_effort,
    )
