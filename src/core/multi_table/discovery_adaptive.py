"""Adaptive (Good-Turing) multi-table group discovery.

The single-pass confirm step (:func:`resolver.resolve_workbook`) asks the
LLM ONCE per workbook which sheets form a union/join group. This module
repeats that confirm under perturbation - row shuffles and prompt-ensemble
"lenses" - and accumulates the recall-first UNION of discovered groups,
stopping per workbook once the Good-Turing / Chao1 missing-mass estimate
says the undiscovered-group mass is below threshold.

It reuses the generic loop in :func:`src.adaptive.inner_loop.run_adaptive`
(same stop predicate the detection-side adaptive loop uses), so the
saturation semantics are identical across the codebase.

Species
-------
A discovered group's identity for Good-Turing is the COARSE key
``(kind, frozenset(sheets))`` - i.e. *which sheets form what kind of
relationship*. The canonical column-mapping / primary-key are downstream
refinements that can jitter run-to-run, so folding them into the species
would inflate the count and prevent saturation. The final
``ResolvedWorkbook`` keeps, per species, the most complete
``ResolvedGroup`` seen across passes.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.adaptive._chao1 import compute_core
from src.adaptive.inner_loop import run_adaptive
from src.adaptive.perturbations import perturb_frames
from src.adaptive.scoped_gt import ArmConfig

from .resolver import (
    ConfirmPromptConfig,
    ResolvedGroup,
    ResolvedWorkbook,
    resolve_workbook,
)

logger = logging.getLogger(__name__)


# Prompt-ensemble "lenses": appended to the confirm system prompt on
# perturbed passes (run_idx >= 1), cycled by run index. Each nudges the
# LLM toward a different slice of the relationship space so successive
# passes surface groups a single framing might miss. Run 0 is always the
# unperturbed identity pass (matches the single-pass baseline).
_DISCOVERY_PROMPT_LENSES: Tuple[str, ...] = (
    "ADDITIONAL LENS: Look hard for UNION groups whose sheets hold the "
    "same entity even when their headers differ a lot (renamed, "
    "abbreviated, reordered, or unit-suffixed columns). If a real group is "
    "missing from the candidate list, ADD it with a fresh group_id.",
    "ADDITIONAL LENS: Look hard for JOIN / foreign-key relationships you "
    "might overlook - a small lookup / reference / dimension sheet feeding "
    "a larger transactional sheet. ADD a new join group (fresh group_id) "
    "when warranted.",
    "ADDITIONAL LENS: Re-examine sheets you would normally skip (very "
    "small, very wide, or oddly named) - they can still be union/join "
    "members. ADD new groups for any clearly related pair not already "
    "listed as a candidate.",
)


@dataclass(frozen=True)
class AdaptiveDiscoveryConfig:
    """Tunable knobs for adaptive multi-table discovery.

    ``k_min`` / ``k_max`` / ``stop_when`` and the two perturbation
    toggles are surfaced to the launch UI; the epsilons and
    ``require_two_consecutive`` fall back to sensible defaults.
    """

    enabled: bool = False
    k_min: int = 2
    k_max: int = 6
    stop_when: str = "gt"            # "gt" | "chao1" | "both"
    epsilon: float = 0.05
    epsilon_chao1: float = 0.05
    require_two_consecutive: bool = False
    shuffle_rows: bool = True
    prompt_ensemble: bool = True
    base_seed: int = 1234

    def to_arm_config(self) -> ArmConfig:
        """Resolve into a fully-populated :class:`ArmConfig`."""
        return ArmConfig(
            epsilon=float(self.epsilon),
            epsilon_chao1=float(self.epsilon_chao1),
            stop_when=str(self.stop_when),
            k_min=int(self.k_min),
            k_max=int(self.k_max),
            require_two_consecutive=bool(self.require_two_consecutive),
        ).merge_into(ArmConfig.default())

    @classmethod
    def from_adaptive_settings(
        cls, settings: Optional[Mapping[str, Any]],
    ) -> "AdaptiveDiscoveryConfig":
        """Derive the discovery config from the global experiment adaptive
        settings.

        There is no separate multi-table-discovery toggle: a single
        ``experiment.mode=adaptive`` drives the WHOLE run, multi-table
        discovery included. ``k_min`` / ``k_max`` come from the global
        ``min_runs`` / ``num_runs``, the epsilons from
        ``target_missing_mass`` (or the per-arm config), and the
        perturbation toggles + seed from the same Good-Turing settings the
        detection loop uses. Returns a disabled config when ``settings`` is
        absent or not in adaptive mode (i.e. ``mode=single``).
        """
        if not settings or str(settings.get("gt_mode")) != "adaptive":
            return cls(enabled=False)
        arm = None
        per_arm = settings.get("per_arm")
        if isinstance(per_arm, tuple) and per_arm:
            arm = per_arm[0]
        eps = float(settings.get("target_missing_mass", 0.05) or 0.05)
        return cls(
            enabled=True,
            k_min=int(settings.get("min_runs", 2) or 2),
            k_max=int(settings.get("num_runs", 6) or 6),
            stop_when=str(getattr(arm, "stop_when", None) or "gt"),
            epsilon=float(getattr(arm, "epsilon", None) or eps),
            epsilon_chao1=float(getattr(arm, "epsilon_chao1", None) or eps),
            require_two_consecutive=bool(
                getattr(arm, "require_two_consecutive", False)
            ),
            shuffle_rows=bool(settings.get("shuffle_rows", True)),
            prompt_ensemble=bool(settings.get("prompt_ensemble", True)),
            base_seed=int(settings.get("base_seed") or 1234),
        )


def group_species_key(group: ResolvedGroup) -> Tuple[str, frozenset]:
    """Coarse, perturbation-stable identity of a confirmed group."""
    return (group.kind, frozenset(group.sheets))


def _mapping_size(group: ResolvedGroup) -> int:
    return sum(len(v) for v in (group.column_mapping or {}).values())


def resolve_workbook_adaptive(
    *,
    workbook: str,
    sheets: Sequence[str],
    frames: Mapping[str, pd.DataFrame],
    digests: Mapping[str, Any],
    llm: Any,
    model: str,
    prompt_cfg: ConfirmPromptConfig,
    cfg: AdaptiveDiscoveryConfig,
    fk_candidate_cap: int = 12,
    speculative_max_sheets: int = 8,
) -> Tuple[ResolvedWorkbook, Dict[str, Any]]:
    """Run adaptive confirm for one workbook.

    Returns ``(resolved, stats)`` where ``resolved`` is a
    :class:`ResolvedWorkbook` carrying the recall-first union of
    representative groups (and run-0 priors/metadata so step 3 can still
    render priors), and ``stats`` is the per-workbook Good-Turing
    summary.
    """
    # Only this workbook's sheets are needed; perturbing the full corpus
    # frame map every pass would be wasteful.
    wb_frames: Dict[str, pd.DataFrame] = {
        s: frames[s] for s in sheets if s in frames
    }

    best_by_species: Dict[Tuple[str, frozenset], ResolvedGroup] = {}
    base_holder: Dict[str, ResolvedWorkbook] = {}

    def _sample(run_idx: int) -> List[Tuple[str, frozenset]]:
        # Perturb row order on passes >= 1 (run 0 = identity baseline).
        if cfg.shuffle_rows and run_idx > 0 and wb_frames:
            views = perturb_frames(
                wb_frames, run_id=run_idx, base_seed=cfg.base_seed,
                shuffle_rows=True, shuffle_columns=False,
            )
            run_frames: Mapping[str, pd.DataFrame] = {
                v.source: v.frame for v in views
            }
        else:
            run_frames = wb_frames

        run_prompt = prompt_cfg
        if cfg.prompt_ensemble and run_idx > 0 and _DISCOVERY_PROMPT_LENSES:
            lens = _DISCOVERY_PROMPT_LENSES[
                (run_idx - 1) % len(_DISCOVERY_PROMPT_LENSES)
            ]
            run_prompt = dataclasses.replace(
                prompt_cfg,
                system_prompt=prompt_cfg.system_prompt + "\n\n" + lens,
            )

        rw = resolve_workbook(
            workbook=workbook,
            sheets=sheets,
            frames=run_frames,
            digests=digests,
            llm=llm,
            model=model,
            prompt_cfg=run_prompt,
            fk_candidate_cap=fk_candidate_cap,
            speculative_max_sheets=speculative_max_sheets,
        )
        if run_idx == 0:
            base_holder["rw"] = rw

        species: List[Tuple[str, frozenset]] = []
        for g in rw.groups:
            if not g.confirmed:
                continue
            sp = group_species_key(g)
            species.append(sp)
            prev = best_by_species.get(sp)
            if prev is None or _mapping_size(g) > _mapping_size(prev):
                best_by_species[sp] = g
        return species

    result = run_adaptive(
        arm_id=workbook, sample_fn=_sample, cfg=cfg.to_arm_config(),
    )

    # Assemble the final workbook from run-0 metadata + the representative
    # union of groups. Run 0 always executes (k_min >= 1), so base exists.
    base = base_holder.get("rw") or ResolvedWorkbook(
        workbook=workbook, sheets=tuple(sheets),
    )
    reps = list(best_by_species.values())
    base.groups = reps
    if reps and base.error:
        # We did discover groups on a later pass; don't surface run-0's
        # transient error as the workbook verdict.
        base.error = None

    n_passes = len(result.species_per_run)
    freqs = [len(runs) for runs in result.species_to_runs.values()]
    core = compute_core(freqs, num_runs=max(1, n_passes))
    stats: Dict[str, Any] = {
        "workbook": workbook,
        "n_passes": n_passes,
        "n_groups": len(reps),
        "n_confirmed_species": core.n_observed,
        "good_turing_missing_mass": core.good_turing_missing_mass,
        "chao1_estimated_total": core.chao1_estimated_total,
        "chao1_estimated_missed": core.chao1_estimated_missed,
        "stopped_reason": result.stopped_reason,
    }
    return base, stats


def summarize_discovery(per_workbook: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll per-workbook GT stats into a run-level summary dict."""
    if not per_workbook:
        return {
            "n_workbooks": 0,
            "total_passes": 0,
            "mean_missing_mass": 0.0,
            "stopped_reasons": {},
        }
    total_passes = sum(int(s.get("n_passes", 0)) for s in per_workbook)
    masses = [float(s.get("good_turing_missing_mass", 0.0)) for s in per_workbook]
    reasons: Dict[str, int] = {}
    for s in per_workbook:
        r = str(s.get("stopped_reason", "k_max"))
        reasons[r] = reasons.get(r, 0) + 1
    return {
        "n_workbooks": len(per_workbook),
        "total_passes": total_passes,
        "mean_passes_per_workbook": total_passes / len(per_workbook),
        "mean_missing_mass": sum(masses) / len(masses),
        "max_missing_mass": max(masses) if masses else 0.0,
        "stopped_reasons": reasons,
    }
