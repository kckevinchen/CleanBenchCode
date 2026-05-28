"""Relation-review endpoints under ``/api/runs/{run_id}/relation-reviews``.

Backed by ``<run_dir>/relation_reviews.jsonl``. Latest decision per
``relation_id`` (= species_key) wins. ``added`` decisions carry a full
``relation`` snapshot so user-authored relations survive when the LLM
rerun changes the discovered set.

Also exposes ``/api/runs/{run_id}/relations`` — a read-only listing of
the discovered relations joined with the latest review verdict per
species. The review UI uses it as its primary queue.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from ..relation_review_store import (
    RelationReviewStore,
    ReviewDecision as RelationReviewDecision,
    ReviewIn as RelationReviewIn,
)
from ._context import RouterContext


_LAYER_FILES = [
    ("aggregated", "relations_aggregated.jsonl"),
    ("originals", "relations.jsonl"),
]


def _read_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001 - skip malformed lines
                continue
    return out


def _read_verifications(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Index ``relation_verifications.jsonl`` by ``species_key`` so the
    UI can show pass-rate / verifier-kind alongside each relation."""
    path = run_dir / "relation_verifications.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    for rec in _read_jsonl_safe(path):
        sk = rec.get("species_key") or rec.get("relation_species_key")
        if sk:
            out[str(sk)] = rec
    return out


def register(app: FastAPI, ctx: RouterContext) -> None:
    cache = ctx.cache

    # Per-app store cache, identical to the closure version in the
    # pre-split create_app.
    stores: Dict[str, RelationReviewStore] = {}
    lock = threading.Lock()

    def _store_for(run_id: str) -> RelationReviewStore:
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")
        with lock:
            store = stores.get(run_id)
            if store is None:
                store = RelationReviewStore(rd / "relation_reviews.jsonl")
                stores[run_id] = store
            return store

    @app.get("/api/runs/{run_id}/relation-reviews")
    def list_relation_reviews(run_id: str) -> Dict[str, Any]:
        store = _store_for(run_id)
        records = [r.model_dump(mode="json") for r in store.all()]
        latest = {
            rid: rec.model_dump(mode="json")
            for rid, rec in store.latest_by_relation().items()
        }
        return {"records": records, "latest": latest}

    @app.get("/api/runs/{run_id}/relations")
    def list_relations(
        run_id: str,
        layer: str = "auto",
        error_type: Optional[str] = None,
        verdict: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the discovered relations joined with their latest review
        verdict and (when present) validator pass-rate.

        Query params:
          * ``layer`` — ``aggregated`` | ``originals`` | ``auto`` (default;
            picks the highest layer that exists).
          * ``error_type`` — filter to a single error_type key.
          * ``verdict`` — keep only relations whose latest decision matches
            (``approved`` | ``rejected`` | ``added`` | ``cleared`` |
            ``unreviewed``).
        """
        rd = cache._safe_run_dir(run_id)
        if not rd.is_dir():
            raise HTTPException(404, f"run not found: {run_id}")

        # Resolve layer.
        chosen_layer: Optional[str] = None
        relations_path: Optional[Path] = None
        if layer == "auto":
            for name, fname in _LAYER_FILES:
                p = rd / fname
                if p.is_file():
                    chosen_layer, relations_path = name, p
                    break
        else:
            wanted = {n: f for n, f in _LAYER_FILES}
            fname = wanted.get(layer)
            if not fname:
                raise HTTPException(
                    400, f"invalid layer={layer!r}; want one of "
                    f"{[n for n, _ in _LAYER_FILES] + ['auto']}",
                )
            p = rd / fname
            if p.is_file():
                chosen_layer, relations_path = layer, p
        if not relations_path:
            return {"layer": None, "relations": [], "counts": {}}

        relations = _read_jsonl_safe(relations_path)
        store = _store_for(run_id)
        latest = store.latest_by_relation()
        verifications = _read_verifications(rd)

        out: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for r in relations:
            sk = str(r.get("species_key") or "")
            rec = latest.get(sk)
            v = verifications.get(sk) or {}
            decision = rec.decision.value if rec else "unreviewed"
            if verdict and decision != verdict:
                continue
            if error_type and r.get("error_type") != error_type:
                continue
            counts[decision] = counts.get(decision, 0) + 1
            out.append({
                "relation": r,
                "review": rec.model_dump(mode="json") if rec else None,
                "verdict": decision,
                "verification": {
                    "verifier_kind": v.get("verifier_kind"),
                    "pass_rate": v.get("pass_rate"),
                    "verifiable": v.get("verifiable"),
                    "refusal_reason": v.get("refusal_reason"),
                    "sample_violators": v.get("sample_violators"),
                } if v else None,
            })

        # Surface ``added`` records that aren't in the relations file
        # (operator-introduced rules with no source LLM proposal).
        existing_keys = {r.get("relation", {}).get("species_key") for r in out}
        for rec in store.added_relations():
            if rec.relation_id in existing_keys:
                continue
            if verdict and verdict != "added":
                continue
            snap = (rec.relation.model_dump(mode="json") if rec.relation else None) or {}
            if error_type and snap.get("error_type") != error_type:
                continue
            counts["added"] = counts.get("added", 0) + 1
            out.append({
                "relation": {**snap, "species_key": rec.relation_id, "added": True},
                "review": rec.model_dump(mode="json"),
                "verdict": "added",
                "verification": None,
            })

        return {
            "layer": chosen_layer,
            "counts": counts,
            "relations": out,
        }

    @app.post("/api/runs/{run_id}/relation-reviews")
    def post_relation_review(
        run_id: str,
        payload: RelationReviewIn,
    ) -> Dict[str, Any]:
        if payload.decision == RelationReviewDecision.ADDED:
            if payload.relation is None:
                raise HTTPException(
                    400,
                    "decision=added requires a 'relation' snapshot in the body",
                )
            if not payload.relation.kind:
                raise HTTPException(400, "relation.kind is required")
            if not payload.relation.columns:
                raise HTTPException(400, "relation.columns must be non-empty")
        store = _store_for(run_id)
        record = store.append(payload)
        return {
            "record": record.model_dump(mode="json"),
            "latest": {
                rid: rec.model_dump(mode="json")
                for rid, rec in store.latest_by_relation().items()
            },
        }
