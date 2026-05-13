"""Relation-review endpoints under ``/api/runs/{run_id}/relation-reviews``.

Backed by ``<run_dir>/relation_reviews.jsonl``. Latest decision per
``relation_id`` (= species_key) wins. ``added`` decisions carry a full
``relation`` snapshot so user-authored relations survive when the LLM
rerun changes the discovered set.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from ..relation_review_store import (
    RelationReviewStore,
    ReviewDecision as RelationReviewDecision,
    ReviewIn as RelationReviewIn,
)
from ._context import RouterContext


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
