"""Job launcher endpoints under ``/api/launcher/``.

Owns the experiment job lifecycle (start, list, status, log tail,
cancel, delete) plus the ``/api/launcher/config`` defaults bundle
the launcher UI uses to populate its forms.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, HTTPException, Query

from ..runs import list_datasets, load_config_defaults, load_taxonomy_keys
from ._context import RouterContext

logger = logging.getLogger(__name__)


def register(app: FastAPI, ctx: RouterContext) -> None:
    project_root = ctx.project_root
    jobs = ctx.jobs

    @app.get("/api/launcher/config")
    def launcher_config() -> Dict[str, Any]:
        defaults = load_config_defaults(project_root)
        # Relation-discovery prompt: surface the subset of error types
        # for which we have a discovery template, so the UI can filter
        # the chip list down to multi-column / multi-row / multi-table
        # types only.
        relations_types: List[str] = []
        relations_type_scopes: Dict[str, str] = {}
        try:
            disc_path = project_root / "prompts" / "discover_relations.yaml"
            if disc_path.is_file():
                raw = yaml.safe_load(disc_path.read_text(encoding="utf-8")) or {}
                types_node = raw.get("types") or {}
                if isinstance(types_node, dict):
                    relations_types = sorted(types_node.keys())
                    for et_key, et_node in types_node.items():
                        if isinstance(et_node, dict):
                            scope = str(et_node.get("scope") or "row")
                            relations_type_scopes[et_key] = scope
        except Exception:  # noqa: BLE001
            relations_types = []
            relations_type_scopes = {}
        return {
            "project_root": str(project_root),
            "defaults": defaults,
            "error_types": load_taxonomy_keys(project_root),
            "relations_error_types": relations_types,
            "relations_error_type_scopes": relations_type_scopes,
            "datasets": list_datasets(project_root / "data"),
            "scopes": ["single-column", "multi-column", "multi-row", "multi-table"],
            "column_modes": ["auto", "single", "all"],
            "modes": ["single", "adaptive"],
            "reasoning_efforts": ["low", "medium", "high"],
            "gt_modes": ["fixed", "adaptive"],
        }

    @app.get("/api/launcher/jobs")
    def list_jobs() -> Dict[str, Any]:
        return {"jobs": [j.to_dict() for j in jobs.list()]}

    @app.post("/api/launcher/jobs")
    def launch_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        kind = (payload.get("kind") or "detection").lower()
        params = payload.get("params") or {}
        try:
            if kind == "detection":
                job = jobs.launch_detection(params)
            elif kind == "report":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=report")
                try:
                    job = jobs.launch_report(run_id)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind in ("curate", "curate_tables", "curate_pbi"):
                # Unified curate launcher; legacy kind aliases preserved
                # so old UI builds and any saved client requests still work.
                job = jobs.launch_curate(params)
            elif kind == "relations":
                job = jobs.launch_relations(params)
            elif kind == "relation_detection":
                job = jobs.launch_relation_detection(params)
            elif kind == "relation_report":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=relation_report")
                try:
                    job = jobs.launch_relation_report(run_id)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind == "relation_validation":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=relation_validation")
                try:
                    job = jobs.launch_relation_validation(run_id, params=params)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind == "column_verify":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=column_verify")
                try:
                    job = jobs.launch_column_verify(run_id, params=params)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            elif kind == "web_verify":
                run_id = payload.get("run_id")
                if not run_id:
                    raise HTTPException(400, "run_id is required for kind=web_verify")
                try:
                    job = jobs.launch_web_verify(run_id, params=params)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
            else:
                raise HTTPException(400, f"unknown kind: {kind}")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/launcher/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job.to_dict()

    @app.get("/api/launcher/jobs/{job_id}/log")
    def get_job_log(
        job_id: str,
        offset: int = Query(0, description="Byte offset; use next_offset to tail."),
    ) -> Dict[str, Any]:
        try:
            return jobs.read_log(job_id, offset=offset)
        except KeyError as exc:
            raise HTTPException(404, f"job not found: {job_id}") from exc

    @app.post("/api/launcher/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return job.to_dict()

    @app.delete("/api/launcher/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, Any]:
        try:
            jobs.delete(job_id)
        except KeyError as exc:
            raise HTTPException(404, f"job not found: {job_id}") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, "job_id": job_id}
