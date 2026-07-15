"""The generic RFT lifecycle engine.

One :class:`RftEngine` drives the *entire* Azure OpenAI lifecycle for any
:class:`RFT._common.spec.RftTaskSpec`:

    prepare -> upload -> submit -> status -> deploy -> run
                              \\-> supervise (auto-resubmit on failure)
    cancel / delete / models / list-jobs

Everything that used to be copy-pasted into each ``*_rft_vN.py`` launcher
(the ``method`` payload, ``cmd_submit`` / ``cmd_all``, the supervisor loop that
only inconsistency_v1 had) lives here once and is parameterised by the spec.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from RFT._common import deploy as _deploy
from RFT._common._rft_common import (
    DEFAULT_API_VERSION,
    assert_model_fine_tunable,
    build_client,
    list_fine_tunable_models,
    wait_for_file_ready,
)
from RFT._common.spec import Hparams, RftTaskSpec
from RFT._common.state import RunState

logger = logging.getLogger("rft.engine")

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class RftEngine:
    """Stateful driver bound to one task spec + (endpoint, output_dir)."""

    def __init__(
        self,
        spec: RftTaskSpec,
        *,
        endpoint: str,
        output_dir: Path,
        api_version: str = DEFAULT_API_VERSION,
    ) -> None:
        self.spec = spec
        self.endpoint = endpoint
        self.output_dir = Path(output_dir)
        self.api_version = api_version

    # -- helpers -----------------------------------------------------------

    def _client(self, endpoint: str | None = None):
        return build_client(endpoint or self.endpoint, self.api_version)

    def _state(self) -> RunState:
        return RunState.load(self.output_dir)

    def _train_path(self) -> Path:
        return self.spec.data_dir / self.spec.train_filename

    def _valid_path(self) -> Path:
        return self.spec.data_dir / self.spec.valid_filename

    def _eval_path(self) -> Path:
        return self.spec.data_dir / self.spec.eval_filename

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        n = 0
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)  # validate each record parses
                    n += 1
        if n == 0:
            raise ValueError(f"{path} contained no records.")
        return n

    def _build_method(
        self, *, variant: str, grader_model: str, hparams: Hparams
    ) -> dict[str, Any]:
        """The single source of truth for the RFT ``method`` payload."""
        return {
            "type": "reinforcement",
            "reinforcement": {
                "grader": self.spec.build_grader(variant, grader_model),
                "response_format": self.spec.response_format(),
                "hyperparameters": hparams.to_payload(),
            },
        }

    # -- prepare / upload --------------------------------------------------

    def prepare(self) -> None:
        """Validate the version's splits exist and are non-empty JSONL."""
        for path in (self._train_path(), self._valid_path()):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- run this task's build_splits script first."
                )
            logger.info("Validated %d records in %s", self._count_jsonl(path), path)

    def upload(self) -> None:
        """Upload train + valid splits to the endpoint; persist file ids."""
        self.prepare()
        client = self._client()
        state = self._state()
        state.endpoint = self.endpoint
        for attr, path in (
            ("train_file_id", self._train_path()),
            ("valid_file_id", self._valid_path()),
        ):
            with path.open("rb") as fh:
                resp = client.files.create(file=fh, purpose="fine-tune")
            setattr(state, attr, resp.id)
            logger.info("Uploaded %s -> %s", path.name, resp.id)
            state.save(self.output_dir)  # save eagerly so a slow import can't lose it
            wait_for_file_ready(client, resp.id)

    # -- submit ------------------------------------------------------------

    def submit(
        self,
        *,
        variant: str,
        grader_model: str,
        base_model: str,
        suffix: str,
        hparams: Hparams,
        skip_model_check: bool = False,
    ) -> str:
        """Create the RFT job from already-uploaded files. Returns job id."""
        state = self._state()
        if not (state.train_file_id and state.valid_file_id):
            raise RuntimeError("Train/valid file IDs missing. Run `upload` first.")
        client = self._client()
        assert_model_fine_tunable(client, base_model, skip=skip_model_check)
        wait_for_file_ready(client, state.train_file_id)
        wait_for_file_ready(client, state.valid_file_id)

        job = client.fine_tuning.jobs.create(
            model=base_model,
            training_file=state.train_file_id,
            validation_file=state.valid_file_id,
            method=self._build_method(
                variant=variant, grader_model=grader_model, hparams=hparams
            ),
            suffix=suffix,
        )
        state.job_id = job.id
        state.endpoint = self.endpoint
        state.base_model = base_model
        state.suffix = suffix
        state.save(self.output_dir)
        logger.info(
            "Submitted %s [%s] job %s (status=%s) on %s",
            self.spec.name, variant, job.id, job.status, self.endpoint,
        )
        print(f"\nJob ID: {job.id}  (endpoint={self.endpoint}, suffix={suffix})")
        return job.id

    # -- status / events ---------------------------------------------------

    def _stream_events(self, client, job_id: str, after: str | None) -> str | None:
        page = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=20, after=after
        )
        events = list(reversed(list(page.data)))  # SDK is newest-first
        for ev in events:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.created_at))
            print(f"  [{ts}] {ev.level:<7} {ev.message}")
        return events[-1].id if events else after

    def status(
        self,
        *,
        job_id: str | None = None,
        follow: bool = False,
        poll_interval: int = 30,
        client=None,
    ) -> str:
        """Print status + stream events; return the final status seen."""
        state = self._state()
        job_id = job_id or state.job_id
        if not job_id:
            raise RuntimeError("No job id known. Pass --job-id or run `submit` first.")
        client = client or self._client()
        after: str | None = None
        tick = 0
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            ftm = getattr(job, "fine_tuned_model", None)
            print(
                f"\n[{tick:>3}] status={job.status} "
                f"trained_tokens={getattr(job, 'trained_tokens', None)} "
                f"fine_tuned_model={ftm}"
            )
            after = self._stream_events(client, job_id, after)
            if job.status == "succeeded" and ftm and state.fine_tuned_model != ftm:
                state.fine_tuned_model = ftm
                state.save(self.output_dir)
            if job.status in _TERMINAL_STATUSES or not follow:
                return job.status
            tick += 1
            time.sleep(poll_interval)

    def cancel(self, *, job_id: str | None = None) -> None:
        state = self._state()
        job_id = job_id or state.job_id
        if not job_id:
            raise RuntimeError("No job id known. Pass --job-id or run `submit` first.")
        job = self._client().fine_tuning.jobs.cancel(job_id)
        logger.info("Cancelled %s (status=%s)", job_id, job.status)

    # -- supervise (auto-resubmit on failure) ------------------------------

    def supervise(
        self,
        *,
        submit_kwargs: dict[str, Any],
        endpoints: list[str],
        attach_job_id: str | None = None,
        max_attempts: int = 4,
        max_poll_errors: int = 20,
        max_runtime_hours: float = 48.0,
        poll_interval: int = 60,
    ) -> str | None:
        """Watch the job; on terminal ``failed`` re-upload + resubmit to the
        next endpoint in the pool. Survives transient endpoint-side 500s.

        Returns the succeeded ``fine_tuned_model`` (or ``None`` if it gave up).
        """
        if not endpoints:
            raise RuntimeError("Supervise endpoint pool is empty.")
        state = self._state()
        job_id = attach_job_id or state.job_id
        cur = 0
        if not job_id:
            logger.info("No job to attach; submitting fresh on %s.", endpoints[0])
            job_id = self._submit_on(endpoints[0], submit_kwargs)
            if not job_id:
                raise RuntimeError(f"Initial submit failed on {endpoints[0]}.")

        attempts, poll_fails = 0, 0
        deadline = time.time() + max_runtime_hours * 3600 if max_runtime_hours > 0 else None
        while True:
            if deadline and time.time() > deadline:
                logger.error("Max runtime (%.1fh) exceeded; stopping.", max_runtime_hours)
                self._record_supervise(state, job_id, endpoints[cur], attempts, "runtime")
                return None
            endpoint = endpoints[cur]
            logger.info("Supervising %s on %s (attempt %d/%d)",
                        job_id, endpoint, attempts + 1, max_attempts)
            client = self._client(endpoint)
            try:
                status = self._poll_terminal(client, job_id, poll_interval)
                poll_fails = 0
            except Exception as exc:  # noqa: BLE001
                poll_fails += 1
                if poll_fails >= max_poll_errors:
                    logger.error("Polling %s failed %d times; giving up.", job_id, poll_fails)
                    self._record_supervise(state, job_id, endpoint, attempts, "poll_errors")
                    return None
                logger.warning("Poll %s errored (%d/%d): %s; retrying.",
                               job_id, poll_fails, max_poll_errors, exc)
                time.sleep(poll_interval)
                continue

            ftm = getattr(client.fine_tuning.jobs.retrieve(job_id), "fine_tuned_model", None)
            state.history.append(
                {"job": job_id, "endpoint": endpoint, "status": status, "fine_tuned_model": ftm}
            )
            if status == "succeeded":
                state.endpoint, state.fine_tuned_model = endpoint, ftm
                self._record_supervise(state, job_id, endpoint, attempts, "succeeded")
                print(f"\nSUCCESS  job={job_id}  model={ftm}  endpoint={endpoint}")
                return ftm
            if status == "cancelled":
                logger.info("CANCELLED: %s -- stopping (user action).", job_id)
                return None

            attempts += 1
            logger.warning("Job %s FAILED on %s (attempt %d/%d).",
                           job_id, endpoint, attempts, max_attempts)
            if attempts >= max_attempts:
                logger.error("Resubmission limit reached; giving up.")
                self._record_supervise(state, job_id, endpoint, attempts, "exhausted")
                return None
            new_job = None
            for _ in range(len(endpoints)):
                cur = (cur + 1) % len(endpoints)
                new_job = self._submit_on(endpoints[cur], submit_kwargs)
                if new_job:
                    break
            if not new_job:
                logger.error("No endpoint accepted a resubmit; giving up.")
                return None
            job_id = new_job

    def _poll_terminal(self, client, job_id: str, poll_interval: int) -> str:
        after = None
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            after = self._stream_events(client, job_id, after)
            if job.status in _TERMINAL_STATUSES:
                return job.status
            time.sleep(poll_interval)

    def _submit_on(self, endpoint: str, submit_kwargs: dict[str, Any]) -> str | None:
        """Upload + submit on one endpoint; None on any error (skip bad endpoint)."""
        try:
            saved_endpoint = self.endpoint
            self.endpoint = endpoint
            try:
                self.upload()
                return self.submit(**submit_kwargs)
            finally:
                self.endpoint = saved_endpoint
        except Exception as exc:  # noqa: BLE001
            logger.warning("Submit on %s failed (%s: %s); next endpoint.",
                           endpoint, type(exc).__name__, exc)
            return None

    def _record_supervise(
        self, state: RunState, job_id: str, endpoint: str, attempts: int, result: str
    ) -> None:
        state.job_id, state.endpoint, state.attempts = job_id, endpoint, attempts
        state.save(self.output_dir)

    # -- deploy / delete ---------------------------------------------------

    def _resolve_ft_model(self, job_id: str | None) -> str:
        state = self._state()
        if state.fine_tuned_model and not job_id:
            return state.fine_tuned_model
        jid = job_id or state.job_id
        if not jid:
            raise RuntimeError("No fine-tuned model known. Pass --job-id or run `status`.")
        job = self._client().fine_tuning.jobs.retrieve(jid)
        ftm = getattr(job, "fine_tuned_model", None)
        if not ftm:
            raise RuntimeError(f"Job {jid} has no fine_tuned_model (status={job.status}).")
        return ftm

    def deploy(
        self,
        *,
        deployment_name: str,
        resource_group: str | None,
        job_id: str | None = None,
        model: str | None = None,
        sku_name: str = _deploy.DEFAULT_SKU_NAME,
        sku_capacity: int = _deploy.DEFAULT_SKU_CAPACITY,
    ) -> dict:
        ft_model = model or self._resolve_ft_model(job_id)
        result = _deploy.create_deployment(
            endpoint=self.endpoint,
            resource_group=resource_group,
            deployment_name=deployment_name,
            model_name=ft_model,
            sku_name=sku_name,
            sku_capacity=sku_capacity,
        )
        state = self._state()
        state.deployment_name = deployment_name
        state.fine_tuned_model = ft_model
        state.save(self.output_dir)
        print(f"\nDeployed {ft_model} -> deployment '{deployment_name}'")
        return result

    def delete(
        self,
        *,
        targets: set[str],
        resource_group: str | None = None,
        job_id: str | None = None,
    ) -> None:
        """Tear down chosen artifacts. ``targets`` ⊆ {deployment, model, files, job}."""
        state = self._state()
        client = self._client()
        if "deployment" in targets and state.deployment_name:
            _deploy.delete_deployment(
                endpoint=self.endpoint,
                resource_group=resource_group,
                deployment_name=state.deployment_name,
            )
            state.deployment_name = None
        if "model" in targets:
            ftm = state.fine_tuned_model or self._resolve_ft_model(job_id)
            client.models.delete(ftm)
            logger.info("Deleted fine-tuned model %s", ftm)
            state.fine_tuned_model = None
        if "files" in targets:
            for attr in ("train_file_id", "valid_file_id"):
                fid = getattr(state, attr)
                if fid:
                    try:
                        client.files.delete(fid)
                        logger.info("Deleted file %s", fid)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not delete file %s: %s", fid, exc)
                    setattr(state, attr, None)
        if "job" in targets and state.job_id:
            try:
                client.fine_tuning.jobs.cancel(state.job_id)
            except Exception:  # noqa: BLE001 - already terminal is fine
                pass
            logger.info("Released job %s from state.", state.job_id)
            state.job_id = None
        state.save(self.output_dir)

    # -- run (invoke the deployed model over eval.jsonl) -------------------

    def run(
        self,
        *,
        deployment: str | None = None,
        split: str = "eval",
        concurrency: int = 8,
        limit: int = 0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Call the deployed FT model over the eval split; parse + score."""
        state = self._state()
        deployment = deployment or state.deployment_name
        if not deployment:
            raise RuntimeError("No deployment known. Pass --deployment or run `deploy`.")
        path = self.spec.data_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if limit:
            rows = rows[:limit]
        client = self._client()
        response_format = self.spec.response_format()
        effort = reasoning_effort or self.spec.run_reasoning_effort

        def _call(row: dict) -> str | None:
            kwargs: dict[str, Any] = {
                "model": deployment,
                "messages": row["messages"],
                "response_format": response_format,
            }
            if effort:
                kwargs["reasoning_effort"] = effort
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                logger.warning("run call failed: %s", exc)
                return None

        raws: list[str | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_call, row): i for i, row in enumerate(rows)}
            for fut in as_completed(futs):
                raws[futs[fut]] = fut.result()

        parser = self.spec.prediction_parser
        preds = [parser(r) if parser else set() for r in raws]
        out_dir = self.output_dir / "eval_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"{split}_predictions.jsonl"
        with pred_path.open("w") as fh:
            for row, raw, pred in zip(rows, raws, preds):
                fh.write(json.dumps({
                    "source": row.get("source"),
                    "column": row.get("column"),
                    "raw": raw,
                    "predicted": sorted(pred),
                }) + "\n")

        summary: dict[str, Any] = {
            "task": self.spec.name,
            "deployment": deployment,
            "split": split,
            "reasoning_effort": effort,
            "n_rows": len(rows),
            "api_errors": sum(1 for r in raws if r is None),
        }
        if self.spec.scorer:
            summary["metrics"] = self.spec.scorer(rows, preds)
        (out_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"\nPredictions -> {pred_path}")
        return summary

    # -- introspection -----------------------------------------------------

    def list_models(self, *, show_all: bool = False) -> None:
        client = self._client()
        rows = list(client.models.list().data)
        if not show_all:
            rows = list_fine_tunable_models(client)
        for m in sorted(rows, key=lambda r: r.id):
            print(m.id)

    def list_jobs(self, *, limit: int = 20) -> None:
        client = self._client()
        for job in client.fine_tuning.jobs.list(limit=limit).data:
            print(
                f"{job.id}  status={job.status:<10} "
                f"model={getattr(job, 'fine_tuned_model', None)}"
            )
