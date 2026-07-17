"""Unified RFT CLI -- one entry point for every registered task.

    python -m RFT.cli --list-tasks
    python -m RFT.cli --task typo_v10 all --reward-variant r1 --endpoint <ep>
    python -m RFT.cli --task typo_v10 status --follow
    python -m RFT.cli --task typo_v10 deploy --deployment-name o4-mini-v10-r1 \\
        --resource-group <rg>
    python -m RFT.cli --task typo_v10 run --deployment o4-mini-v10-r1
    python -m RFT.cli --task inconsistency_v1 supervise --attach-job-id <job>

Flags such as ``--reward-variant`` choices, ``--grader-model`` choices, and the
hyperparameter defaults are derived from the selected task's spec, so they can
never drift out of sync with the task definition.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RFT._common._rft_common import (  # noqa: E402
    DEFAULT_API_VERSION,
    resolve_default_endpoint,
)
from RFT._common.engine import RftEngine  # noqa: E402
from RFT._common.spec import Hparams, RftTaskSpec  # noqa: E402
from RFT import registry  # noqa: E402

logger = logging.getLogger("rft.cli")

DEFAULT_SUPERVISE_POOL: list[str] = []


# ---------------------------------------------------------------------------
# Argparse builders (spec-driven)
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser, spec: RftTaskSpec) -> None:
    p.add_argument(
        "--endpoint", default=None, help="Azure OpenAI endpoint (auto-resolved if omitted)."
    )
    p.add_argument("--api-version", default=DEFAULT_API_VERSION)
    p.add_argument("--output-dir", type=Path, default=spec.default_output_dir)


def _add_submit(p: argparse.ArgumentParser, spec: RftTaskSpec) -> None:
    if len(spec.reward_variants) > 1:
        p.add_argument(
            "--reward-variant", choices=list(spec.reward_variants),
            default=spec.default_reward_variant,
            help=f"Reward variant (default {spec.default_reward_variant}).",
        )
    p.add_argument("--base-model", default=spec.base_model)
    grader_kw = {"default": spec.default_grader_model}
    if spec.grader_model_whitelist:
        grader_kw["choices"] = list(spec.grader_model_whitelist)
    p.add_argument("--grader-model", **grader_kw)
    p.add_argument("--suffix", default=spec.default_suffix)
    h = spec.hparams
    p.add_argument("--n-epochs", type=int, default=h.n_epochs)
    p.add_argument("--batch-size", type=int, default=h.batch_size)
    p.add_argument("--eval-interval", type=int, default=h.eval_interval)
    p.add_argument("--eval-samples", type=int, default=h.eval_samples)
    p.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default=h.reasoning_effort
    )
    p.add_argument("--learning-rate-multiplier", type=float, default=h.learning_rate_multiplier)
    p.add_argument("--compute-multiplier", type=float, default=h.compute_multiplier)
    p.add_argument("--seed", type=int, default=h.seed)
    p.add_argument("--skip-model-check", action="store_true")


def _build_parser(spec: RftTaskSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"RFT.cli --task {spec.name}",
        description=f"RFT lifecycle for {spec.name} (error_type={spec.error_type}).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("prepare", "upload"):
        sp = sub.add_parser(name)
        _add_common(sp, spec)

    sp = sub.add_parser("submit")
    _add_common(sp, spec)
    _add_submit(sp, spec)

    sp = sub.add_parser("all", help="prepare -> upload -> submit -> status --follow")
    _add_common(sp, spec)
    _add_submit(sp, spec)
    sp.add_argument("--poll-interval", type=int, default=30)

    sp = sub.add_parser("status")
    _add_common(sp, spec)
    sp.add_argument("--job-id", default=None)
    sp.add_argument("--follow", action="store_true")
    sp.add_argument("--poll-interval", type=int, default=30)

    sp = sub.add_parser("cancel")
    _add_common(sp, spec)
    sp.add_argument("--job-id", default=None)

    sp = sub.add_parser(
        "supervise",
        help="Poll the job; on terminal 'failed', re-upload + resubmit to the "
        "next endpoint in --endpoints. Runs for hours -- launch detached.",
    )
    _add_common(sp, spec)
    _add_submit(sp, spec)
    sp.add_argument(
        "--endpoints",
        default=os.environ.get("AZURE_OPENAI_ENDPOINTS", ""),
        help="Comma-separated retry pool (default: $AZURE_OPENAI_ENDPOINTS).",
    )
    sp.add_argument("--attach-job-id", default=None)
    sp.add_argument("--max-attempts", type=int, default=4)
    sp.add_argument("--max-poll-errors", type=int, default=20)
    sp.add_argument("--max-runtime-hours", type=float, default=48.0)
    sp.add_argument("--poll-interval", type=int, default=60)

    sp = sub.add_parser("deploy", help="Create a deployment for the fine-tuned model.")
    _add_common(sp, spec)
    sp.add_argument("--deployment-name", required=True)
    sp.add_argument("--resource-group", default=None, help="Or set $AZURE_RESOURCE_GROUP.")
    sp.add_argument("--job-id", default=None, help="Resolve FT model from this job.")
    sp.add_argument("--model", default=None, help="FT model id (overrides job lookup).")
    sp.add_argument("--sku-name", default="Standard")
    sp.add_argument("--sku-capacity", type=int, default=1)

    sp = sub.add_parser("delete", help="Tear down deployment / model / files / job.")
    _add_common(sp, spec)
    sp.add_argument(
        "--target", action="append", default=None,
        choices=["deployment", "model", "files", "job"],
        help="Repeatable. Default: deployment only.",
    )
    sp.add_argument("--resource-group", default=None)
    sp.add_argument("--job-id", default=None)
    sp.add_argument("--yes", action="store_true", help="Skip confirmation.")

    sp = sub.add_parser("run", help="Invoke the deployed model over the eval split; score.")
    _add_common(sp, spec)
    sp.add_argument("--deployment", default=None, help="Deployment name (default: from state).")
    sp.add_argument("--split", default="eval")
    sp.add_argument("--concurrency", type=int, default=8)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--reasoning-effort", default=None)

    sp = sub.add_parser("models", help="List models on the endpoint.")
    _add_common(sp, spec)
    sp.add_argument("--all", action="store_true")

    sp = sub.add_parser("list-jobs")
    _add_common(sp, spec)
    sp.add_argument("--limit", type=int, default=20)

    return p


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _hparams_from(args: argparse.Namespace, spec: RftTaskSpec) -> Hparams:
    return spec.hparams.with_overrides(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_samples=args.eval_samples,
        reasoning_effort=args.reasoning_effort,
        learning_rate_multiplier=args.learning_rate_multiplier,
        compute_multiplier=args.compute_multiplier,
        seed=args.seed,
    )


def _submit_kwargs(args: argparse.Namespace, spec: RftTaskSpec) -> dict:
    return {
        "variant": getattr(args, "reward_variant", spec.default_reward_variant),
        "grader_model": args.grader_model,
        "base_model": args.base_model,
        "suffix": args.suffix,
        "hparams": _hparams_from(args, spec),
        "skip_model_check": args.skip_model_check,
    }


def _dispatch(engine: RftEngine, spec: RftTaskSpec, args: argparse.Namespace) -> None:
    cmd = args.command
    if cmd == "prepare":
        engine.prepare()
    elif cmd == "upload":
        engine.upload()
    elif cmd == "submit":
        engine.submit(**_submit_kwargs(args, spec))
    elif cmd == "all":
        engine.upload()
        engine.submit(**_submit_kwargs(args, spec))
        engine.status(follow=True, poll_interval=args.poll_interval)
    elif cmd == "status":
        engine.status(job_id=args.job_id, follow=args.follow, poll_interval=args.poll_interval)
    elif cmd == "cancel":
        engine.cancel(job_id=args.job_id)
    elif cmd == "supervise":
        engine.supervise(
            submit_kwargs=_submit_kwargs(args, spec),
            endpoints=[e.strip() for e in args.endpoints.split(",") if e.strip()],
            attach_job_id=args.attach_job_id,
            max_attempts=args.max_attempts,
            max_poll_errors=args.max_poll_errors,
            max_runtime_hours=args.max_runtime_hours,
            poll_interval=args.poll_interval,
        )
    elif cmd == "deploy":
        engine.deploy(
            deployment_name=args.deployment_name,
            resource_group=args.resource_group,
            job_id=args.job_id,
            model=args.model,
            sku_name=args.sku_name,
            sku_capacity=args.sku_capacity,
        )
    elif cmd == "delete":
        targets = set(args.target or ["deployment"])
        if not args.yes:
            resp = input(f"Delete {sorted(targets)} for {spec.name}? [y/N] ")
            if resp.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return
        engine.delete(targets=targets, resource_group=args.resource_group, job_id=args.job_id)
    elif cmd == "run":
        engine.run(
            deployment=args.deployment,
            split=args.split,
            concurrency=args.concurrency,
            limit=args.limit,
            reasoning_effort=args.reasoning_effort,
        )
    elif cmd == "models":
        engine.list_models(show_all=args.all)
    elif cmd == "list-jobs":
        engine.list_jobs(limit=args.limit)
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(f"Unknown command {cmd!r}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argv = list(sys.argv[1:] if argv is None else argv)

    # `new` scaffolds a fresh task; it has no --task (the task doesn't exist yet).
    if argv and argv[0] == "new":
        from RFT import scaffold

        return scaffold.main(argv)

    # Pre-parse the task selector / list flag before building the spec parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task")
    pre.add_argument("--list-tasks", action="store_true")
    known, rest = pre.parse_known_args(argv)
    if known.list_tasks or not known.task:
        print(registry.format_task_list())
        if not known.task and not known.list_tasks:
            print("\nSelect a task with --task <name>, then a command.")
        return 0

    spec = registry.get(known.task)
    args = _build_parser(spec).parse_args(rest)

    endpoint = args.endpoint or resolve_default_endpoint()
    if not args.endpoint:
        logger.info("Using endpoint %s (auto-resolved).", endpoint)
    engine = RftEngine(
        spec, endpoint=endpoint, output_dir=args.output_dir, api_version=args.api_version
    )
    _dispatch(engine, spec, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
