"""Deployment management for fine-tuned models.

Deploying an Azure OpenAI fine-tuned model is a **management-plane** action
(the data-plane ``openai`` SDK can't do it), so we shell out to ``az
cognitiveservices account deployment`` -- consistent with the repo's
``AzureCliCredential`` posture (``az login`` is already required for any live
call). The account name is the endpoint subdomain; the resource group is not
derivable from the endpoint and must be supplied (``--resource-group`` or
``$AZURE_RESOURCE_GROUP``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from urllib.parse import urlparse

logger = logging.getLogger("rft.deploy")

# Default SKU for a fine-tuned deployment. "Standard" is the broadly-available
# data-zone tier for o4-mini FT; override per-call if a resource needs another.
DEFAULT_SKU_NAME = "Standard"
DEFAULT_SKU_CAPACITY = 1
DEFAULT_MODEL_FORMAT = "OpenAI"


def account_name_from_endpoint(endpoint: str) -> str:
    """``https://my-aoai.openai.azure.com/`` -> ``my-aoai``."""
    host = urlparse(endpoint).hostname or endpoint
    return host.split(".")[0]


def resolve_resource_group(resource_group: str | None) -> str:
    rg = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    if not rg:
        raise RuntimeError(
            "Deployment needs a resource group, which is not derivable from the "
            "endpoint. Pass --resource-group <name> or set $AZURE_RESOURCE_GROUP. "
            "Find it with: az cognitiveservices account list "
            "--query \"[].{name:name,rg:resourceGroup}\" -o table"
        )
    return rg


def _az(args: list[str]) -> dict | list | None:
    """Run an ``az`` command, returning parsed JSON stdout (or None)."""
    cmd = ["az", *args, "-o", "json"]
    logger.info("az %s", " ".join(args))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def create_deployment(
    *,
    endpoint: str,
    resource_group: str | None,
    deployment_name: str,
    model_name: str,
    model_version: str = "1",
    sku_name: str = DEFAULT_SKU_NAME,
    sku_capacity: int = DEFAULT_SKU_CAPACITY,
    model_format: str = DEFAULT_MODEL_FORMAT,
) -> dict:
    """Create (or update) a deployment for a fine-tuned model. Returns its JSON."""
    account = account_name_from_endpoint(endpoint)
    rg = resolve_resource_group(resource_group)
    result = _az(
        [
            "cognitiveservices",
            "account",
            "deployment",
            "create",
            "--name",
            account,
            "--resource-group",
            rg,
            "--deployment-name",
            deployment_name,
            "--model-name",
            model_name,
            "--model-version",
            model_version,
            "--model-format",
            model_format,
            "--sku-name",
            sku_name,
            "--sku-capacity",
            str(sku_capacity),
        ]
    )
    logger.info(
        "Deployed %s -> deployment %r on %s (rg=%s)",
        model_name,
        deployment_name,
        account,
        rg,
    )
    return result or {}


def delete_deployment(
    *, endpoint: str, resource_group: str | None, deployment_name: str
) -> None:
    account = account_name_from_endpoint(endpoint)
    rg = resolve_resource_group(resource_group)
    _az(
        [
            "cognitiveservices",
            "account",
            "deployment",
            "delete",
            "--name",
            account,
            "--resource-group",
            rg,
            "--deployment-name",
            deployment_name,
        ]
    )
    logger.info("Deleted deployment %r on %s", deployment_name, account)


def list_deployments(*, endpoint: str, resource_group: str | None) -> list[dict]:
    account = account_name_from_endpoint(endpoint)
    rg = resolve_resource_group(resource_group)
    out = _az(
        [
            "cognitiveservices",
            "account",
            "deployment",
            "list",
            "--name",
            account,
            "--resource-group",
            rg,
        ]
    )
    return out if isinstance(out, list) else []
