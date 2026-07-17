"""Shared plumbing for the GPT-5.5 baseline labeler.

Vendored (and trimmed) from the RFT tree so ``external/baseline/`` is self-contained:

* ``build_client``                         <- RFT._common._rft_common
* ``chat_rf_to_responses_text``,
  ``extract_responses_output_text``        <- RFT.typo.v5.eval.compare_ft_vs_base
* ``norm``, ``textual_columns``,
  ``values_with_counts``, ``parse_typos``,
  ``issues_for_csv``                       <- RFT.typo.v8.scripts.label_dataset_v8

The only non-vendored dependency is ``src.core.*`` (the shared detection-run
artifact pipeline), imported by the labeler entry point, not here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from external.baseline._task import MAX_DISTINCT

logger = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Azure client
# ---------------------------------------------------------------------------


def build_client(endpoint: str, api_version: str) -> AzureOpenAI:
    """Construct an AzureOpenAI client, preferring Entra over API keys."""
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    if api_key:
        logger.info("Using API key auth (AZURE_OPENAI_KEY is set).")
        return AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    logger.info("Using AzureCliCredential (run `az login` if this fails).")
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )


# ---------------------------------------------------------------------------
# Responses-API adapters
# ---------------------------------------------------------------------------


def chat_rf_to_responses_text(chat_rf: dict[str, Any]) -> dict[str, Any]:
    """Convert chat-API ``response_format`` to Responses-API ``text.format``.

    Chat shape:    ``{type: 'json_schema', json_schema: {name, schema, strict}}``
    Responses:     ``{type: 'json_schema', name, schema, strict}``  (flat)
    """
    if chat_rf.get("type") != "json_schema":
        return chat_rf
    inner = chat_rf.get("json_schema", {})
    return {
        "type": "json_schema",
        "name": inner.get("name", "output"),
        "schema": inner.get("schema", {}),
        "strict": inner.get("strict", True),
    }


def extract_responses_output_text(resp: Any) -> str:
    """Pull plain text out of a Responses-API result."""
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    out = getattr(resp, "output", None) or []
    parts: list[str] = []
    for item in out:
        content = getattr(item, "content", None) or []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


# ---------------------------------------------------------------------------
# CSV / issue helpers
# ---------------------------------------------------------------------------


def norm(s: Any) -> str:
    return " ".join(str(s).split())


def textual_columns(df: pd.DataFrame) -> list[str]:
    """Columns worth sending to a typo detector: >=30% alphabetic, >=2 distinct."""
    out = []
    for c in df.columns:
        vals = [v for v in df[c].astype(str).tolist() if v.strip()][:200]
        if not vals or df[c].nunique(dropna=True) < 2:
            continue
        alpha = sum(1 for v in vals if any(ch.isalpha() for ch in v))
        if alpha / len(vals) >= 0.30:
            out.append(c)
    return out


def values_with_counts(df: pd.DataFrame, col: str) -> list[tuple[str, int]]:
    vc = df[col].astype(str).value_counts()
    return [(str(v), int(c)) for v, c in vc.items() if str(v).strip()][:MAX_DISTINCT]


def parse_typos(raw: str | None) -> list[dict]:
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    out = []
    for it in obj.get("typos", []) or []:
        if isinstance(it, dict) and isinstance(it.get("value"), str):
            out.append(it)
    return out


def issues_for_csv(df: pd.DataFrame, col_typos: dict[str, list[dict]]) -> list[dict]:
    issues = []
    for col, typos in col_typos.items():
        norm_series = df[col].astype(str).map(norm)
        for typ in typos:
            val = typ["value"]
            rows = [int(i) for i in df.index[norm_series == norm(val)].tolist()]
            if not rows:  # value not actually in the column -> hallucination, drop
                continue
            corr = typ.get("correction")
            flav = (typ.get("flavour") or "").strip()
            rat = (typ.get("rationale") or "").strip()
            issues.append(
                {
                    "row": rows[0],
                    "columns": [col],
                    "affected_rows": rows,
                    "affected_rows_total": len(rows),
                    "observed": val,
                    "explanation": (f"typo:{flav} - {rat}" if flav else rat) or "typo",
                    "suggested_fix": corr,
                    "corrected_value": {col: corr} if isinstance(corr, str) else None,
                }
            )
    return issues


def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV as all-strings, latin-1 fallback. Returns None on failure."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
        except Exception as exc:
            logger.warning("CSV load failed for %s: %s", path, exc)
            return None
    except Exception as exc:
        logger.warning("CSV load failed for %s: %s", path, exc)
        return None
