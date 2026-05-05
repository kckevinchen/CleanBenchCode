"""Live regression test: re-probe the columns that produced NFI singletons
in the kmax sweep ``20260501T001835Z_gpt-5.2`` run, using the *new*
prompt with the QUANTITATIVE-COLUMN GATE.

Why this exists
---------------
The kmax sweep produced 89 ``number_format_inconsistency`` singletons,
~80 of which originated from columns that are CODES / IDENTIFIERS, not
quantities (zip, FIPS, county/town code, etc.). We diagnosed those as
false positives and patched the prompt to fast-skip code/identifier
columns. This test re-runs the patched prompt on the exact (source,
column) pairs and asserts the model now returns ``issues == []`` for
the code-column cases.

Cost / safety
-------------
Each test in this module makes a real LLM call. The whole module is
SKIPPED unless the user opts in via ``RUN_LLM_TESTS=1`` (and Azure
credentials are configured). One column passes through the gate to
verify the prompt still allows legitimate quantity columns to fire
when warranted - we don't assert a positive there because variance
across runs is real.

Run with:
    RUN_LLM_TESTS=1 pytest tests/test_nfi_singleton_regression.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pytest

from src.core.data_loader import COLUMN_MODE_SINGLE, CsvChunk
from src.core.detector import parse_llm_json
from src.core.error_taxonomy import load_taxonomy
from src.core.llm_client import LLMClient


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "prompts" / "error_types.yaml"
DATA_ROOT = REPO_ROOT / "data" / "curated_bing21_min_filtered"

# Model used in the kmax sweep.
MODEL = os.environ.get("NFI_REGRESSION_MODEL", "gpt-5.2")

# Per-column timeout multiplier - reasoning models are slow.
REASONING_EFFORT = os.environ.get("NFI_REGRESSION_EFFORT", "medium")


# ---------------------------------------------------------------------------
# Singleton-producing arms from the kmax sweep
# ---------------------------------------------------------------------------
#
# Each entry: (csv_basename, parent_xlsx, column, kind, expect_nfi_empty)
#   kind: human-readable rationale
#   expect_nfi_empty: True if we expect the patched prompt to suppress
#     the finding (code/identifier columns); False for the legitimate
#     quantity column which we sanity-check loads but don't assert on.
SINGLETON_ARMS = [
    (
        "f5f0334b.csv",
        "44245e7c13a5606b43cdeb924ee28b5c72374a15.xlsx",
        "AMOUNT",
        "genuine numeric column - the 25 singletons here likely came "
        "from precision/decimal-point variance, not a code FP. The "
        "new gate does not explicitly target this; we don't assert "
        "empty - just sanity-check the prompt still produces parseable "
        "output.",
        False,
    ),
    (
        "a7d68ceb.csv",
        "9403ed7a28420de5f0f3a5ac3d625ebf034fe44f.xlsx",
        "County Code",
        "code column - 2-digit FIPS-like with leading zeros",
        True,
    ),
    (
        "00f1d635.csv",
        "e5739481e3c904ac0904b5cf366b85d4f068b9f9.xlsx",
        "Town Code",
        "code column - 3-digit padded codes",
        True,
    ),
    (
        "f150959e.csv",
        "414b30bdcab1ceee8808c0f45a690cd4596ec729.xlsx",
        "C02451V02968",
        "opaque identifier column with mixed widths (e.g. '17' vs '023')",
        True,
    ),
    (
        "13d0446f.csv",
        "52409d4cfd6525f96b5cac1cef18788e24eddb7b.xlsx",
        "Reading Quiz 1 %",
        "quiz-percent column - quantity-like, gate should let it "
        "through; we don't assert presence/absence here",
        False,
    ),
    (
        "e9088bff.csv",
        "9bd685db03b6491afee6a17cc840aa6ec56dde30.xlsx",
        "% of Facilities who Reported Resident Vaccinations in the Current or Prior Week",
        "legitimate quantity column - this is the only NFI arm that "
        "fired stably in the sweep; we don't assert empty here",
        False,
    ),
]


# ---------------------------------------------------------------------------
# Module-level skip if the live-LLM opt-in flag is missing.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="Live LLM regression - opt in with RUN_LLM_TESTS=1",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(PROMPT_PATH)


@pytest.fixture(scope="module")
def llm():
    return LLMClient()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_csv(basename: str, xlsx: str) -> Optional[Path]:
    """Locate the CSV in either curated or raw bing21 layouts."""
    candidates = [
        DATA_ROOT / xlsx / basename,
        REPO_ROOT / "data" / "bing21_grouped" / xlsx / basename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _build_single_column_chunk(csv_path: Path, column: str) -> CsvChunk:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if column not in df.columns:
        raise pytest.skip.Exception(
            f"column '{column}' not found in {csv_path.name} "
            f"(have: {list(df.columns)[:8]}...)"
        )
    sub = df[[column]].copy()
    sub.index = range(len(sub))
    return CsvChunk(
        source=csv_path,
        start_row=0,
        end_row=len(sub),
        headers=[column],
        frame=sub,
        column_mode=COLUMN_MODE_SINGLE,
    )


def _query_nfi(llm: LLMClient, taxonomy, chunk: CsvChunk):
    nfi = taxonomy.get("number_format_inconsistency")
    rendered = taxonomy.render_user_prompt(nfi, chunk)
    response = llm.query(
        MODEL,
        rendered,
        reasoning_effort=REASONING_EFFORT,
        reasoning_summary="concise",
        system_prompt=taxonomy.system_prompt,
    )
    if not response.ok:
        pytest.fail(f"LLM call failed: {response.error}")
    issues, parse_err = parse_llm_json(response.text)
    assert parse_err is None, (
        f"Model returned unparseable JSON. parse_err={parse_err!r} "
        f"raw[:300]={response.text[:300]!r}"
    )
    return issues, response.text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename,xlsx,column,kind,expect_empty",
    SINGLETON_ARMS,
    ids=[f"{b}::{c}" for b, _, c, _, _ in SINGLETON_ARMS],
)
def test_nfi_singleton_arm_under_new_prompt(
    llm, taxonomy, basename, xlsx, column, kind, expect_empty
):
    csv_path = _resolve_csv(basename, xlsx)
    if csv_path is None:
        pytest.skip(f"source CSV not present locally: {xlsx}/{basename}")

    chunk = _build_single_column_chunk(csv_path, column)
    issues, raw = _query_nfi(llm, taxonomy, chunk)

    # Diagnostic info captured in the failure message.
    summary = (
        f"\n  source        = {basename}"
        f"\n  column        = {column}"
        f"\n  kind          = {kind}"
        f"\n  rows seen     = {chunk.row_count}"
        f"\n  issues count  = {len(issues)}"
        f"\n  issue sample  = {issues[:1]}"
        f"\n  raw[:200]     = {raw[:200]!r}"
    )

    if expect_empty:
        assert issues == [], (
            "Patched NFI prompt still flagged a code/identifier column "
            "as a number-format problem. The QUANTITATIVE-COLUMN GATE "
            "must fast-skip these to {{\"issues\": []}}." + summary
        )
    else:
        # Sanity: prompt is well-formed and produced a parseable result.
        # We do NOT assert presence/absence of issues for legitimate
        # quantity columns - run-to-run variance is expected.
        assert isinstance(issues, list), summary
