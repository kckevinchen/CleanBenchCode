"""Regression tests for the auto-zoom viewport scaling.

The data-cleaning workbench is a fixed 3-pane layout designed against
a ~1600x900 viewport. On 1366x768 / 1280x720 laptops the centre CSV
viewer gets squeezed and operators have to horizontally scroll inside
cells. To avoid that, every entry-point HTML loads (or inlines) the
viewport-zoom logic that sets ``html { zoom: N }`` from the actual
viewport size.

These tests are the regression guard: if anyone removes the script
tag from a page (or breaks the constants the script depends on),
CI fails with a clear pointer to this file.

They DON'T spin up a browser - they verify:

* The script file exists, exports the expected ``__workbenchZoom``
  API, and uses the documented design-viewport constants.
* Each of the four entry-point pages either references the external
  script (labelling pages) or inlines the same logic (self-contained
  file:// templates).
* The script is loaded synchronously from ``<head>`` BEFORE the
  stylesheets so the first paint is at the right scale.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
LABELING = FRONTEND / "labeling"


# ---------------------------------------------------------------------------
# The external script (loaded by the labelling pages)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zoom_script_text() -> str:
    path = LABELING / "viewport-zoom.js"
    assert path.is_file(), f"missing zoom helper: {path}"
    return path.read_text(encoding="utf-8")


def test_zoom_script_uses_design_viewport_constants(zoom_script_text: str) -> None:
    """The script tunes against 1600x900. If the design viewport
    changes, update both this test and the inlined copies in the
    self-contained templates (relations_template.html /
    report_template.html) at the same time."""
    assert "DESIGN_WIDTH = 1600" in zoom_script_text
    assert "DESIGN_HEIGHT = 900" in zoom_script_text


def test_zoom_script_clamps_within_min_max(zoom_script_text: str) -> None:
    """Auto-zoom scales DOWN on small screens (so the centre CSV
    viewer doesn't squeeze) and UP on large screens (so 13px type
    doesn't disappear on 1440p / 4K monitors), clamped to a tested
    range. Power users can still pin a factor outside the auto-zoom
    range via ``__workbenchZoom.setOverride``."""
    assert "MAX_ZOOM = 1.5" in zoom_script_text
    assert "MIN_ZOOM = 0.65" in zoom_script_text


def test_zoom_script_exposes_devtools_api(zoom_script_text: str) -> None:
    """``window.__workbenchZoom`` lets operators override the zoom
    from devtools without editing files. The labelling docs reference
    this API."""
    assert "window.__workbenchZoom" in zoom_script_text
    assert "setOverride" in zoom_script_text
    assert "getCurrent" in zoom_script_text


def test_zoom_script_persists_override_in_localStorage(
    zoom_script_text: str,
) -> None:
    """Overrides survive page reloads via localStorage; the storage
    key is shared with the inlined copies in the templates."""
    assert "workbench:zoom-override" in zoom_script_text


def test_zoom_script_publishes_vh_scale_variable(zoom_script_text: str) -> None:
    """The script must publish ``--vh-scale = 1/zoom`` (and the same
    for ``--vw-scale``) on the documentElement so CSS rules using
    ``calc(100vh * var(--vh-scale, 1))`` can inflate their logical
    height to compensate for the visual scale-down. Without this the
    body paints at ``zoom * 100vh`` and leaves empty space below the
    grid - see the html, body rule in styles.css."""
    assert "--vh-scale" in zoom_script_text
    assert "--vw-scale" in zoom_script_text
    assert "1 / z" in zoom_script_text


# ---------------------------------------------------------------------------
# CSS uses --vh-scale wherever it uses full-viewport sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "css_path",
    [
        LABELING / "styles.css",
        LABELING / "runs.css",
    ],
)
def test_full_viewport_rules_use_vh_scale(css_path: Path) -> None:
    """Every ``100vh`` declaration in the labelling stylesheets must
    pass through ``var(--vh-scale, 1)``. Otherwise CSS ``zoom`` paints
    the body at ``zoom * 100vh`` and the bottom of the viewport is
    left blank - exactly the bug this scaffold exists to prevent."""
    css = css_path.read_text(encoding="utf-8")
    # Strip C-style block comments before scanning so the rule
    # discussion in comments doesn't trip the regex.
    import re

    code = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    # Find every "100vh" / "50vh" etc. and assert it's inside a calc
    # that references --vh-scale. We allow ``height: auto`` overrides
    # (the @media tablet breakpoint resets to flow layout).
    for match in re.finditer(r"\d+vh\b", code):
        # Walk backwards to the surrounding declaration (up to ``;``
        # or ``{``). If --vh-scale is mentioned in that declaration,
        # the rule is compensated; otherwise it's a bare ``100vh``.
        start = max(
            code.rfind(";", 0, match.start()),
            code.rfind("{", 0, match.start()),
        )
        decl = code[start + 1 : match.end() + 60]
        assert "--vh-scale" in decl, (
            f"{css_path.name}: bare ``{match.group()}`` in declaration "
            f"``{decl.strip()[:120]}``. Wrap in "
            "``calc(... * var(--vh-scale, 1))`` so CSS zoom doesn't "
            "leave empty space at the viewport bottom."
        )


# ---------------------------------------------------------------------------
# Pages that load the script from /ui/viewport-zoom.js
# ---------------------------------------------------------------------------


LABELING_PAGES = ["index.html", "label.html"]


@pytest.mark.parametrize("page", LABELING_PAGES)
def test_labeling_page_loads_zoom_script(page: str) -> None:
    """Each labelling page must include the zoom helper. Loaded
    synchronously from ``<head>`` BEFORE the stylesheets so the
    first paint is at the right scale (no FOUC)."""
    html = (LABELING / page).read_text(encoding="utf-8")
    assert '<script src="./viewport-zoom.js"></script>' in html, (
        f"{page} no longer loads ./viewport-zoom.js - the auto-zoom for "
        "low-resolution screens is broken; restore the script tag in "
        "<head> BEFORE the stylesheets."
    )


@pytest.mark.parametrize("page", LABELING_PAGES)
def test_zoom_script_loads_before_stylesheets(page: str) -> None:
    """If the zoom tag falls below ``<link rel='stylesheet'>``, the
    page paints once at zoom=1 and then jumps - the FOUC the script
    was supposed to prevent. Enforce script-before-styles ordering."""
    html = (LABELING / page).read_text(encoding="utf-8")
    script_idx = html.find('viewport-zoom.js')
    style_idx = html.find('<link rel="stylesheet"')
    assert script_idx != -1, f"{page} missing viewport-zoom.js script tag"
    assert style_idx != -1, f"{page} missing stylesheet link"
    assert script_idx < style_idx, (
        f"{page}: <script src='./viewport-zoom.js'> must precede "
        "the first <link rel='stylesheet'> to avoid first-paint FOUC."
    )


# ---------------------------------------------------------------------------
# Self-contained file:// templates (inline copy of the same logic)
# ---------------------------------------------------------------------------


SELF_CONTAINED_TEMPLATES = [
    FRONTEND / "relations_template.html",
    FRONTEND / "report_template.html",
]


@pytest.mark.parametrize("path", SELF_CONTAINED_TEMPLATES)
def test_self_contained_template_inlines_zoom_logic(path: Path) -> None:
    """The file:// reports can't reference external scripts (CSP +
    portability), so they inline the same zoom logic. Verify the
    inlined copy is still there and references the shared storage
    key + constants."""
    html = path.read_text(encoding="utf-8")
    assert "DESIGN_W = 1600" in html, (
        f"{path.name} no longer inlines the viewport-zoom logic; "
        "self-contained file:// reports must scale themselves on "
        "low-resolution screens."
    )
    assert "DESIGN_H = 900" in html
    assert "workbench:zoom-override" in html
    assert "window.__workbenchZoom" in html


@pytest.mark.parametrize("path", SELF_CONTAINED_TEMPLATES)
def test_self_contained_template_zoom_runs_before_styles(path: Path) -> None:
    html = path.read_text(encoding="utf-8")
    # The inline script must appear before the <style> block for the
    # same FOUC reason as the labelling pages.
    script_idx = html.find("__workbenchZoom")
    style_idx = html.find("<style>")
    assert script_idx != -1, f"{path.name} missing inlined zoom script"
    assert style_idx != -1, f"{path.name} missing <style> block"
    assert script_idx < style_idx, (
        f"{path.name}: inlined zoom <script> must precede <style> to "
        "avoid first-paint FOUC."
    )
