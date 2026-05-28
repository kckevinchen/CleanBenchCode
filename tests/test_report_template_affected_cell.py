"""Guard tests for the report's per-cluster "affected cell" UI.

The report template renders each finding cluster as a row in a big
table; the user-visible improvement here is a prominent banner at
the top of the drill-down that names the offending
``source / row / column = value`` and (for inconsistency) deep-links
to the column board. We also paint the affected column(s) yellow
inside the existing "Source row values" grid.

These are source-inspection tests: we only verify the helpers /
CSS hooks / wiring exist. Visual rendering is covered by manual
smoke + the existing report screenshots.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_TEMPLATE = PROJECT_ROOT / "frontend" / "report_template.html"


def _html() -> str:
    return REPORT_TEMPLATE.read_text(encoding="utf-8")


class TestAffectedColumnHelpers:
    def test_cluster_affected_columns_helper_present(self) -> None:
        """Single source of truth for the affected-column set; must
        union ``cluster.columns`` and every detection's ``columns``
        (so single-type clusters and multi-detection clusters both
        light up the right cells)."""
        html = _html()
        assert "function clusterAffectedColumns(cluster)" in html
        assert "cluster.columns" in html
        assert "cluster.detections" in html

    def test_render_row_values_grid_accepts_affected_set(self) -> None:
        """The grid renderer must take an explicit affected-cols arg
        so the same helper can be called from places that don't know
        about clusters."""
        html = _html()
        assert "function renderRowValuesGrid(rowValues, affectedCols)" in html

    def test_drill_section_passes_affected_set(self) -> None:
        """The 'Source row values' drill section must thread the
        affected set into the grid; the banner above the section uses
        the same set so they stay in sync."""
        html = _html()
        assert "var affectedCols = clusterAffectedColumns(c);" in html
        assert "renderRowValuesGrid(c.row_values, affectedCols)" in html


class TestAffectedCellBanner:
    def test_banner_helper_present(self) -> None:
        html = _html()
        assert "function renderAffectedCellBanner(cluster)" in html

    def test_banner_appended_first_in_drill(self) -> None:
        """The banner is the first thing the operator sees on expand;
        verify it's appended BEFORE the Synthesis section."""
        html = _html()
        idx_banner = html.find("var banner = renderAffectedCellBanner(c);")
        idx_synth = html.find('h("h3", { text: "Synthesis (Phase B)" })')
        assert idx_banner != -1
        assert idx_synth != -1
        assert idx_banner < idx_synth, (
            "Affected-cell banner must be appended before the "
            "Synthesis drill-section so it surfaces first on expand."
        )

    def test_banner_deep_links_to_column_board(self) -> None:
        """Inconsistency clusters get a same-origin link to
        /ui/column_board.html?run=&source=&column= so the operator
        can hop from the report straight to labelling."""
        html = _html()
        assert "/ui/column_board.html?run=" in html
        assert '"&source=" + encodeURIComponent(src)' in html
        assert '"&column=" + encodeURIComponent(cols[0])' in html
        # Only inconsistency clusters get the link (other error types
        # don't have a per-column board).
        assert 'etypes.indexOf("inconsistency")' in html

    def test_banner_shows_inferred_marker(self) -> None:
        """Synthesised sibling rows have empty row_values; the banner
        must say so explicitly (else the operator wonders why no
        value is printed)."""
        html = _html()
        assert "inferred sibling" in html


class TestAffectedCellStyles:
    def test_affected_highlight_in_css(self) -> None:
        """Affected-column rows in the row-values grid get a yellow
        background; verify the selectors exist."""
        html = _html()
        assert ".row-values-grid .v.affected" in html
        assert ".row-values-grid .k.affected" in html

    def test_banner_styles_in_css(self) -> None:
        html = _html()
        assert ".affected-cell-banner" in html
        assert ".acb-loc" in html
        assert ".acb-col" in html
        assert ".acb-val" in html
        assert ".acb-link" in html
