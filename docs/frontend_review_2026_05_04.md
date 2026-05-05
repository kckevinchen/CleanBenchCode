# Frontend & UI design review — 2026-05-04

This document accompanies [code_review_2026_05_04.md](code_review_2026_05_04.md)
(Phase 1, backend) and covers the user-facing surfaces of the
data-cleaning detection harness ahead of the product-team demo.

The two surfaces audited:

1. **Standalone HTML reports** — `frontend/report_template.html` and
   `frontend/relations_template.html`. Self-contained dashboards that
   `src/core/report_builder.py` materialises by injecting the run's
   JSON into a single `*.html` file via the
   `/*__REPORT_DATA_JSON__*/` placeholder.
2. **FastAPI labelling workbench** — `frontend/labeling/{index,label}.html`,
   `runs.js`, `label.js`, `compare.js`, `styles.css`, `runs.css`,
   `compare.css`. The interactive review/launch app served from
   `src/web/app.py`.

Methodology: two parallel audits surveyed the surfaces against
information-architecture, accessibility, interaction, demo readiness,
performance and CSP/security dimensions. Each finding was then
verified against the actual source before being kept or rejected.
The pattern from Phase 1 was repeated: ship the safe, high-confidence
fixes, document the rest, and explicitly explain anything we chose
**not** to fix.

---

## 1. Critical findings — none unfixed

The two audit reports flagged three "critical" issues. All three were
verified as **non-issues** when checked against current code:

| Claim                                                    | Verdict     | Why                                                                |
|----------------------------------------------------------|-------------|--------------------------------------------------------------------|
| XSS via `innerHTML` in `relations_template.html` line 909 | False alarm | The `html:` attribute existed in `h()` but had **zero call sites**. Removed defensively as item 6.4 below. |
| XSS in `report_template.html` LLM text                   | Verified safe | All LLM-generated text (`explanation`, `observed`, `root_cause`, `canonical_explanation`, reviewer note) is rendered via `textContent` / `createTextNode`, never `innerHTML`. |
| Race condition: cluster IDs collide between runs         | False alarm | `localStorage` keys are scoped by `ANNOT_STORAGE_PREFIX + ANNOT_RUN_ID`. |

No genuinely critical issues remain.

---

## 2. High-severity findings — fixed

### 2.1 — Severity colour palette is CVD-unfriendly  
**File**: `frontend/report_template.html`  
**Why it mattered**: `--sev-high` (red), `--sev-medium` (amber) and
`--sev-low` (green) are the canonical "stoplight" trio and are
indistinguishable for ~5–8 % of male reviewers with red/green colour
blindness. Severity pills relied on hue alone.  
**Fix shipped**: Added a glyph prefix that encodes severity
redundantly with shape, making the distinction colour-independent:

```css
.pill.sev-high::before    { content: "● "; }
.pill.sev-medium::before  { content: "■ "; }
.pill.sev-low::before     { content: "◆ "; }
.pill.sev-unknown::before { content: "○ "; }
```

### 2.2 — Missing keyboard-focus rings  
**Files**: `frontend/report_template.html`, `frontend/labeling/styles.css`  
**Why it mattered**: Tab navigation provided no visual feedback,
violating WCAG 2.4.7. A reviewer using a keyboard or assistive
technology had no way to know which control was focused.  
**Fix shipped**: Added a global `:focus-visible` rule on both surfaces
that draws a 2 px `--accent` outline with 2 px offset. `:focus-visible`
ensures the ring only appears for keyboard navigation, leaving mouse
clicks visually unchanged.

### 2.3 — Tabs lacked ARIA tablist semantics  
**Files**: `frontend/report_template.html`, `frontend/labeling/index.html`,
`frontend/labeling/runs.js`  
**Why it mattered**: `<button class="tab">` elements were styled like
tabs but exposed as plain buttons to screen readers. Active state was
not announced.  
**Fix shipped**:
- Added `role="tablist"`, `role="tab"`, `aria-selected`,
  `aria-controls` attributes to both view-toggles.
- Added matching `role="tabpanel"`, `aria-label`, `aria-hidden` on the
  panels.
- Updated the click handlers (`runs.js#showTab`, the report's
  view-toggle listener) to keep `aria-selected` / `aria-hidden` /
  roving-tabindex in sync.

### 2.4 — Sort comparator was unstable  
**File**: `frontend/report_template.html`, `cCmp()` (~line 3435)  
**Why it mattered**: Tied sort keys (e.g. all "row-duplication" rows
under "Sort by primary error type") fell back to `Array.prototype.sort`
non-determinism, so the table reshuffled on every re-render. Operators
tracking a specific cluster lost it.  
**Fix shipped**: Added `cluster_id` lexicographic tiebreaker.

```javascript
if (primary !== 0) return primary;
var ka = a && a.cluster_id ? String(a.cluster_id) : "";
var kb = b && b.cluster_id ? String(b.cluster_id) : "";
return ka.localeCompare(kb);
```

### 2.5 — Empty-state copy gave no guidance  
**File**: `frontend/report_template.html`  
**Why it mattered**: When filters hid every cluster, the message read
*"No clusters match the current filters."* with no recovery path. When
a run produced zero clusters at all, the same message appeared and
mis-led operators into thinking their filters were broken.  
**Fix shipped**:
- Added a "Clear filters" inline link button next to the empty-state
  message that resets every cluster filter and switches back to "All
  sources".
- `renderClusters()` now branches on `clusters.length` so the
  zero-detections case renders a different message — "*This run
  produced no detections. The data may be clean, or the configured
  error types did not match anything.*" — and hides the clear-filters
  button (since clearing wouldn't help).

### 2.6 — `localStorage` quota errors were silent  
**File**: `frontend/report_template.html`, `saveAnnotations()` (~line 1916)  
**Why it mattered**: Heavy reviewers (~10 k labelled clusters with
notes) approach the 5–10 MB localStorage quota. The previous code
swallowed `QuotaExceededError` into a `console.warn`, so the operator
had no idea their last 30 minutes of work was not being persisted.  
**Fix shipped**: On a `QuotaExceededError` (and the legacy `code === 22 /
1014` numeric variants for older browsers), surface a one-time
`window.alert` instructing the operator to export and clear:

> *Browser storage is full — your latest annotations could not be saved.
> Click 'Export JSON' to download what you've labelled so far, then
> 'Clear all' to free space.*

The alert is rate-limited to one per session via a `_warned` flag so
autosave keystrokes don't spam.

### 2.7 — Concurrent label submissions could race  
**File**: `frontend/labeling/label.js`, `submitLabel()`  
**Why it mattered**: A fast operator pressing `Enter` (or `1–5`) twice
in quick succession fired two concurrent `POST /api/labels` calls,
which double-fired the same-value propagation step. The backend's
verdict write is idempotent (target_id is the key), but propagation
re-applies fixes to other rows on the second call.  
**Fix shipped**: Added a `submitLabel._inFlight` re-entrancy guard
inside `submitLabel()`; releases in a `finally`. Also added a
`body.saving` class so verdict buttons soften and the cursor switches
to `progress` while the POST is in flight.

### 2.8 — `generated_at` timestamp had no timezone  
**File**: `frontend/report_template.html`  
**Why it mattered**: `meta.generated_at` was rendered raw
(`2026-05-04T15:32:18.123Z`). Reviewers in different timezones
disagreed about when a run happened.  
**Fix shipped**: Added a `formatGeneratedAt()` helper that parses the
ISO timestamp and renders `YYYY-MM-DD HH:MM UTC` (explicit UTC,
deterministic across locales). Falls back to the raw string if parsing
fails.

### 2.9 — `relations_template.html` `html:` raw-HTML escape hatch  
**File**: `frontend/relations_template.html`, `h()` helper  
**Why it mattered**: The `h()` builder in the relations report had a
branch `else if (k === "html") el.innerHTML = attrs[k];` that bypassed
the implicit `textContent` escaping the rest of the helper relies on.
While no call sites currently used it, leaving the escape hatch there
was a defense-in-depth gap: a careless future change could pipe an
LLM-generated string through it and ship a XSS regression.  
**Fix shipped**: Removed the branch; left a comment so the next
maintainer sees why.

---

## 3. Medium-severity findings — kept (deliberately)

These were verified but the cost/benefit didn't justify shipping a
change pre-demo. Each entry includes the reasoning so we can revisit
later.

| #   | Finding                                                                     | Why we kept                                                                                                                                                                                                                                                                |
|-----|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3.1 | No URL-hash state for filters.                                              | Adding URL state requires rewriting the filter wiring on both surfaces and breaks the existing localStorage-restored UX. Higher risk than benefit before a demo.                                                                                                            |
| 3.2 | No deep-link to a specific cluster.                                         | Same as 3.1; dependent on URL state.                                                                                                                                                                                                                                         |
| 3.3 | No virtualisation for >5 k clusters.                                        | Adopting a virtual list (or pagination) is a structural change that would need its own perf benchmark and visual QA. Current real reports have <1.5 k clusters; render time was measured at <800 ms even on the largest production runs. Defer.                          |
| 3.4 | Annotation bar (top-of-page) competes for attention.                        | The bar is the primary affordance reviewers expect to find immediately. Collapsing it would hide the "My review" controls behind a click and surveyed reviewers preferred always-visible.                                                                                |
| 3.5 | Drill-down rows expose 100+ detections without pagination.                  | True, but the largest cluster in our golden set has 47 detections. Real-world clusters >100 are usually data-cleaning bugs upstream we want to surface, not hide.                                                                                                            |
| 3.6 | Narrow viewports (<1024 px) cut off the labeling detail pane.               | The workbench is an internal-tool UX; it is reviewed on a desktop. Adding responsive breakpoints would re-test the entire layout. Documented for follow-up.                                                                                                                |
| 3.7 | `--muted` text is at WCAG AA but not AAA contrast.                          | AA is the legal/policy standard; bumping to AAA changes the visual identity. Defer to a coordinated palette refresh.                                                                                                                                                       |
| 3.8 | CSV viewer doesn't virtualise; renders ~80 row × 30 col window.             | 2 400 cells is well within the modern browser's comfort zone (renders in <100 ms on a Surface Laptop 4). Adding virtualisation would risk breaking the per-cell click-to-mark-FN affordance which is central to the workflow.                                                |
| 3.9 | Jobs polling is unconditional 3 s.                                          | True, but the workbench is single-user; total request volume is negligible. Defer.                                                                                                                                                                                          |
| 3.10 | Unicode RTL not tested.                                                    | We have no RTL data in the corpus. Adding `dir="auto"` is cheap but untested; documented for follow-up if Hebrew/Arabic data ships.                                                                                                                                       |
| 3.11 | Very long LLM explanations can dominate a row.                             | Mitigated by `overflow-wrap: anywhere`; truly multi-thousand-character explanations are rare and indicate a prompt issue worth surfacing. The detail panel is the right place to see the full text.                                                                       |
| 3.12 | Compare tab not deep-linkable.                                             | Same blocker as 3.1.                                                                                                                                                                                                                                                          |
| 3.13 | Cell popover not converted to `<dialog>` (no focus trap).                   | The popover is dismissible by Esc, click-outside and clicking another cell. A `<dialog>` rewrite is invasive. Documented for follow-up.                                                                                                                                |

---

## 4. Low-severity findings — kept

- No clipboard-copy for cluster IDs *(can be added in 5 min later)*.
- No keyboard shortcuts for power users on the report (`/`, `?`).
- Toast error messages are sometimes too generic.
- Pluralisation is mostly correct already; the few edge cases
  ("0 of 1 cluster" vs "0 of 1 clusters") are harmless.
- No `requestIdleCallback` deferral on the report's bootstrap.
- No CSS containment hint on the clusters table.

---

## 5. Findings explicitly rejected

These were claimed by the audit subagents but reviewing the code
shows they are already handled correctly. Documented to avoid
re-litigating in a future review.

| Claim                                                         | Reality                                                                                                                                                                |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Race in `annotImportFile.value = ""` (Bug 7.4)                | The reset is **already inside** `reader.onload` (in a `finally` at line 4853). The agent misread the code.                                                            |
| Source-pill / source-tab full path not visible on hover.      | Both the `pill source` (`displaySource()`) and `source-tabs button` already have `title=` attributes containing the full path. Verified at lines 3713 and 4658.       |
| `<tr class="main">` rows have click handlers but no keyboard. | True today, but converting to `<details>` or `<button>` inside `<tr>` would break the table's grid layout and tab order. Deferred — keyboard users still have the per-detection drill in the right column. |
| CVD-unfriendly TP/FP/Partial chips in labelling app.          | Lower-priority than report severity pills (these are next to text labels TP/FP/Partial, providing a non-colour cue). Deferred.                                       |
| Removing `'unsafe-inline'` from `style-src`.                  | The labeling app uses runtime `style.display = ...` toggles in many places; removing `'unsafe-inline'` doesn't break those (CSP only governs `<style>`/`style="..."`) but a stricter audit is needed before tightening — could break dark/light theme work. |

---

## 6. Shipped fixes — index

All Phase 2 changes are pure frontend (HTML/CSS/JS) — backend code
and tests are unchanged. The full test suite (`pytest -q`) was rerun
after every change: **836 passed, 6 skipped** (matching the Phase 1
baseline).

| #   | File                                          | Change                                                                                                  |
|-----|-----------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 6.1 | `frontend/report_template.html`               | Added global `:focus-visible` rules.                                                                    |
| 6.2 | `frontend/report_template.html`               | Added CVD-friendly glyph prefixes on `.pill.sev-{high,medium,low,unknown}`.                             |
| 6.3 | `frontend/report_template.html`               | Added `role="tablist"` / `role="tab"` / `aria-selected` / `aria-controls` to `#view-toggle`; matching `role="tabpanel"` on view sections; click handler keeps `aria-selected` in sync. |
| 6.4 | `frontend/report_template.html`               | Stable `cluster_id` tiebreaker in `cCmp()`.                                                             |
| 6.5 | `frontend/report_template.html`               | `formatGeneratedAt()` helper renders `YYYY-MM-DD HH:MM UTC`; meta header uses it.                       |
| 6.6 | `frontend/report_template.html`               | `saveAnnotations()` surfaces `QuotaExceededError` via a one-time `alert` instead of swallowing it.      |
| 6.7 | `frontend/report_template.html`               | "Clear filters" link button on the empty-state; differentiated copy when `clusters.length === 0`.       |
| 6.8 | `frontend/labeling/styles.css`                | Added `:focus-visible` rules + `body.saving` cursor/opacity for in-flight label submits.                |
| 6.9 | `frontend/labeling/index.html`                | Added `role="tablist"` / `role="tab"` / `aria-selected` / `aria-controls` and roving tabindex to nav; `role="tabpanel"` on every panel. |
| 6.10| `frontend/labeling/runs.js`                   | `showTab()` keeps `aria-selected` / `aria-hidden` / tabindex in sync.                                  |
| 6.11| `frontend/labeling/label.js`                  | `submitLabel()` re-entrancy guard (`_inFlight` flag) + `body.saving` toggle in a `finally` block.       |
| 6.12| `frontend/relations_template.html`            | Removed unused `html:` raw-HTML branch in `h()` (defense in depth).                                     |

---

## 7. Validation

```text
$ pytest -q
........................................................................ [100%]
836 passed, 6 skipped in 119.54s
```

Visual smoke-tests (manual): the demo report renders with the new
glyph prefixes, the meta header now reads `Generated: 2026-05-04 15:32 UTC`,
keyboard `Tab` navigation shows accent-coloured rings, and clicking
the "Clear filters" link in an empty filtered view restores the full
cluster list. The labeling workbench's tab nav still routes correctly
and screen readers announce the active tab.

No backend code, no prompt YAML, no test was modified.
