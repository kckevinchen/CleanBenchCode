// Compare a labelled benchmark JSONL (under ``label_result/``) against an
// external detector's TSV (under ``external/``). One row per per-cell join
// state -- false_negative / false_positive / true_positive / agreed_clean /
// ambiguous_in_bench / mixed_conflict / unmatched_prediction. The drawer
// lets the operator EDIT the benchmark JSONL row in place so labelling
// mistakes can be fixed without leaving the browser.
//
// Backed by /api/external-compare/{sources,diff,source-frame,benchmark/update}.

import { $, $$, api, escapeHtml as escape } from "./shared/utils.js";

// Short, human-readable description for each state -- used both in the
// breakdown table and the filter dropdown's <option>s.
const STATE_INFO = {
  false_negative: {
    label: "false negative",
    desc: "bench says <b>error</b>, external did NOT flag it",
    cls: "ext-state-fn",
  },
  false_positive: {
    label: "false positive",
    desc: "bench says <b>clean</b>, external flagged it",
    cls: "ext-state-fp",
  },
  true_positive: {
    label: "true positive",
    desc: "both agree the cell is in error",
    cls: "ext-state-tp",
  },
  agreed_clean: {
    label: "agreed clean",
    desc: "both treat the cell as clean / both ignore it",
    cls: "ext-state-clean",
  },
  ambiguous_in_bench: {
    label: "ambiguous in bench",
    desc: "bench label is <b>ambiguous</b> -- review manually",
    cls: "ext-state-amb",
  },
  mixed_conflict: {
    label: "mixed conflict",
    desc: "bench has both error AND clean rows for this cell",
    cls: "ext-state-mixed",
  },
  unmatched_prediction: {
    label: "unmatched prediction",
    desc: "external flagged a (source, column, value) the bench never labelled",
    cls: "ext-state-unmatched",
  },
  // Pseudo-state: a subset of unmatched_prediction whose value DOES
  // appear in the source CSV. These are the most useful candidates
  // for expanding the labelled benchmark, hence the dedicated row.
  // Only populated when the operator picked a dataset.
  bench_gap: {
    label: "bench coverage gap",
    desc: "external flagged a real CSV cell that hasn't been labelled in the bench yet",
    cls: "ext-state-unmatched",
  },
  unmatched_hallucination: {
    label: "unmatched (no CSV hit)",
    desc: "external flagged a value that doesn't appear anywhere in the source CSV",
    cls: "ext-state-unmatched",
  },
};

// The states that are "actionable" -- where the operator should typically
// look first. Excludes true_positive + agreed_clean (those are agreements).
const ACTIONABLE = new Set([
  "false_negative",
  "false_positive",
  "ambiguous_in_bench",
  "mixed_conflict",
  "unmatched_prediction",
]);

const ERROR_TYPE_VOCAB = [
  "typo",
  "incomplete_value",
  "missing_value",
  "format_inconsistency",
  "semantic_incompatible",
  "out_of_domain",
  "unit_mismatch",
  "encoding_artifact",
  "case_mismatch",
];

const state = {
  loadedPickers: false,
  sources: null,           // /api/external-compare/sources payload
  view: null,              // /api/external-compare/diff payload
  page: { offset: 0, limit: 50 },
  // Default state filter is empty (= show all cells). Previously this
  // defaulted to "actionable" which hides true_positive + agreed_clean;
  // operators kept hitting "my row disappeared" because numeric-only
  // cells collapse to a single agreed_clean row and were thus invisible.
  filters: { state: "", source: "", search: "" },
};

// =====================================================================
// Sub-mode toggle
// =====================================================================

function switchMode(mode) {
  $$(".cmp-mode").forEach((btn) => {
    const active = btn.dataset.cmpMode === mode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  $$(".cmp-mode-panel").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.cmpModePanel !== mode);
  });
  // Lazy-load pickers when external mode opens for the first time.
  if (mode === "external" && !state.loadedPickers) loadPickers();
}

// =====================================================================
// Picker population
// =====================================================================

async function loadPickers() {
  let data;
  try { data = await api("/api/external-compare/sources"); }
  catch (err) { setMeta(`Error: ${err.message}`); return; }
  state.sources = data;
  // Benchmarks. Prefer ``.canonical.jsonl`` builds at the top of the
  // list -- they're the ones with the corrected_value column normalized.
  const benchOpts = data.benchmarks.slice().sort((a, b) => {
    const aCan = a.name.includes(".canonical");
    const bCan = b.name.includes(".canonical");
    if (aCan !== bCan) return aCan ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  $("#ext-benchmark").innerHTML = `<option value="">(pick a benchmark)</option>` +
    benchOpts.map((b) => {
      const tag = b.name.includes(".canonical") ? " · canonical" : "";
      return `<option value="${escape(b.name)}">${escape(b.name)}${escape(tag)}</option>`;
    }).join("");
  $("#ext-predictions").innerHTML = `<option value="">(pick an external run)</option>` +
    buildExternalRunOptions(data.external_runs);
  $("#ext-dataset").innerHTML = `<option value="">(skip — no CSV context)</option>` +
    data.datasets.map((d) => `<option value="${escape(d.name)}">${escape(d.name)}</option>`).join("");
  // Sensible default for the dataset most-used here.
  const defaultDataset = data.datasets.find((d) => d.name === "curated_bing21_200");
  if (defaultDataset) $("#ext-dataset").value = defaultDataset.name;
  state.loadedPickers = true;
  updateButtonState();
}

/**
 * Render the external-runs <select> body, grouping nested files under
 * <optgroup>s. The backend returns each TSV as
 * ``{ name, basename, group }`` where ``name`` is the path relative to
 * ``external/`` (e.g. ``GPT_score/output_xxx.scoring.tsv``) and ``group``
 * is the parent directory (empty string for root-level files). Root files
 * are listed flat; everything else is bucketed by ``group`` so the
 * dropdown stays browsable as more batches land.
 */
function buildExternalRunOptions(runs) {
  const rootFiles = runs.filter((r) => !r.group);
  const grouped = new Map();
  for (const r of runs) {
    if (!r.group) continue;
    if (!grouped.has(r.group)) grouped.set(r.group, []);
    grouped.get(r.group).push(r);
  }
  const parts = [];
  for (const r of rootFiles) {
    // value MUST be the full rel_path so the diff endpoint can resolve it.
    parts.push(`<option value="${escape(r.name)}">${escape(r.basename || r.name)}</option>`);
  }
  for (const [groupName, items] of grouped) {
    parts.push(`<optgroup label="${escape(groupName)} (${items.length})">` +
      items.map((r) =>
        `<option value="${escape(r.name)}">${escape(r.basename || r.name)}</option>`
      ).join("") +
      `</optgroup>`);
  }
  return parts.join("");
}

function updateButtonState() {
  const ok = !!($("#ext-benchmark").value && $("#ext-predictions").value);
  $("#ext-load").disabled = !ok;
}

// =====================================================================
// Diff fetch + render
// =====================================================================

async function loadDiff() {
  const benchmark = $("#ext-benchmark").value;
  const predictions = $("#ext-predictions").value;
  const includeTrivial = $("#ext-include-trivial").checked;
  const dataset = $("#ext-dataset").value;
  if (!benchmark || !predictions) return;
  setMeta("Loading…");
  $("#ext-results").hidden = true;
  $("#ext-warn").hidden = true;
  // ``dataset`` is forwarded so the backend can scan the underlying
  // CSV for each unmatched prediction and tell the operator apart
  // bench-coverage gaps (value IS in the CSV) from hallucinations
  // (value NOT in the CSV). When the operator skipped the dataset
  // picker we just don't enrich -- the grid still works, the
  // ``bench_gap`` pseudo-state simply stays at zero.
  let url = `/api/external-compare/diff?benchmark=${encodeURIComponent(benchmark)}` +
    `&predictions=${encodeURIComponent(predictions)}&include_trivial=${includeTrivial}`;
  if (dataset) url += `&dataset=${encodeURIComponent(dataset)}`;
  let data;
  try { data = await api(url); }
  catch (err) { setMeta(`Error: ${err.message}`); return; }
  state.view = data;
  state.page = { offset: 0, limit: 50 };
  // Preserve the operator's last state-filter choice across reloads, but
  // leave the empty default (= show all) alone -- don't promote to
  // "actionable" automatically; that hides agreed cells and confuses
  // people looking for a specific row.
  state.filters.source = "";
  state.filters.search = "";
  $("#ext-filter-state").value = state.filters.state;
  $("#ext-filter-source").value = "";
  $("#ext-filter-search").value = "";
  renderMeta();
  renderMetrics(data.summary);
  renderStateTable(data.states);
  renderCanonBar();
  renderCells();
  $("#ext-results").hidden = false;
}

function setMeta(html) { $("#ext-meta").innerHTML = html; }

function renderMeta() {
  const v = state.view;
  setMeta(
    `<b>${escape(v.benchmark)}</b> vs <b>${escape(v.predictions)}</b> · ` +
    `${v.summary.benchmark_total_cells} cells in benchmark · ` +
    `${v.summary.predictions_in_scope} of ${v.summary.predictions_total} predictions in scope` +
    (v.summary.trivial_skipped ? ` · ${v.summary.trivial_skipped} trivial diffs skipped` : "")
  );
}

function renderMetrics(s) {
  const fmt = (x) => (x == null ? "—" : (typeof x === "number" ? x.toFixed(3) : String(x)));
  $("#ext-m-precision").textContent = fmt(s.precision);
  $("#ext-m-recall").textContent = fmt(s.recall);
  $("#ext-m-f1").textContent = fmt(s.f1);
  $("#ext-m-errors").textContent = `${s.benchmark_error_cells} / ${s.benchmark_total_cells}`;
  $("#ext-m-preds").textContent = `${s.predictions_in_scope} / ${s.predictions_total}`;
  const corrTotal = s.correction_exact_match_total ?? 0;
  const corrHit = s.correction_exact_match ?? 0;
  $("#ext-m-corr").textContent = corrTotal
    ? `${corrHit} / ${corrTotal} (${(corrHit / corrTotal * 100).toFixed(0)}%)`
    : "—";
}

function renderStateTable(states) {
  const tbody = $("#ext-state-table tbody");
  // Show every known state (in the canonical ordering) even if its
  // count is zero -- that way the operator can see "ah, no FPs here".
  // ``bench_gap`` / ``unmatched_hallucination`` are pseudo-states
  // computed by the backend when a dataset was provided; we only
  // render them if the operator opted in (the backend signals
  // ``summary.csv_resolver_used`` in that case).
  const baseOrder = [
    "false_negative", "false_positive", "ambiguous_in_bench",
    "mixed_conflict", "unmatched_prediction",
    "true_positive", "agreed_clean",
  ];
  const csvOn = !!state.view?.summary?.csv_resolver_used;
  // Inject the pseudo-states right after unmatched_prediction so
  // they read like a sub-breakdown of it.
  const order = [];
  for (const k of baseOrder) {
    order.push(k);
    if (k === "unmatched_prediction" && csvOn) {
      order.push("bench_gap", "unmatched_hallucination");
    }
  }
  tbody.innerHTML = order.map((key) => {
    const info = STATE_INFO[key];
    const n = states[key] || 0;
    const dim = n === 0 ? ' style="opacity:0.5;"' : "";
    // Indent the two pseudo-states so they look subordinate to
    // unmatched_prediction in the breakdown table.
    const isPseudo = key === "bench_gap" || key === "unmatched_hallucination";
    const indent = isPseudo ? ' style="padding-left:18px;font-style:italic;"' : "";
    return `<tr${dim}>
      <td${indent}><span class="ext-state-badge ${info.cls}">${escape(info.label)}</span></td>
      <td class="num">${n > 0
        ? `<span class="clickable" data-state="${key}">${escape(String(n))}</span>`
        : `<span>${escape(String(n))}</span>`}</td>
      <td>${info.desc}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".clickable").forEach((el) => {
    el.addEventListener("click", () => {
      state.filters.state = el.dataset.state;
      $("#ext-filter-state").value = el.dataset.state;
      state.page.offset = 0;
      renderCells();
    });
  });
}

function filteredCells() {
  if (!state.view) return [];
  const f = state.filters;
  const needle = f.search.trim().toLowerCase();
  return state.view.cells.filter((c) => {
    if (f.state === "actionable") {
      if (!ACTIONABLE.has(c.state)) return false;
    } else if (f.state === "bench_gap") {
      // Subset of unmatched_prediction whose value IS in the CSV --
      // the actionable candidates for adding to the benchmark.
      if (c.state !== "unmatched_prediction") return false;
      if (c.csv_match_status !== "in_csv") return false;
    } else if (f.state === "unmatched_hallucination") {
      if (c.state !== "unmatched_prediction") return false;
      if (c.csv_match_status !== "not_in_csv") return false;
    } else if (f.state && c.state !== f.state) {
      return false;
    }
    if (f.source && !c.source.toLowerCase().includes(f.source.toLowerCase())) return false;
    if (needle && !String(c.original_value ?? "").toLowerCase().includes(needle)) return false;
    return true;
  });
}

function renderCells() {
  const all = filteredCells();
  $("#ext-cell-count").textContent =
    `(${all.length.toLocaleString()} of ${state.view.cells.length.toLocaleString()})`;
  const { offset, limit } = state.page;
  const page = all.slice(offset, offset + limit);
  const tbody = $("#ext-cells-table tbody");
  if (!page.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="cmp-issue-empty">No cells match the current filters.</td></tr>`;
  } else {
    tbody.innerHTML = page.map((c, i) => {
      const info = STATE_INFO[c.state] || { label: c.state, cls: "" };
      const bench = c.benchmark || null;
      const pred = c.prediction || null;
      const cellKey = offset + i;
      // Bench label cell. Three flavours:
      //   * no bench record (unmatched_prediction) -> static "(not in bench)".
      //   * mixed_conflict -> static "mixed" + sibling rows (drawer is the
      //     right place to adjudicate; inline relabel of one row would
      //     hide the conflict).
      //   * everything else -> inline segmented control so a misclassified
      //     row can be flipped without opening the drawer. Triggers
      //     ``quickRelabel`` which POSTs to /benchmark/update.
      let benchLabel;
      const editable = bench && bench.record_index != null && bench.record_index >= 0;
      if (!bench) {
        benchLabel = `<span class="cmp-issue-empty">(not in bench)</span>`;
      } else if (bench.summary_label === "mixed_conflict") {
        const rows = (bench.all_rows || []).join(", ");
        benchLabel = `<span class="ext-lab ext-lab-ambiguous">mixed</span> <span class="muted" style="font-size:10.5px;">rows ${escape(rows)}</span>`;
      } else if (editable) {
        benchLabel = renderQuickRelabel(bench, cellKey);
      } else {
        benchLabel = `<span class="ext-lab ext-lab-${escape(bench.label || "")}">${escape(bench.label || "")}</span>`;
      }
      const benchCorr = bench && bench.corrected_value
        ? `<code>${escape(bench.corrected_value)}</code>`
        : `<span class="cmp-issue-empty">—</span>`;
      // External prediction column: show original→correction. The
      // external TSV has no error_type; presence of a record means
      // it flagged the cell, and we use the row colour for the verdict.
      let predOut;
      if (!pred) {
        predOut = `<span class="cmp-issue-empty">—</span>`;
      } else {
        const orig = escape(pred.predicted_original_value ?? "");
        const corr = escape(pred.predicted_correction ?? "");
        predOut = `<code>${orig}</code>${corr ? ` <span class="muted">→</span> <code>${corr}</code>` : ""}`;
        if (pred.trivial_prediction) predOut += ` <span class="muted" style="font-size:10px;">(trivial)</span>`;
      }
      const colShown = c.column === "(whole row)"
        ? `<span class="muted" style="font-style:italic;">whole row</span>`
        : `<code>${escape(c.column)}</code>`;
      // For unmatched_prediction we computed a sub-badge ("bench coverage
      // gap" vs "no CSV hit") if a dataset was provided. Surface it
      // inline so the operator can triage the unmatched list without
      // opening every drawer.
      let subBadge = "";
      if (c.state === "unmatched_prediction" && c.csv_match_status) {
        if (c.csv_match_status === "in_csv") {
          subBadge = ` <span class="ext-state-badge ext-state-unmatched" title="value found in CSV — labeling candidate" style="font-size:10px;">gap</span>`;
        } else if (c.csv_match_status === "not_in_csv") {
          subBadge = ` <span class="ext-state-badge ext-state-unmatched" title="value not in CSV — likely hallucination" style="font-size:10px;opacity:0.6;">no-csv</span>`;
        }
      }
      const editBtn = `<button type="button" class="ext-view-btn" data-cell-key="${cellKey}">${editable ? "View / edit" : "View"}</button>`;
      return `<tr data-cell-key="${cellKey}">
        <td><span class="ext-state-badge ${info.cls}">${escape(info.label)}</span>${subBadge}</td>
        <td><code>${escape(c.source)}</code></td>
        <td class="num">${escape(String(c.row ?? ""))}</td>
        <td>${colShown}</td>
        <td><code>${escape(String(c.original_value ?? "")).slice(0, 120)}</code></td>
        <td>${benchLabel}</td>
        <td>${benchCorr}</td>
        <td>${predOut}</td>
        <td>${editBtn}</td>
      </tr>`;
    }).join("");
    tbody.querySelectorAll(".ext-view-btn").forEach((btn) => {
      btn.addEventListener("click", () => openDrawer(all[+btn.dataset.cellKey]));
    });
    // Wire inline relabel pills. ``all`` is the filtered list -- we
    // index back into it via the data-cell-key set on each row + pill.
    tbody.querySelectorAll(".ext-quick-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cellKey = +btn.closest(".ext-quick-relabel").dataset.cellKey;
        const newLabel = btn.dataset.label;
        quickRelabel(all[cellKey], newLabel, btn);
      });
    });
  }
  renderPager(all.length);
}

// Tiny segmented control that lives in the Bench-label column. Each
// pill carries data-label; clicking one POSTs an update for just that
// field. The current label is .active; the others are neutral.
function renderQuickRelabel(bench, cellKey) {
  const current = bench.label || "";
  const pill = (lab, glyph, title) =>
    `<button type="button" class="ext-quick-btn ext-quick-${lab}${current === lab ? " active" : ""}"
       data-label="${lab}" title="${title}">${glyph}</button>`;
  return `<div class="ext-quick-relabel" data-cell-key="${cellKey}">
    ${pill("error", "E", "Mark as error")}
    ${pill("clean", "C", "Mark as clean")}
    ${pill("ambiguous", "A", "Mark as ambiguous")}
    <span class="ext-quick-status muted"></span>
  </div>`;
}

// Inline POST to /benchmark/update for a single label change. Updates
// the in-memory cell on success so the state badge recolours and any
// subsequent filter pass treats the row as the new state. Re-renders
// the grid after a short delay so rows that no longer match the
// current filter slip out naturally.
async function quickRelabel(cell, newLabel, btnEl) {
  const bench = cell.benchmark;
  if (!bench || bench.label === newLabel) return;
  const container = btnEl.closest(".ext-quick-relabel");
  const status = container.querySelector(".ext-quick-status");
  const pills = container.querySelectorAll(".ext-quick-btn");
  pills.forEach((p) => p.disabled = true);
  status.textContent = "Saving…";
  status.className = "ext-quick-status muted";
  try {
    const res = await api("/api/external-compare/benchmark/update", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        benchmark: state.view.benchmark,
        record_index: bench.record_index,
        source: cell.source,
        row: cell.row,
        column: cell.column,
        label: newLabel,
      }),
    });
    const rec = res.record;
    Object.assign(bench, {
      label: rec.label,
      corrected_value: rec.corrected_value,
      note: rec.note,
      error_types: rec.error_types || [],
      annotators: rec.annotators || [],
      _label_history: rec._label_history || [],
    });
    cell.state = recomputeState(cell, rec);
    // Reflect the new active pill + state badge immediately so the
    // change is visible even before the deferred re-render.
    pills.forEach((p) => p.classList.toggle("active", p.dataset.label === newLabel));
    const row = btnEl.closest("tr");
    const badgeCell = row && row.querySelector(".ext-state-badge");
    const info = STATE_INFO[cell.state];
    if (badgeCell && info) {
      badgeCell.className = `ext-state-badge ${info.cls}`;
      badgeCell.textContent = info.label;
    }
    status.textContent = "✓ saved";
    status.className = "ext-quick-status ext-quick-ok";
    // Defer the full re-render so the operator sees the green tick
    // before rows that fell out of the current filter disappear.
    setTimeout(() => renderCells(), 900);
    renderCanonBar();
  } catch (err) {
    status.textContent = `✗ ${err.message.slice(0, 80)}`;
    status.className = "ext-quick-status ext-quick-err";
    pills.forEach((p) => p.disabled = false);
  }
}

function renderPager(total) {
  const { offset, limit } = state.page;
  const start = total ? offset + 1 : 0;
  const end = Math.min(offset + limit, total);
  $("#ext-page-info").textContent = total ? `${start}–${end} of ${total}` : "0 of 0";
  $("#ext-prev").disabled = offset <= 0;
  $("#ext-next").disabled = end >= total;
}

function applyFilters() {
  state.filters = {
    state: $("#ext-filter-state").value,
    source: $("#ext-filter-source").value.trim(),
    search: $("#ext-filter-search").value.trim(),
  };
  state.page.offset = 0;
  renderCells();
}

function clearFilters() {
  state.filters = { state: "", source: "", search: "" };
  $("#ext-filter-state").value = "";
  $("#ext-filter-source").value = "";
  $("#ext-filter-search").value = "";
  state.page.offset = 0;
  renderCells();
}

// =====================================================================
// Drawer + inline edit form
// =====================================================================

async function openDrawer(cell) {
  $("#ext-drawer").hidden = false;
  $("#ext-drawer-title").textContent = `${cell.source} · row ${cell.row ?? "—"} · ${cell.column}`;
  $("#ext-drawer-body").innerHTML = `<div class="cmp-issue-empty">Loading…</div>`;
  let frame = null;
  const dataset = $("#ext-dataset").value;
  if (dataset && cell.row != null) {
    try {
      frame = await api(
        `/api/external-compare/source-frame?dataset=${encodeURIComponent(dataset)}` +
        `&source=${encodeURIComponent(cell.source)}` +
        `&center_row=${cell.row}&radius=3`
      );
    } catch (err) {
      frame = { _error: err.message };
    }
  }
  $("#ext-drawer-body").innerHTML = renderDrawer(cell, frame);
  wireEditForm(cell);
}

function renderDrawer(cell, frame) {
  const info = STATE_INFO[cell.state] || { label: cell.state, cls: "", desc: "" };
  const bench = cell.benchmark || null;
  const pred = cell.prediction || null;

  // CSV context (or a placeholder if dataset wasn't picked / file wasn't found).
  let ctxBlock = "";
  if (frame === null) {
    ctxBlock = `<div class="cmp-issue-empty">Pick a dataset in the top-of-panel selector to see CSV context here.</div>`;
  } else if (frame._error) {
    ctxBlock = `<div class="cmp-issue-empty">CSV context unavailable: ${escape(frame._error)}</div>`;
  } else {
    ctxBlock = `<div class="cmp-drawer-context">${renderFrameContext(frame, cell.row, cell.column)}</div>`;
  }

  // Prediction details. The external TSV has no ``error_type`` -- the
  // semantics are: presence means "model flagged this cell"; the
  // original→correction pair is the model's proposed fix.
  const predBlock = pred
    ? `<dl>
        ${pred.predicted_original_value != null ? `<dt>observed</dt><dd><code>${escape(pred.predicted_original_value)}</code></dd>` : ""}
        ${pred.predicted_correction != null ? `<dt>suggested correction</dt><dd><code>${escape(pred.predicted_correction)}</code></dd>` : ""}
        ${pred.match_method && pred.match_method !== "exact" ? `<dt>match method</dt><dd><code>${escape(pred.match_method)}</code> <small style="color:#888">(linked to bench cell as substring of full value)</small></dd>` : ""}
        ${pred.gt_status ? `<dt>gt_status</dt><dd><code>${escape(pred.gt_status)}</code></dd>` : ""}
        ${pred.precision_outcome ? `<dt>precision_outcome</dt><dd><code>${escape(pred.precision_outcome)}</code></dd>` : ""}
        ${pred.trivial_prediction ? `<dt>trivial?</dt><dd>yes</dd>` : ""}
        ${pred.notes ? `<dt>notes</dt><dd>${escape(pred.notes)}</dd>` : ""}
       </dl>`
    : `<div class="cmp-issue-empty">External run did not flag this cell.</div>`;

  // Unmatched predictions only: when the backend had access to the
  // underlying CSV it tells us whether the predicted value is actually
  // present in the file. ``in_csv`` => labeling candidate, ``not_in_csv``
  // => almost certainly a hallucination, ``unknown`` => operator didn't
  // pick a dataset.
  let csvHitBlock = "";
  if (cell.state === "unmatched_prediction" && cell.csv_match_status) {
    const status = cell.csv_match_status;
    const rows = Array.isArray(cell.csv_match_rows) ? cell.csv_match_rows : [];
    const method = cell.csv_match_method || "";
    const shown = rows.slice(0, 20).map((r) => escape(String(r))).join(", ");
    const more = rows.length > 20 ? ` (+${rows.length - 20} more)` : "";
    let badge;
    if (status === "in_csv") {
      badge = `<span class="ext-state-badge ext-state-unmatched">bench coverage gap</span>`;
    } else if (status === "not_in_csv") {
      badge = `<span class="ext-state-badge ext-state-unmatched">no CSV hit</span>`;
    } else {
      badge = `<span class="muted">unknown (no dataset selected)</span>`;
    }
    let rowsLine = "";
    if (status === "in_csv") {
      rowsLine = `<dt>CSV rows</dt><dd>${shown}${more}${method ? ` <small style="color:#888">(${escape(method)} match)</small>` : ""}</dd>`;
    } else if (status === "not_in_csv") {
      rowsLine = `<dt>scan</dt><dd>value not found anywhere in the column</dd>`;
    }
    csvHitBlock = `
      <div class="cmp-drawer-section">
        <h4>CSV verification</h4>
        <dl>
          <dt>status</dt><dd>${badge}</dd>
          ${rowsLine}
        </dl>
      </div>`;
  }

  // Benchmark details. ``cell.benchmark`` is always a single representative
  // record (the lowest-row record on conflicts). ``all_rows`` / ``all_labels``
  // surface any sibling rows so we render a compact multi-row banner.
  let benchBlock;
  if (!bench) {
    benchBlock = `<div class="cmp-issue-empty">Benchmark has no row for this cell.</div>`;
  } else {
    benchBlock = renderBenchRecord(bench, 0);
    if (Array.isArray(bench.all_rows) && bench.all_rows.length > 1) {
      const pairs = bench.all_rows.map((r, i) => `row ${escape(String(r))} → <span class="ext-lab ext-lab-${escape(bench.all_labels?.[i] || "")}">${escape(bench.all_labels?.[i] || "")}</span>`);
      benchBlock += `<div class="cmp-drawer-issue"><dl><dt>all sibling bench rows for this (source, column, normalized value)</dt><dd>${pairs.join(", ")}</dd></dl></div>`;
    }
  }

  // Edit form -- only enabled when we have a single editable record_index.
  const editable = bench && bench.record_index != null && bench.record_index >= 0;
  const editBlock = editable
    ? renderEditForm(bench)
    : `<div class="cmp-issue-empty">No editable benchmark row for this cell (e.g. unmatched prediction).</div>`;

  return `
    <div class="cmp-drawer-section">
      <div class="ext-drawer-state">
        <span class="ext-state-badge ${info.cls}">${escape(info.label)}</span>
        <span class="muted">${info.desc}</span>
      </div>
    </div>
    <div class="cmp-drawer-section">
      <h4>CSV context</h4>
      ${ctxBlock}
    </div>
    <div class="cmp-drawer-section">
      <h4>External prediction</h4>
      ${predBlock}
    </div>
    ${csvHitBlock}
    <div class="cmp-drawer-section">
      <h4>Benchmark record${bench && bench.all_rows && bench.all_rows.length > 1 ? "s" : ""}</h4>
      ${benchBlock}
    </div>
    <div class="cmp-drawer-section">
      <h4>Edit benchmark label</h4>
      ${editBlock}
    </div>
  `;
}

function renderBenchRecord(rec, idx) {
  const labCls = `ext-lab ext-lab-${escape(rec.label || "")}`;
  const history = (rec._label_history && rec._label_history.length)
    ? `<dt>label history</dt><dd>${rec._label_history.map((h) =>
        `<div class="muted" style="font-size:10.5px;">${escape(h.ts)} · ${escape(h.annotator)}: <code>${escape(h.previous_label || "")}</code>${h.previous_corrected_value ? ` → <code>${escape(h.previous_corrected_value)}</code>` : ""}</div>`).join("")}</dd>`
    : "";
  return `<div class="cmp-drawer-issue">
    <dl>
      <dt>label</dt><dd><span class="${labCls}">${escape(rec.label || "")}</span></dd>
      ${rec.corrected_value ? `<dt>corrected_value</dt><dd><code>${escape(rec.corrected_value)}</code></dd>` : ""}
      ${rec.error_types && rec.error_types.length ? `<dt>error_types</dt><dd>${rec.error_types.map((e) => `<code>${escape(e)}</code>`).join(", ")}</dd>` : ""}
      ${rec.note ? `<dt>note</dt><dd>${escape(rec.note)}</dd>` : ""}
      ${rec.annotators && rec.annotators.length ? `<dt>annotators</dt><dd>${rec.annotators.map((a) => escape(a)).join(", ")}</dd>` : ""}
      <dt>record_index</dt><dd><code>${escape(String(rec.record_index ?? "?"))}</code></dd>
      ${history}
    </dl>
  </div>`;
}

function renderEditForm(bench) {
  const label = bench.label || "";
  const corr = bench.corrected_value || "";
  const note = ""; // notes are appended (not replaced); start empty
  const errs = new Set(bench.error_types || []);
  const typeCheckboxes = ERROR_TYPE_VOCAB.map((t) =>
    `<label class="ext-chip"><input type="checkbox" name="ext-err-type" value="${escape(t)}"${errs.has(t) ? " checked" : ""}> ${escape(t)}</label>`).join("");
  return `<form class="ext-edit-form" id="ext-edit-form">
    <div class="ext-edit-row">
      <span class="ext-edit-key">Label</span>
      <div class="ext-edit-val">
        <label><input type="radio" name="ext-label" value="error"${label === "error" ? " checked" : ""}> error</label>
        <label><input type="radio" name="ext-label" value="clean"${label === "clean" ? " checked" : ""}> clean</label>
        <label><input type="radio" name="ext-label" value="ambiguous"${label === "ambiguous" ? " checked" : ""}> ambiguous</label>
      </div>
    </div>
    <div class="ext-edit-row">
      <span class="ext-edit-key">Corrected value</span>
      <div class="ext-edit-val">
        <input type="text" id="ext-edit-corrected" value="${escape(corr)}" placeholder="(canonical fix; only used when label = error)" />
      </div>
    </div>
    <div class="ext-edit-row">
      <span class="ext-edit-key">Error types</span>
      <div class="ext-edit-val ext-chips">${typeCheckboxes}</div>
    </div>
    <div class="ext-edit-row">
      <span class="ext-edit-key">Note (appended)</span>
      <div class="ext-edit-val">
        <textarea id="ext-edit-note" rows="2" placeholder="optional comment, will be appended to existing note">${escape(note)}</textarea>
      </div>
    </div>
    <div class="ext-edit-actions">
      <button type="submit" class="ext-edit-save">Save edit</button>
      <button type="button" class="ext-edit-cancel">Cancel</button>
      <span id="ext-edit-status" class="muted" style="font-size:11px;"></span>
    </div>
  </form>`;
}

function wireEditForm(cell) {
  const form = $("#ext-edit-form");
  if (!form) return;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    saveEdit(cell);
  });
  form.querySelector(".ext-edit-cancel").addEventListener("click", closeDrawer);
}

async function saveEdit(cell) {
  const bench = cell.benchmark || {};
  const form = $("#ext-edit-form");
  const status = $("#ext-edit-status");
  const newLabel = form.querySelector('input[name="ext-label"]:checked')?.value;
  const newCorrected = $("#ext-edit-corrected").value.trim();
  const newNote = $("#ext-edit-note").value.trim();
  const newErrorTypes = Array.from(form.querySelectorAll('input[name="ext-err-type"]:checked')).map((el) => el.value);
  // Only send fields that actually changed -- otherwise the server adds
  // a spurious history entry on every save.
  const body = {
    benchmark: state.view.benchmark,
    record_index: bench.record_index,
    source: cell.source,
    row: cell.row,
    column: cell.column,
  };
  let any = false;
  if (newLabel && newLabel !== bench.label) { body.label = newLabel; any = true; }
  const currentCorrected = bench.corrected_value || "";
  if (newCorrected !== currentCorrected) { body.corrected_value = newCorrected; any = true; }
  const currentTypes = (bench.error_types || []).slice().sort().join(",");
  const newTypesKey = newErrorTypes.slice().sort().join(",");
  if (currentTypes !== newTypesKey) { body.error_types = newErrorTypes; any = true; }
  if (newNote) { body.note = newNote; any = true; }
  if (!any) {
    status.textContent = "Nothing to save.";
    return;
  }
  status.textContent = "Saving…";
  try {
    const res = await api("/api/external-compare/benchmark/update", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    status.innerHTML = `<span style="color:#34d399;">Saved.</span> backup: <code>${escape(res.backup_path.split("/").pop())}</code>`;
    // Splice the updated record back into the in-memory view so the grid
    // can refresh without a full reload. Also update the cell's bench
    // snapshot in place so re-opening the drawer shows the new state.
    const rec = res.record;
    Object.assign(bench, {
      label: rec.label,
      corrected_value: rec.corrected_value,
      note: rec.note,
      error_types: rec.error_types || [],
      annotators: rec.annotators || [],
      _label_history: rec._label_history || [],
    });
    // Recompute the cell's state heuristically: relabelling clean<->error
    // changes FN/FP/TP/agreed; ambiguous always lands in ambiguous_in_bench.
    cell.state = recomputeState(cell, rec);
    renderCells();
    renderCanonBar();
  } catch (err) {
    status.innerHTML = `<span style="color:#f87171;">Error: ${escape(err.message)}</span>`;
  }
}

// After an edit, locally derive the new state so the row recolours without
// a server round-trip. The server is the source of truth on the next full
// reload; this is just a UX nicety.
function recomputeState(cell, rec) {
  const flagged = !!cell.prediction; // presence == flagged in this TSV format
  if (rec.label === "ambiguous") return "ambiguous_in_bench";
  if (rec.label === "error" && flagged) return "true_positive";
  if (rec.label === "error" && !flagged) return "false_negative";
  if (rec.label === "clean" && flagged) return "false_positive";
  if (rec.label === "clean" && !flagged) return "agreed_clean";
  return cell.state;
}

// =====================================================================
// Canonicalize-relabels action
// =====================================================================
//
// This client-side heuristic approximates the server's authoritative
// "needs a canonical corrected_value" predicate so the UI can show a live
// count without a request after every relabel. The server still decides which
// rows actually run.
const _INSTRUCTION_LEADING_VERBS = /^\s*(?:replace|change|correct|fix|review|verify|spell|consider|suggest|rename|update|remove|delete|add|rewrite|use|should|if|possibly)\b/i;
const _INSTRUCTION_PHRASES = /\bshould be\b|\bpossible correction\b|\bpossibly\b|\bintended (?:value|suffix|spelling)\b|\baffected rows?\b|\b(?:e\.g\.|i\.e\.)\b|\b->\b|\b=>\b|['"][^'"]{1,80}['"]\s+(?:to|with)\s+['"]/i;

function looksLikeInstruction(text) {
  if (!text) return false;
  const t = String(text).trim();
  if (!t) return false;
  return _INSTRUCTION_LEADING_VERBS.test(t) || _INSTRUCTION_PHRASES.test(t);
}

function bench_needs_canonicalization(bench) {
  if (!bench || bench.label !== "error") return false;
  const cv = (bench.corrected_value || "").trim();
  if (!cv) return true;
  return looksLikeInstruction(cv);
}

// Count cells whose bench record has BOTH non-empty _label_history
// (it's been edited via the UI) AND needs canonicalization.
function countRelabelCandidates() {
  if (!state.view) return 0;
  let n = 0;
  const seen = new Set(); // dedupe by record_index so collapsed cells count once
  for (const c of state.view.cells) {
    const b = c.benchmark;
    if (!b || !b._label_history || !b._label_history.length) continue;
    if (b.record_index == null || seen.has(b.record_index)) continue;
    if (!bench_needs_canonicalization(b)) continue;
    seen.add(b.record_index);
    n += 1;
  }
  return n;
}

function renderCanonBar() {
  if (!state.view) {
    $("#ext-canon-bar").hidden = true;
    return;
  }
  $("#ext-canon-bar").hidden = false;
  const n = countRelabelCandidates();
  const btn = $("#ext-canon-run");
  btn.disabled = n === 0 || state.canonRunning;
  $("#ext-canon-count").textContent = n
    ? `${n} relabeled row${n === 1 ? "" : "s"} still need${n === 1 ? "s" : ""} a canonical corrected value.`
    : "All relabeled rows have a canonical corrected value.";
}

async function runCanonicalize() {
  if (state.canonRunning) return;
  const n = countRelabelCandidates();
  if (!n) return;
  state.canonRunning = true;
  const btn = $("#ext-canon-run");
  const stat = $("#ext-canon-status");
  btn.disabled = true;
  stat.className = "ext-canon-status muted";
  stat.textContent = `Calling LLM on ${n} relabeled row${n === 1 ? "" : "s"}…`;
  try {
    const res = await api("/api/external-compare/benchmark/canonicalize", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        benchmark: state.view.benchmark,
        scope: "relabeled",
      }),
    });
    const parts = [
      `${res.updated} updated`,
      `${res.left_unchanged} ambiguous (left for manual review)`,
    ];
    if (res.errors) parts.push(`${res.errors} errors`);
    stat.innerHTML =
      `<span class="ext-canon-ok">✓</span> ${parts.join(" · ")}` +
      `<button type="button" id="ext-canon-detail" class="ext-secondary" style="margin-left: 10px;">Show details</button>`;
    stat.className = "ext-canon-status";
    // Reload the diff so the new corrected_values are visible in the grid.
    await loadDiff();
    // Wire the "show details" toggle after reload (which clears innerHTML).
    // The detail panel is built lazily so the report sits in state.
    state.lastCanonReport = res.report || [];
    state.lastCanonModel = res.model || "";
    const detailBtn = document.getElementById("ext-canon-detail");
    if (detailBtn) detailBtn.addEventListener("click", showCanonDetail);
  } catch (err) {
    stat.innerHTML = `<span class="ext-canon-err">✗ ${escape(err.message)}</span>`;
    stat.className = "ext-canon-status";
  } finally {
    state.canonRunning = false;
    renderCanonBar();
  }
}

function showCanonDetail() {
  const report = state.lastCanonReport || [];
  const model = state.lastCanonModel || "(unknown)";
  // Reuse the right-side drawer for the report. Show one table per status
  // bucket so the operator can quickly scan ambiguous cases that still
  // need manual attention.
  $("#ext-drawer").hidden = false;
  $("#ext-drawer-title").textContent = `Canonicalize report (model: ${model})`;
  const byStatus = { updated: [], left_unchanged: [], error: [] };
  for (const e of report) (byStatus[e.status] || (byStatus[e.status] = [])).push(e);
  const section = (title, list, statusClass) => {
    if (!list.length) return "";
    const rows = list.map((e) => `<tr>
      <td><code>${escape(e.source)}</code></td>
      <td class="num">${escape(String(e.row ?? ""))}</td>
      <td><code>${escape(e.column ?? "")}</code></td>
      <td><code>${escape(String(e.original_value ?? "")).slice(0, 80)}</code></td>
      <td>${e.previous_corrected_value
            ? `<code>${escape(String(e.previous_corrected_value)).slice(0, 80)}</code>`
            : `<span class="cmp-issue-empty">(null)</span>`}</td>
      <td>${e.new_corrected_value
            ? `<code>${escape(e.new_corrected_value)}</code>`
            : `<span class="cmp-issue-empty">${e.error ? escape(e.error).slice(0, 80) : "—"}</span>`}</td>
    </tr>`).join("");
    return `<div class="cmp-drawer-section">
      <h4>${escape(title)} <span class="muted" style="font-weight: normal; font-size: 11px;">(${list.length})</span></h4>
      <table class="compare-table">
        <thead><tr><th>Source</th><th class="num">Row</th><th>Column</th><th>Original</th><th>Previous</th><th>New / reason</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  };
  $("#ext-drawer-body").innerHTML =
    section("Updated", byStatus.updated || []) +
    section("Left unchanged (ambiguous, manual review needed)", byStatus.left_unchanged || []) +
    section("Errors", byStatus.error || []);
}

// =====================================================================
// CSV context (mirrors compare.js renderFrameContext)
// =====================================================================

function renderFrameContext(frame, targetRow, targetCol) {
  if (!frame.rows?.length) return `<span class="cmp-issue-empty">(no rows)</span>`;
  const cols = frame.columns;
  const colWidth = 18;
  const fmt = (v) => {
    const s = String(v ?? "");
    return s.length > colWidth ? s.slice(0, colWidth - 1) + "…" : s.padEnd(colWidth, " ");
  };
  const wholeRow = targetCol === "(whole row)";
  const header = ["row".padEnd(5), ...cols.map(fmt)].join(" │ ");
  const lines = [escape(header), escape("─".repeat(header.length))];
  for (const r of frame.rows) {
    const cells = cols.map((c) => {
      const v = r.values[c];
      const cell = fmt(v);
      if (r.row === targetRow && (wholeRow || c === targetCol)) {
        return `<span class="cell-target">${escape(cell)}</span>`;
      }
      return escape(cell);
    });
    lines.push([escape(String(r.row).padEnd(5)), ...cells].join(" │ "));
  }
  return lines.join("\n");
}

function closeDrawer() { $("#ext-drawer").hidden = true; }

// =====================================================================
// Wiring
// =====================================================================

function init() {
  // Sub-mode toggle (lives at the top of the Compare panel).
  $$(".cmp-mode").forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.cmpMode));
  });
  $("#ext-benchmark").addEventListener("change", updateButtonState);
  $("#ext-predictions").addEventListener("change", updateButtonState);
  $("#ext-load").addEventListener("click", loadDiff);
  $("#ext-canon-run").addEventListener("click", runCanonicalize);
  $("#ext-apply-filters").addEventListener("click", applyFilters);
  $("#ext-clear-filters").addEventListener("click", clearFilters);
  $("#ext-prev").addEventListener("click", () => {
    state.page.offset = Math.max(0, state.page.offset - state.page.limit);
    renderCells();
  });
  $("#ext-next").addEventListener("click", () => {
    state.page.offset += state.page.limit;
    renderCells();
  });
  $("#ext-drawer-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#ext-drawer").hidden) closeDrawer();
  });
  // Enter on either select also enables Compare, so a quick keyboard
  // pick-and-go works.
  $("#ext-include-trivial").addEventListener("change", () => {
    if (!$("#ext-load").disabled) loadDiff();
  });
}

init();
