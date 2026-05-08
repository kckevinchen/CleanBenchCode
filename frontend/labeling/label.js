// Labelling UI - vanilla ES module, no build step.
//
// Reads ?run=<run_id> from the URL (set by the runs dashboard) and
// talks to /api/runs/<run_id>/... for everything.

const params = new URLSearchParams(location.search);
const RUN_ID = params.get("run");
if (!RUN_ID) {
  document.body.innerHTML = `<pre style="padding:20px;color:#f87171;">Missing ?run=&lt;id&gt; query param. <a href="./">Back to runs</a>.</pre>`;
  throw new Error("missing run id");
}
const API = `/api/runs/${encodeURIComponent(RUN_ID)}`;

const state = {
  run: null,
  // Filters are arrays so the operator can multi-select. ``source``
  // entries are either a CSV filename or a synthetic ``workbook:<wb>``
  // token (matches every CSV in that workbook). ``column_status`` is
  // a list of human-friendly arm-status labels (see ARM_STATUS_LABEL)
  // plus the synthetic ``no-estimate`` value for cells the labelling-
  // effort artifact doesn't cover. Empty array == no filter on that
  // facet (i.e. "any").
  filters: { source: [], status: [], error_type: [], column_status: [] },
  queue: [],
  index: -1,
  selectedId: null,
  cluster: null,
  latest: {},
  csvCache: new Map(),
  // Per-source cell-state map: source -> Map("<row>\0<col>" -> entry).
  // Refreshed lazily after each label submission.
  cellState: new Map(),
  // Currently rendered CSV source (for repaint without round-trip).
  currentSource: null,
  currentTargetRow: null,
  currentColumns: new Set(),
  // Per-arm Good-Turing labelling-effort projection (from
  // /api/runs/<id>/labeling-effort). Cached at bootstrap; consulted
  // both by the effort panel and by renderCsv to surface per-(column,
  // error_type) "+N missed" chips next to LLM-annotated cells.
  labelingEffort: null,
  // Pre-built `${source}\x00${column}` -> {cell, byEt} index over the
  // labelling-effort artifact, so renderCsv can do an O(1) lookup
  // per cell instead of a linear scan over ~1000 entries per render.
  labelingEffortIndex: null,
  // Per-source exclusion record: source -> {csv: bool, rows: Set<int>,
  // cols: Set<string>, ids: Map<key,id>}. ``ids`` maps a chip key to
  // the server-side exclusion id so the chip's undo button can issue
  // a DELETE without an extra lookup. Refreshed after every add/undo.
  exclusions: new Map(),
  // Column-wide web-verification ("column audit") state. Populated
  // lazily by the per-row "Verify column" button in the effort table
  // and by an initial bootstrap fetch of the audit-index manifest.
  //   columnAuditIndex: Map<"src\x00col", {counts, ts, path}>
  //     Manifest of audited (source, column) pairs.
  //   columnAuditDetail: Map<"src\x00col", {counts, items, source, column}>
  //     Lazy-loaded full record list per pair; fetched on first
  //     expand or after a job finishes.
  //   columnAuditJobs: Map<"src\x00col", {jobId, status, startedAt}>
  //     In-flight audit jobs keyed by the pair they're verifying.
  columnAuditIndex: new Map(),
  columnAuditDetail: new Map(),
  columnAuditJobs: new Map(),
  // Per-(source, column, row) audit-verdict lookup, derived from
  // ``columnAuditDetail`` whenever detail arrives. Each entry carries
  // a *reference* to the underlying audit item so the cell painter
  // and the popover both see the same canonical/evidence payload
  // without re-walking ``items``.
  //   columnAuditCellLookup: Map<"src\x00col\x00row", auditItem>
  columnAuditCellLookup: new Map(),
};

const CSV_WINDOW = 80; // rows above/below the target row

const $ = (sel) => document.querySelector(sel);

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function toast(msg, ms = 1500) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

// Multi-table integration: cross-sheet findings (a.k.a. "MT" items)
// share the queue with classic clusters. The two backends use
// different verdict vocabularies, so we map between them at the UI
// boundary - the user only ever sees / presses the cluster vocab
// (tp / fp / partial / unsure / skip), and we translate to the MT
// vocab on POST + back on GET. ``skip`` has no MT counterpart so the
// skip button is hidden when an MT item is selected.
const MT_TO_CLUSTER_STATUS = {
  confirmed_inconsistency: "tp",
  benign_variation: "partial",
  not_a_violation: "fp",
  unsure: "unsure",
};
const CLUSTER_TO_MT_STATUS = {
  tp: "confirmed_inconsistency",
  partial: "benign_variation",
  fp: "not_a_violation",
  unsure: "unsure",
  // ``skip`` intentionally omitted: not a valid MT verdict.
};
function isMtItem(item) { return !!(item && item.is_mt); }

// ----------------------------------------------------------- multi-select filter widget
//
// Custom checkbox-dropdown that replaces the native <select> filters
// in the sidebar so the operator can pick multiple values per facet
// (e.g. show both ``unlabeled`` and ``unsure`` rows at once). Uses
// vanilla DOM only - no third-party deps.
//
// Each widget takes:
//   * root: the .multi-filter container (already in the DOM)
//   * groups: [{ label?, options: [{ value, label, title? }, ...] }, ...]
//             - label-less group is rendered inline; labelled groups
//               are rendered with a small uppercase header
//   * onChange: called with the new array of selected values whenever
//               the user toggles a checkbox

const _multiFilters = new Map(); // root -> controller

function buildMultiFilter(root, groups, onChange) {
  // Reset & set up DOM. ``data-empty-label`` overrides the "(any)"
  // text shown when no checkbox is selected.
  const emptyLabel = root.dataset.emptyLabel || "(any)";
  root.innerHTML = `
    <button type="button" class="multi-filter-btn">
      <span class="multi-filter-text">${emptyLabel}</span>
      <span class="caret">▾</span>
    </button>
    <div class="multi-filter-pop" hidden>
      <div class="multi-filter-actions">
        <button type="button" data-action="all">all</button>
        <button type="button" data-action="none">clear</button>
      </div>
      <div class="multi-filter-options"></div>
    </div>`;
  const btn = root.querySelector(".multi-filter-btn");
  const txt = root.querySelector(".multi-filter-text");
  const pop = root.querySelector(".multi-filter-pop");
  const optBox = root.querySelector(".multi-filter-options");

  // Render each option as a clickable pill. The pill toggles a
  // ``.selected`` class on click; ``data-value`` is the canonical
  // value used by getSelected / setSelected.
  for (const g of groups) {
    if (g.label) {
      const h = document.createElement("div");
      h.className = "multi-filter-group";
      h.textContent = g.label;
      optBox.appendChild(h);
    }
    for (const o of g.options) {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "multi-filter-opt";
      pill.dataset.value = o.value;
      if (o.title) pill.title = o.title;
      const span = document.createElement("span");
      span.className = "opt-label";
      span.textContent = o.label;
      pill.appendChild(span);
      optBox.appendChild(pill);
    }
  }

  const allPills = () => Array.from(optBox.querySelectorAll(".multi-filter-opt"));
  const getSelected = () =>
    allPills().filter((p) => p.classList.contains("selected")).map((p) => p.dataset.value);
  const renderBtnText = () => {
    const sel = getSelected();
    if (sel.length === 0) txt.textContent = emptyLabel;
    else if (sel.length === 1) {
      const pill = allPills().find((p) => p.dataset.value === sel[0]);
      txt.textContent = pill ? pill.querySelector(".opt-label").textContent : sel[0];
    } else txt.textContent = `${sel.length} selected`;
  };

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = pop.hidden;
    // Close any other open popovers first so only one is open at a
    // time (avoids overlap when filters are cramped).
    for (const ctrl of _multiFilters.values()) ctrl.close();
    pop.hidden = !willOpen;
  });

  optBox.addEventListener("click", (e) => {
    const pill = e.target.closest(".multi-filter-opt");
    if (!pill || !optBox.contains(pill)) return;
    e.stopPropagation();
    pill.classList.toggle("selected");
    renderBtnText();
    onChange(getSelected());
  });

  pop.querySelector('[data-action="all"]').addEventListener("click", (e) => {
    e.stopPropagation();
    for (const p of allPills()) p.classList.add("selected");
    renderBtnText();
    onChange(getSelected());
  });
  pop.querySelector('[data-action="none"]').addEventListener("click", (e) => {
    e.stopPropagation();
    for (const p of allPills()) p.classList.remove("selected");
    renderBtnText();
    onChange(getSelected());
  });

  // Stop bare clicks inside the popover (e.g. on padding) from
  // bubbling up and triggering the outside-click close handler.
  pop.addEventListener("click", (e) => e.stopPropagation());

  const ctrl = {
    setSelected(values) {
      const wanted = new Set(values || []);
      for (const p of allPills()) {
        p.classList.toggle("selected", wanted.has(p.dataset.value));
      }
      renderBtnText();
    },
    getSelected,
    close() { pop.hidden = true; },
    refreshLabels(byValue) {
      // Update displayed text for an option without rebuilding the
      // widget. Used when source coverage counts change after a label
      // save (mirrors the old refreshSourceMeta behaviour).
      for (const p of allPills()) {
        const meta = byValue.get(p.dataset.value);
        if (!meta) continue;
        const span = p.querySelector(".opt-label");
        if (span && meta.label != null) span.textContent = meta.label;
        if (meta.title != null) p.title = meta.title;
      }
      renderBtnText();
    },
  };
  _multiFilters.set(root, ctrl);
  return ctrl;
}

// Single document-level handler closes any open multi-filter popover
// when the user clicks outside it. Registered once, idempotent.
if (!window.__multiFilterOutsideHandler) {
  document.addEventListener("click", () => {
    for (const ctrl of _multiFilters.values()) ctrl.close();
  });
  window.__multiFilterOutsideHandler = true;
}

// ----------------------------------------------------------- bootstrap

async function bootstrap() {
  const run = await api(`${API}`);
  state.run = run;

  document.getElementById("export-link").href = `${API}/export/benchmark`;
  // The per-type bundle anchor is optional - older cached copies of
  // label.html don't have it. Avoid blowing up bootstrap when the JS
  // is newer than the HTML the browser cached.
  const byType = document.getElementById("export-by-type-link");
  if (byType) byType.href = `${API}/export/benchmark-by-type`;
  // Same defensive guard for the sources-bundle anchor.
  const exportSources = document.getElementById("export-sources-link");
  if (exportSources) exportSources.href = `${API}/export/sources`;

  // Single-error-type export: pick a type from the dropdown, hit the
  // /export/benchmark/<slug> endpoint. The dropdown is populated below
  // alongside the queue's error-type filter so the available choices
  // reflect what was actually clustered/observed in this run.
  const singleSel = document.getElementById("export-single-type");
  const singleLink = document.getElementById("export-single-link");
  if (singleSel && singleLink) {
    const refresh = () => {
      const v = singleSel.value || "";
      if (!v) {
        singleLink.href = "#";
        singleLink.classList.add("muted");
        singleLink.setAttribute("aria-disabled", "true");
        singleLink.removeAttribute("download");
      } else {
        singleLink.href = `${API}/export/benchmark/${encodeURIComponent(v)}`;
        singleLink.classList.remove("muted");
        singleLink.removeAttribute("aria-disabled");
        // Hint the browser at a per-type filename. The server also
        // sets Content-Disposition with a richer name, which wins.
        singleLink.setAttribute("download", `benchmark_${v}.jsonl`);
      }
    };
    singleSel.addEventListener("change", refresh);
    refresh();
  }

  $("#run-meta").textContent =
    `${RUN_ID} · annotator ${run.annotator} · ` +
    `${run.counts.clusters} clusters · ${run.counts.issues} raw issues` +
    (run.counts.multi_table_findings
      ? ` · ${run.counts.multi_table_findings} cross-sheet`
      : "");

  // Populate source filter. For multi-workbook runs (e.g. curated_pbi /
  // Bing21 reconstructed) we group sources under labelled headers by
  // their parent-dir basename, and prepend each group with an "(all
  // in <wb>)" entry whose value is the synthetic ``workbook:<wb>``
  // token. Filtering uses ANY semantics: a cluster matches if its
  // source is checked OR any checked workbook token covers it.
  const multiWb = !!run.multi_workbook;
  // Short, human-friendly label for a source filename. Detects the
  // Bing21 convention "<wb_sha>.xlsx_<sheet_sha>.csv" - those names
  // are unreadable at full length but the leading bytes of each sha
  // are enough to disambiguate within a corpus. Falls back to the
  // original name for anything that doesn't match the pattern.
  const shortSource = (src) => {
    if (!src) return "";
    const m = src.match(/^([0-9a-f]{8})[0-9a-f]*\.xlsx_([0-9a-f]{6})[0-9a-f]*\.csv$/i);
    return m ? `${m[1]}.xlsx / ${m[2]}` : src;
  };
  // ``optionFor`` formats a source row with a checkmark prefix so the
  // operator can see at a glance which CSVs they have already touched.
  // ``\u2713`` = touched (cluster verdict OR FN cell exists),
  // ``\u25CB`` = untouched. ``(N)`` shows cluster count; ``\u2205``
  // flags CSVs the LLM never flagged (zero clusters) so the operator
  // knows they need to browse for FN candidates.
  const sourceOption = (s) => {
    const mark = s.checked ? "\u2713" : "\u25CB";
    const cnt = s.clusters > 0 ? `(${s.clusters})` : "(\u2205)";
    const fn = s.fn_cells ? ` \u00B7 fn:${s.fn_cells}` : "";
    return {
      value: s.name,
      label: `${mark} ${shortSource(s.name)} ${cnt}${fn}`,
      title: s.name,
    };
  };
  const sourceGroups = [];
  if (multiWb && Array.isArray(run.workbooks) && run.workbooks.length) {
    const grouped = new Set();
    for (const wb of run.workbooks) {
      const wbChecked = wb.sources.filter(
        (n) => run.sources.find((x) => x.name === n)?.checked
      ).length;
      const opts = [{
        value: `workbook:${wb.name}`,
        label: `(all in ${wb.name})`,
        title: `Match every CSV under workbook ${wb.name}`,
      }];
      for (const name of wb.sources) {
        const s = run.sources.find((x) => x.name === name);
        if (!s) continue;
        opts.push(sourceOption(s));
        grouped.add(name);
      }
      sourceGroups.push({
        label: `${wb.name}  \u2014  ${wbChecked}/${wb.sources.length} checked, ${wb.clusters} clusters`,
        options: opts,
      });
    }
    const orphans = run.sources.filter((s) => !grouped.has(s.name));
    if (orphans.length) {
      sourceGroups.push({ label: "(other)", options: orphans.map(sourceOption) });
    }
  } else {
    sourceGroups.push({ options: run.sources.map(sourceOption) });
  }

  buildMultiFilter($("#filter-source"), sourceGroups, (selected) => {
    state.filters.source = selected;
    updateSourceXlsxActions();
    renderLabelingEffort();
    renderColumnAuditPanel();
    refreshQueue({ keepSelection: false });
  });

  // Status filter: fixed vocabulary, identical for every run.
  buildMultiFilter($("#filter-status"), [{
    options: [
      { value: "unlabeled", label: "unlabeled" },
      { value: "tp", label: "tp" },
      { value: "fp", label: "fp" },
      { value: "partial", label: "partial" },
      { value: "unsure", label: "unsure" },
      { value: "skip", label: "skip" },
    ],
  }], (selected) => {
    state.filters.status = selected;
    refreshQueue({ keepSelection: false });
  });

  // Populate error-type filter from summary.
  const ets = Object.keys(run.summary?.issues_by_error_type || {});
  // Multi-table findings carry a synthetic ``inter_table_inconsistency``
  // error_type so the operator can isolate them via the existing
  // filter dropdown without a separate UI control.
  if (run.counts?.multi_table_findings && !ets.includes("inter_table_inconsistency")) {
    ets.push("inter_table_inconsistency");
  }
  ets.sort();
  buildMultiFilter($("#filter-error-type"), [{
    options: ets.map((et) => ({ value: et, label: et })),
  }], (selected) => {
    state.filters.error_type = selected;
    refreshQueue({ keepSelection: false });
  });

  // Column-status filter: narrows the queue by the adaptive-loop arm
  // state of the cluster's (source, column) cell. Backed by the per-
  // arm Good-Turing artifact loaded async by refreshLabelingEffort();
  // the wrapper #filter-column-status-cell is unhidden there if the
  // run has the artifact available. Options stay constant per run so
  // we can build the widget here and just toggle visibility later.
  buildMultiFilter($("#filter-column-status"), [{
    options: COLUMN_STATUS_OPTIONS,
  }], (selected) => {
    state.filters.column_status = selected;
    refreshQueue({ keepSelection: false });
  });
  // Color-code the option pills so "budget-capped" stands out from
  // the muted "converged" / "too sparse" pills - operators are
  // typically hunting for capped columns, and the visual cue makes
  // the multi-select scan-friendly. Whitespace in a column-status
  // value (e.g. "too sparse") is illegal in a CSS class token, so
  // we slugify with hyphens before forming the class name.
  for (const opt of $("#filter-column-status").querySelectorAll(".multi-filter-opt")) {
    const slug = String(opt.dataset.value || "").replace(/\s+/g, "-");
    if (slug) opt.classList.add(`col-status-${slug}`);
  }

  // Mirror the same list into the per-type export dropdown so an
  // operator can download just one error_type's benchmark JSONL.
  // We DON'T include the synthetic ``_uncategorized`` bucket here -
  // it's reachable directly via the /export/benchmark/_uncategorized
  // URL but is rarely useful as a deliberate export target, and
  // surfacing it next to the labelled types invites confusion.
  const exportSel = document.getElementById("export-single-type");
  if (exportSel) {
    ets.sort().forEach((et) => {
      const o = document.createElement("option");
      o.value = et; o.textContent = et;
      exportSel.appendChild(o);
    });
    exportSel.dispatchEvent(new Event("change"));
  }

  // Filter change handlers are wired up in buildMultiFilter() above.
  updateSourceXlsxActions();

  setupExclusionBar();
  await Promise.all([
    refreshLatest(),
    refreshStats(),
    refreshLabelingEffort(),
    refreshExclusions(),
    refreshColumnAuditIndex(),
  ]);
  await refreshQueue({ keepSelection: false });

  setupColumnAuditBulkButton();
  setupAuditProgressCancel();
  // Reconstruct in-flight column-audit job state from the launcher
  // BEFORE kicking off the progress poller so the per-row controls
  // know whether to show the progress bar on first paint.
  await recoverInFlightAuditJobs();
  // The effort table was built BEFORE recovery populated state, so
  // its rows are still showing the idle (button) view. Force a
  // refresh so any row whose audit job was recovered flips to the
  // running view immediately. Likewise lock the bulk button if a
  // bulk job was recovered.
  refreshAllColumnAuditControlStates();
  applyBulkAuditButtonState();
  // Kick off the progress poller in case a column-audit job is
  // already running on the server (e.g. operator reloaded the page).
  ensureAuditProgressPoller();

  window.addEventListener("keydown", onKeydown);
}

async function refreshLatest() {
  const r = await api(`${API}/labels`);
  state.latest = r.latest || {};
}

async function refreshStats() {
  const s = await api(`${API}/stats`);
  const o = s.overall;
  const fn = s.fn || { total: 0, shadowed: 0 };
  const box = $("#progress");
  box.innerHTML = "";
  const order = [
    ["labelled", o.total - o.unlabeled],
    ["tp", o.tp], ["fp", o.fp], ["partial", o.partial],
    ["unsure", o.unsure], ["skip", o.skip], ["unlabeled", o.unlabeled],
  ];
  for (const [k, v] of order) {
    const pct = o.total ? Math.round((v / o.total) * 100) : 0;
    const el = document.createElement("span");
    el.className = "pill";
    el.innerHTML = `<strong>${v}</strong>${k} <span class="muted">(${pct}%)</span>`;
    box.appendChild(el);
  }
  // FN pill (cell-level). The shadowed count surfaces double-labelling
  // (FN cell + TP cluster covering it) - it should be 0 in healthy
  // runs because TP via cluster supersedes any FN claim.
  if (fn.total || fn.shadowed) {
    const fnEl = document.createElement("span");
    fnEl.className = "pill";
    const sh = fn.shadowed
      ? ` <span class="muted" title="FN cells already covered by a TP cluster (auto-superseded)">(\u26A0 ${fn.shadowed} shadowed)</span>`
      : "";
    fnEl.innerHTML = `<strong>${fn.total}</strong>fn (cells)${sh}`;
    box.appendChild(fnEl);
  }
}

// Per-(source, column) labelling-effort projection driven by the
// per-arm Good-Turing artifact. Hidden for runs that don't expose it
// (single-pass / fixed-K / non-column stop_scope). The source filter
// is reused to filter the table's visible rows, so the operator can
// jump from a high-missed cell into its source by clicking a row.
async function refreshLabelingEffort() {
  let payload;
  try {
    payload = await api(`${API}/labeling-effort`);
  } catch (err) {
    console.warn("labeling-effort unavailable:", err);
    return;
  }
  const panel = document.getElementById("effort-panel");
  if (!panel) return;
  // The column-status sidebar filter is meaningless without per-arm
  // estimates - hide it when the run doesn't expose them so the
  // sidebar doesn't dangle a non-functional control.
  const colStatusCell = document.getElementById("filter-column-status-cell");
  if (!payload || !payload.available) {
    panel.hidden = true;
    if (colStatusCell) colStatusCell.hidden = true;
    // Drop any stale column-status selection so a previously-hidden
    // filter doesn't silently filter out the queue.
    if ((state.filters.column_status || []).length) {
      state.filters.column_status = [];
      const ctrl = _multiFilters.get(document.getElementById("filter-column-status"));
      if (ctrl) ctrl.setSelected([]);
    }
    return;
  }
  panel.hidden = false;
  if (colStatusCell) colStatusCell.hidden = false;
  state.labelingEffort = payload;
  // Build a (source\x00column) -> {cell, byEt:Map} index ONCE here,
  // so the per-cell chip lookup in renderCsv (~CSV_WINDOW * num_cols
  // calls per render) is O(1) instead of a linear scan over ~1000
  // entries each time. Rebuilt whenever this endpoint refreshes.
  const idx = new Map();
  for (const c of (payload.cells || [])) {
    const byEt = new Map();
    for (const e of (c.by_error_type || [])) byEt.set(e.error_type, e);
    idx.set(`${c.source}\x00${c.column}`, { cell: c, byEt });
  }
  state.labelingEffortIndex = idx;

  const totals = payload.totals || {};
  document.getElementById("effort-headline").textContent =
    `${(totals.cells || 0)} cells \u00B7 ` +
    `seen ${Math.round(totals.n_observed || 0)} \u00B7 ` +
    `Chao1 total \u2248 ${(totals.chao1_estimated_total || 0).toFixed(1)} \u00B7 ` +
    `missed \u2248 ${(totals.chao1_estimated_missed || 0).toFixed(1)} \u00B7 ` +
    `labelled ${totals.labelled_so_far || 0}` +
    (payload.num_runs_completed
      ? `  (after ${payload.num_runs_completed} pass${payload.num_runs_completed === 1 ? "" : "es"})`
      : "");

  // Surface the "Audit all budget-capped" bulk button when at least
  // one cell has an exhausted arm. Cheap O(N) scan; the artifact
  // has at most a few thousand cells.
  const bulkBtn = document.getElementById("audit-all-capped-btn");
  if (bulkBtn) {
    let cappedCount = 0;
    for (const c of (payload.cells || [])) {
      if ((c.by_error_type || []).some((e) => e.status === "exhausted")) {
        cappedCount += 1;
      }
    }
    if (cappedCount > 0) {
      bulkBtn.hidden = false;
      // Stash the idle label so applyBulkAuditButtonState can restore
      // it after the job finishes (the running state replaces the
      // text with "Auditing…").
      bulkBtn.dataset.idleLabel = `Audit all ${cappedCount} budget-capped column${cappedCount === 1 ? "" : "s"}`;
      applyBulkAuditButtonState();
    } else {
      bulkBtn.hidden = true;
    }
  }

  renderLabelingEffort();
}

function applyBulkAuditButtonState() {
  /* Single source of truth for the "Audit all budget-capped" button's
   * enabled / label state. Called whenever ``state._bulkAuditJob``
   * changes (launch / poll terminal / recovery) and from
   * refreshLabelingEffort after the visibility/text is recomputed. */
  const btn = document.getElementById("audit-all-capped-btn");
  if (!btn) return;
  const job = state._bulkAuditJob;
  const running = job
    && (job.status === "running" || job.status === "queued");
  if (running) {
    btn.disabled = true;
    btn.textContent = "Auditing\u2026";
    btn.title = `Bulk audit job ${job.jobId} is in flight — use the ✕ on the progress bar (or a row's progress bar) to cancel.`;
  } else {
    btn.disabled = false;
    const idle = btn.dataset.idleLabel;
    if (idle) btn.textContent = idle;
    btn.title = "Run the column-wide web verifier on every column with at least one budget-capped (status=exhausted) error-type arm.";
  }
}

// ----------------------------------------------------------- exclusions
//
// Operator-driven decisions to skip whole CSVs / specific rows /
// specific columns during labelling. The records persist server-side
// in ``<run_dir>/exclusions.jsonl`` (see ExclusionStore) so a follow-
// up cleaning step can rebuild a sanitised dataset; here we only
// surface a strikethrough mark so the operator can still see the
// content (and undo via the chip's ✕ button if they change their
// mind). Exclusions are independent of TP/FP/PARTIAL/UNSURE/SKIP.

async function refreshExclusions() {
  let payload;
  try {
    payload = await api(`${API}/exclusions`);
  } catch (err) {
    console.warn("exclusions unavailable:", err);
    state.exclusions = new Map();
    return;
  }
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const bySrc = new Map();
  for (const e of items) {
    const src = e.source || "";
    if (!bySrc.has(src)) {
      bySrc.set(src, { csv: null, rows: new Map(), cols: new Map() });
    }
    const bucket = bySrc.get(src);
    if (e.scope === "csv") {
      bucket.csv = e.id;
    } else if (e.scope === "rows") {
      for (const r of (e.rows || [])) bucket.rows.set(Number(r), e.id);
    } else if (e.scope === "columns") {
      for (const c of (e.columns || [])) bucket.cols.set(String(c), e.id);
    }
  }
  state.exclusions = bySrc;
  updateExclusionUi();
  applyExclusionMarks();
  // Repaint the queue so cluster-excluded rows pick up the strikethrough
  // class (or shed it on undo). Cheap: no network round-trip needed.
  paintQueueExclusions();
}

// Returns the per-source bucket built by refreshExclusions(), or an
// empty placeholder so callers don't have to null-check.
function exclusionsForSource(source) {
  return state.exclusions.get(source) || { csv: null, rows: new Map(), cols: new Map() };
}

// Parse a comma + range expression like ``"5, 8-12, 20"`` into a
// flat list of integers. Throws on garbage so the bar can surface a
// clear toast instead of POSTing nonsense.
function parseRowSpec(text) {
  const out = new Set();
  const trimmed = String(text || "").trim();
  if (!trimmed) return [];
  for (const partRaw of trimmed.split(",")) {
    const part = partRaw.trim();
    if (!part) continue;
    const m = part.match(/^(-?\d+)\s*-\s*(-?\d+)$/);
    if (m) {
      const a = Number(m[1]); const b = Number(m[2]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        throw new Error(`bad row range: "${part}"`);
      }
      const lo = Math.min(a, b); const hi = Math.max(a, b);
      for (let i = lo; i <= hi; i += 1) out.add(i);
    } else {
      const n = Number(part);
      if (!Number.isFinite(n)) throw new Error(`bad row index: "${part}"`);
      out.add(n);
    }
  }
  return [...out].sort((a, b) => a - b);
}

// Parse "a, b, c" into a list of column names; trim whitespace and
// drop empties. Quoted column names with commas aren't supported (no
// known dataset uses them); pass them one at a time if needed.
function parseColumnSpec(text) {
  return String(text || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function setupExclusionBar() {
  const bar = document.getElementById("exclude-bar");
  if (!bar) return;

  document.getElementById("exclude-csv-btn").addEventListener("click", async () => {
    const src = state.currentSource;
    if (!src) return;
    const bucket = exclusionsForSource(src);
    if (bucket.csv) { toast("CSV already excluded"); return; }
    if (!confirm(`Exclude entire CSV "${src}" from labelling?`)) return;
    await postExclusion({ scope: "csv", source: src });
  });

  document.getElementById("exclude-rows-btn").addEventListener("click", async () => {
    const src = state.currentSource;
    if (!src) return;
    const input = document.getElementById("exclude-rows-input");
    let rows;
    try { rows = parseRowSpec(input.value); }
    catch (e) { toast(e.message, 2500); return; }
    if (!rows.length) { toast("enter at least one row index"); return; }
    await postExclusion({ scope: "rows", source: src, rows });
    input.value = "";
  });

  document.getElementById("exclude-cols-btn").addEventListener("click", async () => {
    const src = state.currentSource;
    if (!src) return;
    const input = document.getElementById("exclude-cols-input");
    const columns = parseColumnSpec(input.value);
    if (!columns.length) { toast("enter at least one column name"); return; }
    await postExclusion({ scope: "columns", source: src, columns });
    input.value = "";
  });

  // Submit on Enter for both inputs.
  document.getElementById("exclude-rows-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("exclude-rows-btn").click();
  });
  document.getElementById("exclude-cols-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("exclude-cols-btn").click();
  });
}

async function postExclusion(body) {
  try {
    await api(`${API}/exclusions`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    await refreshExclusions();
    toast("excluded");
  } catch (e) {
    toast(`exclude failed: ${e.message}`, 3000);
  }
}

async function deleteExclusion(id) {
  if (!id) return;
  try {
    await api(`${API}/exclusions/${encodeURIComponent(id)}`, { method: "DELETE" });
    await refreshExclusions();
  } catch (e) {
    toast(`undo failed: ${e.message}`, 3000);
  }
}

// Render the active-exclusion chip strip + show/hide the bar based on
// whether a CSV is currently focused. Called after every refresh.
function updateExclusionUi() {
  const bar = document.getElementById("exclude-bar");
  if (!bar) return;
  const src = state.currentSource;
  bar.hidden = !src;
  if (!src) return;
  const bucket = exclusionsForSource(src);
  const active = document.getElementById("exclude-active");
  active.innerHTML = "";

  const addChip = (label, id) => {
    const chip = document.createElement("span");
    chip.className = "exclude-chip";
    chip.textContent = label;
    const undo = document.createElement("button");
    undo.type = "button";
    undo.title = "undo this exclusion";
    undo.textContent = "✕";
    undo.addEventListener("click", () => deleteExclusion(id));
    chip.appendChild(undo);
    active.appendChild(chip);
  };

  if (bucket.csv) addChip("whole CSV", bucket.csv);
  // Group rows / cols by id so a single multi-row exclusion shows as
  // ONE chip ("rows 5,8,12") rather than three chips.
  const rowsByExcl = new Map(); // id -> [row, ...]
  for (const [r, id] of bucket.rows) {
    if (!rowsByExcl.has(id)) rowsByExcl.set(id, []);
    rowsByExcl.get(id).push(r);
  }
  for (const [id, rows] of rowsByExcl) {
    rows.sort((a, b) => a - b);
    addChip(`rows ${rows.join(", ")}`, id);
  }
  const colsByExcl = new Map();
  for (const [c, id] of bucket.cols) {
    if (!colsByExcl.has(id)) colsByExcl.set(id, []);
    colsByExcl.get(id).push(c);
  }
  for (const [id, cols] of colsByExcl) {
    addChip(`cols ${cols.join(", ")}`, id);
  }
  // Disable the "Exclude this CSV" button when it's already done so
  // the user has to undo first.
  const csvBtn = document.getElementById("exclude-csv-btn");
  if (csvBtn) csvBtn.disabled = !!bucket.csv;
}

// Apply (or clear) exclusion marks on the currently-rendered CSV
// table without rebuilding it. Safe to call when no table is on
// screen - just no-ops.
function applyExclusionMarks() {
  const viewer = document.getElementById("csv-viewer");
  if (!viewer) return;
  const src = state.currentSource;
  if (!src) {
    viewer.classList.remove("csv-excluded");
    return;
  }
  const bucket = exclusionsForSource(src);
  viewer.classList.toggle("csv-excluded", !!bucket.csv);

  const table = viewer.querySelector("table");
  if (!table) return;
  // Column header strikethrough.
  for (const th of table.querySelectorAll("thead th")) {
    const col = th.dataset.col || "";
    th.classList.toggle("col-excluded", !!bucket.cols.get(col));
  }
  // Row + cell marks. We iterate per-row so we can apply the row
  // mark and then per-cell column mark in one pass.
  for (const tr of table.querySelectorAll("tbody tr")) {
    const r = Number(tr.dataset.row);
    tr.classList.toggle("row-excluded", Number.isFinite(r) && bucket.rows.has(r));
    for (const td of tr.querySelectorAll("td[data-col]")) {
      const col = td.dataset.col || "";
      td.classList.toggle("col-excluded", !!bucket.cols.get(col));
    }
  }
}

// Mark queue rows whose underlying cluster lives on an excluded
// slice. Cheap O(N) over visible queue rows; called after exclusion
// changes and after queue rebuilds.
function paintQueueExclusions() {
  const list = document.getElementById("queue");
  if (!list) return;
  for (const row of list.querySelectorAll(".row[data-queue-index]")) {
    const idx = Number(row.dataset.queueIndex);
    const c = state.queue[idx];
    if (!c) continue;
    const bucket = exclusionsForSource(c.source || "");
    let excluded = !!bucket.csv;
    if (!excluded && Number.isInteger(c.row) && c.row >= 0) {
      excluded = bucket.rows.has(c.row);
    }
    if (!excluded && Array.isArray(c.columns)) {
      excluded = c.columns.some((col) => bucket.cols.has(col));
    }
    row.classList.toggle("cluster-excluded", excluded);
  }
}

// Per-(source, column, error_type) projected-missed lookup keyed off
// the labelling-effort artifact. Used by renderCsv to attach a small
// "+N <et>" chip next to LLM-annotated cells, so the operator sees
// at a glance how much more of each error type the column is
// expected to surface beyond the cells already annotated. Returns
// ``null`` when the run has no per-arm artifact.

// Map a raw arm-status into operator-facing wording. The raw values
// come straight from src/adaptive/scoped_gt.py::evaluate_stop and
// can read like internal jargon ("exhausted" especially is easy to
// misread as "fully covered" when it actually means "we hit k_max
// with the detector still discovering"). The raw class name is kept
// for CSS / tooltip lookup.
const ARM_STATUS_LABEL = {
  "active":         "exploring",
  "saturated":      "converged",
  "exhausted":      "budget-capped",
  "insufficient":   "too sparse",
  "not_applicable": "n/a",
};
const ARM_STATUS_TOOLTIP = {
  "active":         "Still exploring \u2014 the adaptive loop has not yet decided this arm is done.",
  "saturated":      "Detector has converged: Good-Turing missing-mass dropped below the target and species count is sufficient. Additional runs are unlikely to surface new defects of this type in this column.",
  "exhausted":      "Hit the run-budget ceiling (k_max) before convergence. Good-Turing missing-mass is still above the target \u2014 the detector was still finding new things when we stopped.",
  "insufficient":   "Hit k_max with fewer than s_min distinct species \u2014 too sparse for the estimator to call either way. More runs or a different prompt may help.",
  "not_applicable": "This (scope_kind, error_type) combination is filtered out by the applicability whitelist.",
};
function armStatusLabel(raw) { return ARM_STATUS_LABEL[raw] || raw; }
function armStatusTooltip(raw) { return ARM_STATUS_TOOLTIP[raw] || ""; }

// Column-status filter options. Values match ARM_STATUS_LABEL outputs
// so the queue-side filter just compares the mapped label directly.
// "no-estimate" is a synthetic bucket for cells the labelling-effort
// artifact doesn't cover (e.g. single-pass runs / not-yet-touched
// columns) - useful when the operator wants to surface clusters that
// have NO coverage signal at all.
const COLUMN_STATUS_OPTIONS = [
  { value: "budget-capped", label: "budget-capped",
    title: "Hit the run-budget ceiling (k_max) before convergence \u2014 detector was still finding new things when we stopped. These columns most likely need further inspection." },
  { value: "exploring", label: "exploring",
    title: "Adaptive loop hasn't yet decided this column is done." },
  { value: "converged", label: "converged",
    title: "Detector converged: Good-Turing missing-mass dropped below the target on every applicable arm." },
  { value: "too sparse", label: "too sparse",
    title: "Hit k_max with too few distinct species for the estimator to call either way." },
  { value: "no-estimate", label: "(no estimate)",
    title: "No per-arm coverage signal for this (source, column). Either the run didn't produce a labelling-effort artifact, or this cell never appeared in any arm." },
];

// Returns the set of column-status filter values that match a queue
// item's (source, column) cells. Iterates every (source, column)
// pairing on the cluster and unions the arm-status labels found in
// the labelling-effort artifact. Items with no matching cell in the
// artifact contribute the synthetic "no-estimate" bucket so the
// "(no estimate)" filter option works as expected.
function columnStatusesForItem(item) {
  const cols = Array.isArray(item.columns) ? item.columns.filter(Boolean) : [];
  const sources = item.is_mt
    ? (Array.isArray(item.sheets) ? item.sheets.filter(Boolean) : [])
    : [item.source].filter(Boolean);
  const out = new Set();
  let hadCoveredCell = false;
  if (sources.length === 0 || cols.length === 0) {
    return new Set(["no-estimate"]);
  }
  for (const src of sources) {
    for (const col of cols) {
      const lookup = effortLookupForCell(src, col);
      if (!lookup) continue;
      hadCoveredCell = true;
      for (const arm of (lookup.cell.by_error_type || [])) {
        out.add(armStatusLabel(arm.status));
      }
    }
  }
  if (!hadCoveredCell) out.add("no-estimate");
  return out;
}

function effortLookupForCell(source, column) {
  const idx = state.labelingEffortIndex;
  if (idx) return idx.get(`${source}\x00${column}`) || null;
  // Fallback: artifact is loaded but the index hasn't been rebuilt
  // (e.g. a hot-patched code path before the next refreshLabelingEffort).
  const ef = state.labelingEffort;
  if (!ef || !ef.available || !Array.isArray(ef.cells)) return null;
  const cell = ef.cells.find((c) => c.source === source && c.column === column);
  if (!cell) return null;
  const byEt = new Map();
  for (const e of (cell.by_error_type || [])) byEt.set(e.error_type, e);
  return { cell, byEt };
}

function renderLabelingEffort() {
  const tbody = document.getElementById("effort-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  const cells = (state.labelingEffort && state.labelingEffort.cells) || [];
  if (!cells.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="9" class="muted">(no per-column estimates available yet)</td>`;
    tbody.appendChild(tr);
    return;
  }
  // Filter by selected source (matches the queue's filter so the two
  // panels feel coordinated). Workbook-token filters fall through to
  // "no filter" here because the per-arm artifact keys by source CSV.
  // ``state.filters.source`` is now an array; treat empty / workbook-
  // only selections as "show all", and otherwise accept any cell whose
  // source is in the picked set.
  const srcSel = state.filters?.source || [];
  const sourceSet = new Set(srcSel.filter((v) => !v.startsWith("workbook:")));
  const visible = cells.filter((c) => sourceSet.size === 0 || sourceSet.has(c.source));
  if (!visible.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="9" class="muted">(no estimates for the current source filter)</td>`;
    tbody.appendChild(tr);
    return;
  }
  // Top-N cap so giant runs don't blow up the DOM. Sorted server-side
  // by chao1_estimated_missed desc; we surface the top 100.
  const TOP_N = 100;
  const overflow = visible.length > TOP_N ? visible.length - TOP_N : 0;
  const slice = visible.slice(0, TOP_N);

  for (const c of slice) {
    const tr = document.createElement("tr");
    tr.dataset.source = c.source;
    tr.dataset.column = c.column;
    tr.title = (c.by_error_type || []).slice(0, 5).map((e) =>
      `${e.error_type}: seen ${e.n_observed}, missed ${e.estimated_missed.toFixed(1)} (${armStatusLabel(e.status)})`
    ).join("\n") || "no per-error-type breakdown";
    const armsTotal = c.arms_total || 0;
    const armsSat = (c.arms_saturated || 0) + (c.arms_not_applicable || 0);
    const massPct = `${((c.good_turing_missing_mass || 0) * 100).toFixed(1)}%`;
    const labelled = c.labelled_so_far || 0;
    const armsLabel = armsTotal
      ? `${armsSat}/${armsTotal}`
      : "\u2014";
    // Highlight cells where there's still substantial work to do.
    const remainingFrac = c.chao1_estimated_total
      ? (c.effort_remaining || 0) / c.chao1_estimated_total
      : 0;
    if (remainingFrac >= 0.5) tr.classList.add("effort-high");
    else if (remainingFrac >= 0.2) tr.classList.add("effort-mid");
    else tr.classList.add("effort-low");
    // Click-to-filter: set the source filter to this row's source.
    // Replaces any prior selection so the operator drills into one
    // CSV at a time when scanning the effort table.
    tr.addEventListener("click", () => {
      const ctrl = _multiFilters.get(document.getElementById("filter-source"));
      if (!ctrl) return;
      ctrl.setSelected([c.source]);
      state.filters.source = [c.source];
      updateSourceXlsxActions();
      renderLabelingEffort();
      renderColumnAuditPanel();
      refreshQueue({ keepSelection: false });
    });
    const td = (text, cls) => {
      const el = document.createElement("td");
      if (cls) el.className = cls;
      el.textContent = text;
      return el;
    };
    tr.appendChild(td(c.source, "src"));
    tr.appendChild(td(c.column, "col"));
    tr.appendChild(td(String(Math.round(c.n_observed || 0))));
    tr.appendChild(td((c.chao1_estimated_total || 0).toFixed(1)));
    tr.appendChild(td((c.chao1_estimated_missed || 0).toFixed(1)));
    tr.appendChild(td(massPct));
    tr.appendChild(td(String(labelled)));
    tr.appendChild(td(armsLabel, "arms"));
    // Column-audit action cell. The button is always present so the
    // operator can audit any cell (not just budget-capped ones), but
    // we mark budget-capped rows with a hot CSS class so they jump
    // out as the primary inspection target.
    const actionTd = document.createElement("td");
    actionTd.className = "audit-cell";
    // Stop click propagation so clicking the button doesn't also
    // trigger the row's "filter to this source" handler.
    actionTd.addEventListener("click", (ev) => ev.stopPropagation());
    actionTd.appendChild(makeColumnAuditControls(c));
    tr.appendChild(actionTd);
    tbody.appendChild(tr);
  }
  if (overflow) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="9" class="muted">(\u2026 +${overflow} more cells; widen the source filter to drill in)</td>`;
    tbody.appendChild(tr);
  }
}

// ----------------------------------------------------------- column audit
//
// Column-wide web verification: kick off a background job that audits
// every distinct value in a (source, column) - including never-flagged
// cells - via the Foundry agent, then surface the findings inline.
// Used primarily on budget-capped columns where the LLM detector hit
// its run-budget cap while still finding new defects (so we suspect
// it missed cells).
//
// State machine per (source, column):
//   idle  -> running -> succeeded / failed / cancelled
// Re-clicking while running is a no-op (button stays disabled).

function columnAuditKey(source, column) {
  return `${source}\x00${column}`;
}

function makeColumnAuditControls(cell) {
  /* Returns the action-cell content for one row in the effort table:
   * a "Verify column" button + an inline status badge in the idle
   * state, OR a mini progress bar when the audit job for this row
   * is in flight. The two views share the same wrapper; we toggle
   * them via ``applyColumnAuditState`` so a job transitioning from
   * idle -> running -> done updates in place without rebuilding the
   * row. */
  const wrap = document.createElement("div");
  wrap.className = "audit-controls";
  wrap.dataset.source = cell.source;
  wrap.dataset.column = cell.column;
  const key = columnAuditKey(cell.source, cell.column);

  // Highlight rows with at least one budget-capped (status="exhausted")
  // arm so the operator can spot the primary audit targets at a
  // glance. Uses the per-error-type breakdown that's already on the
  // effort cell; falls back to false if missing.
  const isCapped = (cell.by_error_type || []).some((e) => e.status === "exhausted");
  if (isCapped) wrap.classList.add("audit-capped");

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "audit-btn";
  btn.textContent = "Verify column";
  btn.title = isCapped
    ? "Run the column-wide web verifier on this BUDGET-CAPPED column. The agent will audit every distinct value (including never-flagged cells) for missed errors."
    : "Run the column-wide web verifier on every distinct value of this column.";
  btn.addEventListener("click", () => triggerColumnAudit(cell.source, cell.column, btn));
  wrap.appendChild(btn);

  // Status / counts badge. Shows in-flight job state OR cached audit
  // counts when an audit was already run for this pair. Hidden via
  // ``[hidden]`` (not display:none) so applyColumnAuditState can
  // toggle it without recomputing layout each tick.
  const badge = document.createElement("span");
  badge.className = "audit-badge";
  badge.dataset.key = key;
  wrap.appendChild(badge);

  // Mini progress bar - hidden until the job for this row is running.
  // Rendered as: [text - 70px] [bar - 60px] [cancel - tiny ✕]
  // Compact enough that it fits in the same column width as the
  // "Verify column" button without forcing the table to reflow.
  const prog = document.createElement("span");
  prog.className = "audit-mini-progress";
  prog.hidden = true;
  prog.innerHTML = `
    <span class="audit-mini-text"></span>
    <span class="audit-mini-bar"><span class="audit-mini-fill"></span></span>
    <button type="button" class="audit-mini-cancel" title="Cancel this column-audit job">\u2715</button>
  `;
  wrap.appendChild(prog);
  prog.querySelector(".audit-mini-cancel").addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    cancelColumnAuditJob(cell.source, cell.column);
  });

  applyColumnAuditState(wrap, cell.source, cell.column);
  return wrap;
}

function applyColumnAuditState(wrap, source, column) {
  /* Toggle a row's audit controls between the idle (button + badge)
   * view and the running (mini progress bar) view. Idempotent and
   * cheap so it's safe to call from polling loops.
   *
   * "Running" is true when ANY of these hold:
   *   - the per-row audit job is in flight, OR
   *   - the bulk audit job is in flight AND this row is in its
   *     queue (i.e. budget-capped, since bulk audits every capped
   *     column), OR
   *   - the live progress payload is currently auditing this exact
   *     (source, column) pair (covers the case where the page was
   *     reloaded mid-bulk-job and we don't yet have state about it).
   */
  if (!wrap) return;
  const key = columnAuditKey(source, column);
  const job = state.columnAuditJobs.get(key);
  const bulkJob = state._bulkAuditJob;
  const bulkActive = bulkJob && (bulkJob.status === "running" || bulkJob.status === "queued");
  const isCapped = wrap.classList.contains("audit-capped");
  const progPayload = state._auditProgressLastPayload;
  const progressMatchesRow = progPayload
    && progPayload.available && !progPayload.done
    && progPayload.current_source === source
    && progPayload.current_column === column;

  // Bulk audits every budget-capped column sequentially, so every
  // capped row is in the queue and should show a progress view.
  const bulkOnThisRow = bulkActive && isCapped;

  const btn = wrap.querySelector(".audit-btn");
  const badge = wrap.querySelector(".audit-badge");
  const prog = wrap.querySelector(".audit-mini-progress");
  if (!btn || !badge || !prog) return;

  const running = (job && (job.status === "running" || job.status === "queued"))
    || bulkOnThisRow
    || progressMatchesRow;

  if (running) {
    btn.hidden = true;
    badge.hidden = true;
    prog.hidden = false;
    updateColumnAuditMiniProgress(prog, source, column);
  } else {
    btn.hidden = false;
    badge.hidden = false;
    prog.hidden = true;
    applyColumnAuditBadge(badge, source, column);
  }
}

function updateColumnAuditMiniProgress(prog, source, column) {
  /* Update one per-row mini progress bar from the latest
   * _progress.json payload. When the payload is for THIS column, we
   * use values_done/values_total directly; for queued / not-yet-
   * started rows we show an indeterminate state. */
  if (!prog) return;
  const txt = prog.querySelector(".audit-mini-text");
  const fill = prog.querySelector(".audit-mini-fill");
  const payload = state._auditProgressLastPayload;
  let frac = 0;
  let label = "starting\u2026";
  let indeterminate = true;
  if (payload && payload.available && !payload.done
      && payload.current_source === source
      && payload.current_column === column) {
    const valTotal = Number(payload.values_total || 0);
    const valDone = Number(payload.values_done || 0);
    if (valTotal > 0) {
      frac = valDone / valTotal;
      label = `${valDone}/${valTotal}`;
      indeterminate = false;
    } else {
      label = "preparing\u2026";
    }
  } else {
    // Per-row job is in-flight but the script hasn't reached this
    // column yet (bulk run, queued behind earlier columns) or the
    // progress artifact hasn't been written yet. Show the queue
    // position when available.
    if (payload && payload.available && !payload.done) {
      const totalCols = Number(payload.total_columns || 0);
      const colsDone = Number(payload.columns_done || 0);
      if (totalCols > 0) {
        label = `queued ${colsDone + 1}/${totalCols}`;
      }
    }
  }
  if (!Number.isFinite(frac)) frac = 0;
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  if (txt) txt.textContent = label;
  if (fill) {
    fill.style.width = indeterminate ? "100%" : `${(frac * 100).toFixed(0)}%`;
    fill.classList.toggle("indeterminate", indeterminate);
  }
}

function refreshAllColumnAuditControlStates() {
  /* Walk every effort-table row and refresh its audit-controls view
   * to match the latest job + progress state. Cheap because the DOM
   * only contains rows for the current source filter. */
  const tbody = document.getElementById("effort-body");
  if (!tbody) return;
  for (const wrap of tbody.querySelectorAll(".audit-controls")) {
    const s = wrap.dataset.source;
    const c = wrap.dataset.column;
    if (!s || !c) continue;
    applyColumnAuditState(wrap, s, c);
  }
}

async function cancelColumnAuditJob(source, column) {
  /* Per-row cancel: kills the column-audit job that was launched
   * for this row. For bulk runs this terminates the entire bulk
   * job (the script audits columns sequentially), which is the
   * correct semantics: cancelling here means "stop auditing this
   * column" and the script can't selectively skip ahead. */
  const key = columnAuditKey(source, column);
  let jobId = null;
  const job = state.columnAuditJobs.get(key);
  if (job && (job.status === "running" || job.status === "queued")) {
    jobId = job.jobId;
  } else if (state._bulkAuditJob
      && (state._bulkAuditJob.status === "running"
          || state._bulkAuditJob.status === "queued")) {
    jobId = state._bulkAuditJob.jobId;
  }
  if (!jobId) {
    toast("no audit job is running for this row");
    return;
  }
  try {
    await api(`/api/launcher/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    });
    toast(`cancelled audit job ${jobId}`);
  } catch (err) {
    toast(`cancel failed: ${err.message || err}`);
  }
}

function applyColumnAuditBadge(badge, source, column) {
  /* Set the badge text + classes based on whatever state we know
   * about the (source, column) pair. Called both during initial
   * render and on every job status / cached-audit refresh. */
  const key = columnAuditKey(source, column);
  badge.classList.remove("running", "succeeded", "failed", "fn-found", "no-fn");
  const job = state.columnAuditJobs.get(key);
  if (job && (job.status === "running" || job.status === "queued")) {
    badge.textContent = "running\u2026";
    badge.classList.add("running");
    badge.title = `Audit job ${job.jobId} is in flight.`;
    return;
  }
  if (job && job.status === "failed") {
    badge.textContent = "failed";
    badge.classList.add("failed");
    badge.title = `Audit job ${job.jobId} exited with non-zero status \u2014 see the launcher log.`;
    return;
  }
  // Cached audit available: show "<fn> FN / <total>" when we found
  // missed errors, "ok" when we audited and found nothing.
  const idx = state.columnAuditIndex.get(key);
  if (idx) {
    const c = idx.counts || {};
    const fn = Number(c.fn_candidates || 0);
    const total = Number(c.total || 0);
    if (fn > 0) {
      badge.textContent = `${fn} FN / ${total}`;
      badge.classList.add("fn-found");
      badge.title = `Audit found ${fn} potential missed error${fn === 1 ? "" : "s"} (verdict=confirmed_error on values the LLM never flagged) out of ${total} verified value${total === 1 ? "" : "s"}.`;
    } else {
      badge.textContent = total > 0 ? `ok (${total})` : "ok";
      badge.classList.add("no-fn");
      badge.title = total > 0
        ? `Audit verified ${total} distinct value${total === 1 ? "" : "s"} \u2014 no missed errors found.`
        : "Audit completed with no verifiable values.";
    }
    return;
  }
  badge.textContent = "";
  badge.title = "";
}

function refreshAuditBadge(source, column) {
  /* Find the audit-controls cell for one (source, column) pair (if
   * currently in the DOM) and re-apply its state - either the badge
   * (idle / done) or the mini progress bar (running). No-op when
   * the row has been scrolled / filtered out of the effort table. */
  const tbody = document.getElementById("effort-body");
  if (!tbody) return;
  const tr = tbody.querySelector(
    `tr[data-source="${cssEscape(source)}"][data-column="${cssEscape(column)}"]`,
  );
  if (!tr) return;
  const wrap = tr.querySelector(".audit-controls");
  if (wrap) applyColumnAuditState(wrap, source, column);
}

function cssEscape(v) {
  /* Minimal CSS-attribute-value escape: backslash-escape every
   * char that isn't [a-zA-Z0-9_-]. Sufficient for our column names
   * (which legitimately contain spaces, dots, slashes etc.). */
  return String(v).replace(/[^A-Za-z0-9_-]/g, (c) => `\\${c}`);
}

async function triggerColumnAudit(source, column, btn) {
  /* Launch a column_verify job for one (source, column). Shows a
   * spinner on the button while we wait for the job-id from the
   * server; the badge is then driven by the polling loop. */
  if (!state.run) return;
  const key = columnAuditKey(source, column);
  // Re-click while a job is already in flight is a no-op.
  const existing = state.columnAuditJobs.get(key);
  if (existing && (existing.status === "running" || existing.status === "queued")) {
    toast(`already auditing ${source} \u00B7 ${column}`);
    return;
  }
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "starting\u2026";
  try {
    const body = {
      kind: "column_verify",
      run_id: state.run.run_id,
      params: {
        source,
        column,
        // Reasonable defaults; advanced users can run the CLI
        // directly. ``concurrency=4`` matches the standard verifier.
        concurrency: 4,
      },
    };
    const job = await api(`/api/launcher/jobs`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.columnAuditJobs.set(key, {
      jobId: job.job_id,
      status: job.status || "running",
      startedAt: Date.now(),
      source,
      column,
    });
    refreshAuditBadge(source, column);
    pollColumnAuditJob(source, column);
    ensureAuditProgressPoller();
    toast(`auditing ${source} \u00B7 ${column}\u2026`);
  } catch (err) {
    toast(`audit launch failed: ${err.message || err}`);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

async function pollColumnAuditJob(source, column) {
  /* Poll the job's status every 2s until it leaves "running". On
   * success we fetch the audit detail and refresh the panel; on
   * failure we leave the badge in "failed" state so the operator can
   * retry. */
  const key = columnAuditKey(source, column);
  const job = state.columnAuditJobs.get(key);
  if (!job) return;
  const POLL_MS = 2000;
  const MAX_POLLS = 600; // 20 min ceiling; jobs longer than that should be inspected via the launcher tab
  let polls = 0;
  while (polls < MAX_POLLS) {
    polls += 1;
    await new Promise((r) => setTimeout(r, POLL_MS));
    let fresh;
    try {
      fresh = await api(`/api/launcher/jobs/${encodeURIComponent(job.jobId)}`);
    } catch (err) {
      console.warn(`column-audit job ${job.jobId} poll failed:`, err);
      continue;
    }
    const status = fresh.status || "running";
    job.status = status;
    state.columnAuditJobs.set(key, job);
    refreshAuditBadge(source, column);
    if (status === "running" || status === "queued") continue;
    // Terminal state.
    if (status === "succeeded") {
      await refreshColumnAuditDetail(source, column);
      toast(`audit done: ${source} \u00B7 ${column}`);
    } else {
      toast(`audit ${status}: ${source} \u00B7 ${column} \u2014 see launcher log`);
    }
    return;
  }
  // Timed out polling. Leave the job in "running" state so the
  // operator can see we lost track of it; the launcher tab is the
  // source of truth.
  console.warn(`column-audit job ${job.jobId} poll timed out after ${MAX_POLLS} attempts`);
}

async function recoverInFlightAuditJobs() {
  /* On page load, scan the launcher for any in-flight column_verify
   * jobs and reconstruct ``state._bulkAuditJob`` /
   * ``state.columnAuditJobs`` so the per-row controls render the
   * progress bar immediately - rather than waiting for the operator
   * to re-launch (which would orphan the running child) or for a
   * page refresh after the child finishes. */
  if (!state.run) return;
  let payload;
  try {
    payload = await api(`/api/launcher/jobs`);
  } catch (err) {
    console.warn("launcher jobs list unavailable:", err);
    return;
  }
  const all = (payload && payload.jobs) || [];
  const cverify = all.filter((j) => j && j.kind === "column_verify"
    && (j.status === "running" || j.status === "queued"));
  // Filter to jobs scoped to THIS run. Recovered jobs have
  // ``run_id: null`` because the JobManager didn't reparse argv;
  // walk the argv ourselves to determine the run.
  const ours = [];
  for (const j of cverify) {
    let rid = j.run_id || null;
    const argv = j.argv || [];
    if (!rid) {
      for (let i = 0; i < argv.length; i += 1) {
        if (argv[i] === "--run-id" && i + 1 < argv.length) {
          rid = argv[i + 1];
          break;
        }
      }
    }
    if (!rid || rid === state.run.run_id) ours.push(j);
  }
  for (const j of ours) {
    const argv = j.argv || [];
    // Recover --source / --column or --all-budget-capped from argv.
    let isBulk = false;
    let src = null;
    let col = null;
    for (let i = 0; i < argv.length; i += 1) {
      const a = argv[i];
      if (a === "--all-budget-capped") isBulk = true;
      else if (a === "--source" && i + 1 < argv.length) src = argv[i + 1];
      else if (a === "--column" && i + 1 < argv.length) col = argv[i + 1];
    }
    if (isBulk) {
      // Multiple bulk jobs are unusual; track the most recent.
      state._bulkAuditJob = {
        jobId: j.job_id,
        status: j.status || "running",
        startedAt: j.started_at || Date.now(),
      };
      pollBulkColumnAudit();
      console.log(`recovered in-flight bulk column-audit job ${j.job_id}`);
    } else if (src && col) {
      state.columnAuditJobs.set(columnAuditKey(src, col), {
        jobId: j.job_id,
        status: j.status || "running",
        startedAt: j.started_at || Date.now(),
        source: src,
        column: col,
      });
      pollColumnAuditJob(src, col);
      console.log(`recovered in-flight column-audit job ${j.job_id} (${src} \u00B7 ${col})`);
    }
  }
}

async function refreshColumnAuditIndex() {
  /* Bootstrap-time fetch of the per-(source, column) audit manifest.
   * The manifest is small (one entry per audited pair) and tells us
   * which pairs already have on-disk records so the badge can show
   * cached counts before the operator re-clicks. */
  if (!state.run) return;
  let payload;
  try {
    payload = await api(`${API}/column-verifications`);
  } catch (err) {
    console.warn("column-verifications index unavailable:", err);
    return;
  }
  const idx = new Map();
  for (const item of (payload.items || [])) {
    if (!item || !item.source || !item.column) continue;
    idx.set(columnAuditKey(item.source, item.column), {
      counts: item.counts || {},
      ts: item.ts || "",
      path: item.path || "",
    });
  }
  state.columnAuditIndex = idx;
  // Re-render the effort table so badges pick up cached counts.
  renderLabelingEffort();
  renderColumnAuditPanel();
}

async function refreshColumnAuditDetail(source, column) {
  /* Fetch the full record list for one (source, column) and update
   * both the index (so the badge gets fresh counts) and the panel. */
  if (!state.run) return;
  const key = columnAuditKey(source, column);
  let payload;
  try {
    const url = `${API}/column-verifications/detail`
      + `?source=${encodeURIComponent(source)}`
      + `&column=${encodeURIComponent(column)}`;
    payload = await api(url);
  } catch (err) {
    console.warn(`column-verifications detail failed for ${source} \u00B7 ${column}:`, err);
    return;
  }
  if (!payload || !payload.available) {
    state.columnAuditIndex.delete(key);
    state.columnAuditDetail.delete(key);
  } else {
    state.columnAuditIndex.set(key, {
      counts: payload.counts || {},
      ts: "",
      path: "",
    });
    state.columnAuditDetail.set(key, {
      source: payload.source,
      column: payload.column,
      counts: payload.counts || {},
      items: payload.items || [],
    });
  }
  rebuildAuditCellLookup(source, column);
  refreshAuditBadge(source, column);
  renderColumnAuditPanel();
  // If the CSV viewer is showing this source, repaint the audit
  // markers for the column we just refreshed.
  if (state.currentSource === source) {
    paintAuditMarkersForColumn(source, column);
  }
}

function auditCellKey(source, column, row) {
  return `${source}\x00${column}\x00${row}`;
}

function rebuildAuditCellLookup(source, column) {
  /* Project ``columnAuditDetail[source, column].items`` (one record
   * per distinct value) into a per-row lookup so the CSV painter +
   * cell-popover can answer "what verdict did the audit give to
   * (source, col, row)?" in O(1). Each item is referenced (not
   * copied) so canonical/evidence/reasoning are still available
   * without an extra fetch. */
  const dkey = columnAuditKey(source, column);
  const detail = state.columnAuditDetail.get(dkey);
  // Clear any prior entries for this (source, column) regardless of
  // whether new detail is available - prevents stale paint when an
  // audit is rerun and a value drops out of the verified set.
  for (const k of Array.from(state.columnAuditCellLookup.keys())) {
    if (k.startsWith(`${source}\x00${column}\x00`)) {
      state.columnAuditCellLookup.delete(k);
    }
  }
  if (!detail || !detail.items) return;
  for (const item of detail.items) {
    if (!item || !Array.isArray(item.rows)) continue;
    for (const r of item.rows) {
      const n = Number(r);
      if (!Number.isFinite(n)) continue;
      state.columnAuditCellLookup.set(auditCellKey(source, column, n), item);
    }
  }
}

function ensureAuditDetailsForSource(source) {
  /* Walk the audit index and kick off detail fetches for every
   * audited (source, *) pair we don't already have items for. The
   * fetch updates the per-cell lookup + repaints when it arrives.
   * Best-effort + idempotent: re-invoking does not refetch pairs
   * already in ``columnAuditDetail``. */
  if (!source) return;
  for (const [key, idx] of state.columnAuditIndex) {
    // Key is "<source>\x00<column>".
    const sep = key.indexOf("\x00");
    if (sep < 0) continue;
    const s = key.slice(0, sep);
    const c = key.slice(sep + 1);
    if (s !== source) continue;
    if (state.columnAuditDetail.has(key)) {
      // Already have detail - just (re)build the cell lookup in case
      // it was cleared (e.g. on a fresh load).
      rebuildAuditCellLookup(s, c);
      continue;
    }
    // Skip empty audits (no records): nothing to surface.
    const counts = (idx && idx.counts) || {};
    if (!Number(counts.total || 0)) continue;
    refreshColumnAuditDetail(s, c).catch(() => { /* soft-fail */ });
  }
}

function paintAuditMarkersForColumn(source, column) {
  /* Add / remove the verdict dot inside every <td> in the currently-
   * rendered CSV viewer that belongs to ``source.column``. Idempotent
   * on repeated calls - existing marker spans are removed before
   * the new one is added so reruns don't accumulate badges. */
  if (state.currentSource !== source) return;
  const viewer = $("#csv-viewer");
  if (!viewer) return;
  // Mark the column header so the operator can scan which columns
  // have audit data without inspecting every row.
  const th = viewer.querySelector(
    `thead th[data-col="${cssEscape(column)}"]`,
  );
  if (th) {
    th.classList.add("audited");
    // Count FN candidates in this column for a header tooltip.
    const dkey = columnAuditKey(source, column);
    const detail = state.columnAuditDetail.get(dkey);
    if (detail) {
      const counts = detail.counts || {};
      const fn = Number(counts.fn_candidates || 0);
      const total = Number(counts.total || 0);
      const tipBase = th.getAttribute("data-audit-tip-base")
        || th.getAttribute("title") || "";
      if (!th.getAttribute("data-audit-tip-base")) {
        th.setAttribute("data-audit-tip-base", tipBase);
      }
      const tip = `Web-audited: ${total} value(s) verified`
        + (fn > 0 ? `, ${fn} FN candidate(s)` : "");
      th.setAttribute("title", tipBase ? `${tipBase}\n${tip}` : tip);
    }
  }
  const tds = viewer.querySelectorAll(
    `td.clickable[data-col="${cssEscape(column)}"]`,
  );
  for (const td of tds) {
    paintAuditMarkerOnCell(td, source, column);
  }
}

function paintAuditMarkerOnCell(td, source, column) {
  // Strip any prior marker.
  td.classList.remove(
    "audit-verdict-confirmed_error",
    "audit-verdict-false_positive",
    "audit-verdict-ambiguous",
    "audit-verdict-errored",
    "audit-fn-candidate-cell",
    "audit-marked",
  );
  // Older markup used a real <span class="audit-mark"> child to
  // render the corner glyph; that polluted ``td.textContent`` (and
  // clipboard copies) with "value?" / "valueE", which any future
  // code reading the cell text would mistake for the actual value.
  // Strip leftovers from older renders so we converge on the new
  // pseudo-element scheme below.
  for (const old of td.querySelectorAll(":scope > .audit-mark")) old.remove();
  const row = Number(td.getAttribute("data-row"));
  if (!Number.isFinite(row)) return;
  const item = state.columnAuditCellLookup.get(auditCellKey(source, column, row));
  if (!item) return;
  // ``errored`` short-circuits verdict in the index summary; mirror
  // that so the cell mark uses the same bucket.
  const verdict = item.error
    ? "errored"
    : String(item.verdict || "ambiguous");
  td.classList.add(`audit-verdict-${verdict}`);
  td.classList.add("audit-marked");
  const isFn = verdict === "confirmed_error" && !item.was_flagged_by_llm;
  if (isFn) td.classList.add("audit-fn-candidate-cell");
  // Tooltip: short verdict line + canonical/error so hover still
  // works even when the popover is closed.
  const parts = [`audit: ${verdict}`];
  if (isFn) parts.push("LLM did NOT flag this cell");
  if (item.canonical_value) parts.push(`canonical: ${item.canonical_value}`);
  if (item.error) parts.push(`error: ${item.error}`);
  // Don't clobber the existing title from cell-state; merge with it.
  const existing = td.getAttribute("title") || "";
  const merged = existing
    ? existing.split("\n").filter((s) => !s.startsWith("audit:")).concat(parts).join("\n")
    : parts.join("\n");
  td.setAttribute("title", merged);
  // The corner glyph itself is rendered by a CSS ``::after`` rule
  // keyed off the verdict / FN classes above. We deliberately do NOT
  // append a real DOM child here so:
  //   - ``td.textContent`` returns only the cell's literal value
  //     (any export / clipboard-copy / textContent-based scrape
  //      stays faithful to the underlying CSV).
  //   - the marker survives ``innerHTML`` rewrites without needing a
  //     repaint pass to re-inject the glyph.
  //   - assistive tech / accessibility tree skips the marker.
}
function paintAuditMarkersForSource(source) {
  /* Repaint every audited column belonging to ``source`` against the
   * currently-rendered CSV. Called after CSV (re)build + after a
   * detail fetch arrives. */
  if (!source) return;
  for (const [key, idx] of state.columnAuditIndex) {
    const sep = key.indexOf("\x00");
    if (sep < 0) continue;
    const s = key.slice(0, sep);
    const c = key.slice(sep + 1);
    if (s !== source) continue;
    const counts = (idx && idx.counts) || {};
    if (!Number(counts.total || 0)) continue;
    paintAuditMarkersForColumn(s, c);
  }
}

function renderAuditFindingsForCell(source, row, col) {
  /* Build the "Web audit" section for the cell popover. Returns ""
   * when no audit verdict exists for this (source, row, col); the
   * caller injects the result inline so an empty string disappears
   * cleanly without leaving structural HTML behind. */
  const item = state.columnAuditCellLookup.get(auditCellKey(source, col, row));
  if (!item) return "";
  const verdict = item.error
    ? "errored"
    : String(item.verdict || "ambiguous");
  const isFn = verdict === "confirmed_error" && !item.was_flagged_by_llm;
  const fnFlag = isFn
    ? `<span class="audit-fn-pill" title="The detector did NOT flag this cell, but the web verifier confirmed it as an error - candidate False Negative.">FN candidate</span>`
    : "";
  const flagFlag = !isFn && item.was_flagged_by_llm
    ? `<span class="audit-llm-flag-pill" title="LLM detector also flagged this cell.">LLM flagged</span>`
    : "";
  const conf = Number.isFinite(Number(item.confidence))
    ? ` \u00B7 confidence ${(Number(item.confidence) * 100).toFixed(0)}%`
    : "";
  const canonicalHtml = item.canonical_value
    ? `<div class="audit-pop-row"><b>Canonical:</b> <span class="canon">${escape(String(item.canonical_value))}</span></div>`
    : "";
  const errorHtml = item.error
    ? `<div class="audit-pop-row muted">verifier error: ${escape(String(item.error))}</div>`
    : "";
  const reasoning = String(item.reasoning || "").trim();
  const reasoningHtml = reasoning
    ? `<details class="audit-pop-reasoning"><summary>Reasoning</summary><div class="reasoning-text">${escape(reasoning)}</div></details>`
    : "";
  // Up to 3 evidence URLs, truncated to keep the popover compact.
  const evidence = Array.isArray(item.evidence) ? item.evidence.slice(0, 3) : [];
  const evidenceHtml = evidence.length
    ? `<ul class="audit-pop-evidence">`
      + evidence.map((ev) => {
        const u = String(ev.url || "");
        const t = String(ev.title || u || "(source)");
        const q = String(ev.quote || "").trim();
        const trimmed = u.length > 60 ? u.slice(0, 57) + "\u2026" : u;
        return `<li>`
          + `<a href="${escape(u)}" target="_blank" rel="noopener noreferrer" title="${escape(u)}">${escape(t || trimmed)}</a>`
          + (q ? `<div class="audit-evidence-quote">${escape(q)}</div>` : "")
          + `</li>`;
      }).join("")
      + `</ul>`
    : "";
  const moreEv = (Array.isArray(item.evidence) && item.evidence.length > 3)
    ? `<div class="muted" style="font-size:11px;">+${item.evidence.length - 3} more evidence link(s) in the column-audit panel</div>`
    : "";
  // Verdict pill + headline. The classes mirror the audit-detail-table
  // verdict palette so colours stay consistent across the UI.
  return `<div class="audit-pop-block">
    <div class="audit-pop-head">
      <span class="audit-pop-label">Web audit</span>
      <span class="audit-verdict-pill verdict-${verdict}">${escape(verdict)}</span>
      ${fnFlag}${flagFlag}
      <span class="muted" style="font-size:11px;">${conf}</span>
    </div>
    ${canonicalHtml}
    ${errorHtml}
    ${reasoningHtml}
    ${evidenceHtml}
    ${moreEv}
  </div>`;
}

function renderColumnAuditPanel() {
  /* Render the "Column audit findings" section under the effort
   * panel. One collapsible block per (source, column) with cached
   * detail loaded; clicking the summary lazy-loads the records the
   * first time it expands. */
  const panel = document.getElementById("column-audit-panel");
  const body = document.getElementById("column-audit-body");
  const headline = document.getElementById("column-audit-headline");
  if (!panel || !body || !headline) return;
  body.innerHTML = "";

  const items = Array.from(state.columnAuditIndex.entries()).map(([key, idx]) => {
    const [source, column] = key.split("\x00");
    return { source, column, idx };
  });
  // Filter to the source-filter set (matches the queue + effort
  // table). Empty filter means show all.
  const srcSel = state.filters?.source || [];
  const sourceSet = new Set(srcSel.filter((v) => !v.startsWith("workbook:")));
  const visible = items.filter((it) =>
    sourceSet.size === 0 || sourceSet.has(it.source),
  );

  if (!visible.length) {
    panel.hidden = true;
    headline.textContent = "";
    return;
  }
  panel.hidden = false;
  // Default to expanded the first time the panel becomes visible
  // so the operator immediately sees freshly-audited findings, but
  // preserve the user's collapse choice on subsequent renders so
  // they can keep the panel folded while many audits run. The
  // toggle listener is wired once and ignores the initial
  // programmatic open we trigger here.
  if (!panel.dataset.toggleHooked) {
    panel.dataset.toggleHooked = "1";
    panel.addEventListener("toggle", () => {
      // Skip the bootstrap event the very first time: on render we
      // open the panel ourselves, which fires `toggle` even though
      // the user didn't click. Mark the flag only after that.
      if (panel.dataset.bootstrapDone) {
        panel.dataset.userToggled = "1";
      } else {
        panel.dataset.bootstrapDone = "1";
      }
    });
  }
  if (!panel.dataset.userToggled) {
    panel.open = true;
  }
  // Sort: pairs with FN candidates first, then by ts desc so freshly-
  // audited columns float up.
  visible.sort((a, b) => {
    const fa = Number(a.idx.counts?.fn_candidates || 0);
    const fb = Number(b.idx.counts?.fn_candidates || 0);
    if (fa !== fb) return fb - fa;
    return String(b.idx.ts || "").localeCompare(String(a.idx.ts || ""));
  });

  let totalFN = 0;
  let totalAudited = 0;
  for (const it of visible) {
    totalFN += Number(it.idx.counts?.fn_candidates || 0);
    totalAudited += Number(it.idx.counts?.total || 0);
  }
  headline.textContent =
    `${visible.length} column${visible.length === 1 ? "" : "s"} audited \u00B7 ` +
    `${totalAudited} value${totalAudited === 1 ? "" : "s"} \u00B7 ` +
    `${totalFN} potential missed error${totalFN === 1 ? "" : "s"}`;

  for (const it of visible) {
    body.appendChild(buildColumnAuditBlock(it.source, it.column, it.idx));
  }
}

function buildColumnAuditBlock(source, column, idx) {
  /* One collapsible <details> per (source, column). Stays collapsed
   * until the user clicks the summary; on first expand we lazy-fetch
   * the detail. Subsequent expands reuse the cached detail. */
  const key = columnAuditKey(source, column);
  const c = idx.counts || {};
  const fn = Number(c.fn_candidates || 0);
  const total = Number(c.total || 0);
  const conf = Number(c.confirmed_error || 0);
  const fp = Number(c.false_positive || 0);
  const amb = Number(c.ambiguous || 0);
  const errored = Number(c.errored || 0);

  const wrap = document.createElement("details");
  wrap.className = "column-audit-block";
  if (fn > 0) wrap.classList.add("has-fn");
  wrap.open = false;

  const summary = document.createElement("summary");
  // Counts pills inside the summary so the operator can scan multiple
  // pairs without expanding any.
  summary.innerHTML =
    `<span class="audit-source">${escape(source)}</span>` +
    `<span class="audit-column">${escape(column)}</span>` +
    (fn > 0
      ? `<span class="pill pill-fn">${fn} potential missed</span>`
      : `<span class="pill pill-ok">no missed errors</span>`) +
    `<span class="pill pill-total">${total} verified</span>` +
    (conf > 0 ? `<span class="pill pill-conf">${conf} confirmed_error</span>` : "") +
    (fp > 0 ? `<span class="pill pill-fp">${fp} false_positive</span>` : "") +
    (amb > 0 ? `<span class="pill pill-amb">${amb} ambiguous</span>` : "") +
    (errored > 0 ? `<span class="pill pill-err">${errored} errored</span>` : "");
  wrap.appendChild(summary);

  const slot = document.createElement("div");
  slot.className = "column-audit-detail";
  slot.dataset.loaded = "0";
  wrap.appendChild(slot);

  // Lazy-load the detail on first expand.
  wrap.addEventListener("toggle", async () => {
    if (!wrap.open) return;
    if (slot.dataset.loaded === "1") return;
    const cached = state.columnAuditDetail.get(key);
    if (cached) {
      renderColumnAuditDetail(slot, cached);
      slot.dataset.loaded = "1";
      return;
    }
    slot.innerHTML = `<p class="muted">Loading\u2026</p>`;
    await refreshColumnAuditDetail(source, column);
    const fresh = state.columnAuditDetail.get(key);
    if (fresh) {
      renderColumnAuditDetail(slot, fresh);
      slot.dataset.loaded = "1";
    } else {
      slot.innerHTML = `<p class="muted">No findings (file may have been removed).</p>`;
    }
  });
  return wrap;
}

function renderColumnAuditDetail(slot, detail) {
  /* Render the actual record table. Records are server-sorted with
   * FN candidates first, then confirmed_error, then ambiguous, then
   * false_positive. Each row carries:
   *   - row indices (clickable to jump to the cell viewer)
   *   - the verified value
   *   - verdict pill
   *   - canonical_value (if set)
   *   - evidence URLs (truncated)
   *   - "Mark as FN" button for unflagged confirmed_error rows
   */
  slot.innerHTML = "";
  const items = detail.items || [];
  if (!items.length) {
    slot.innerHTML = `<p class="muted">No records.</p>`;
    return;
  }

  const table = document.createElement("table");
  table.className = "audit-detail-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th title="Row indices in the source where this value appears.">Rows</th>
        <th title="The cell value verified.">Value</th>
        <th title="Agent verdict.">Verdict</th>
        <th title="Was this cell already flagged by the LLM detector?">LLM</th>
        <th title="Canonical replacement (if the agent confirmed an error).">Canonical</th>
        <th title="Web Search evidence URLs the agent cited.">Evidence</th>
        <th title="Per-record actions: jump to the cell, mark as FN.">Actions</th>
      </tr>
    </thead>
    <tbody></tbody>`;
  const tbody = table.querySelector("tbody");

  for (const r of items) {
    const tr = document.createElement("tr");
    const verdict = r.error ? "errored" : String(r.verdict || "ambiguous");
    tr.classList.add(`audit-${verdict.replace(/[^A-Za-z0-9_-]/g, "-")}`);
    const fnCandidate = verdict === "confirmed_error" && !r.was_flagged_by_llm;
    if (fnCandidate) tr.classList.add("audit-fn-candidate");

    // Rows column: comma-list, click to jump.
    const rows = Array.isArray(r.rows) ? r.rows : [];
    const rowsTd = document.createElement("td");
    rowsTd.className = "rows";
    const showRows = rows.slice(0, 5);
    rowsTd.innerHTML = showRows.map((rr) =>
      `<a class="row-link" data-source="${escape(detail.source)}" data-row="${rr}" data-column="${escape(detail.column)}">${rr}</a>`
    ).join(", ") + (rows.length > showRows.length ? ` (\u2026 +${rows.length - showRows.length})` : "");

    // Value column.
    const valTd = document.createElement("td");
    valTd.className = "value";
    valTd.textContent = r.value || "";
    valTd.title = r.value || "";

    // Verdict pill.
    const vTd = document.createElement("td");
    vTd.innerHTML = `<span class="audit-verdict-pill verdict-${verdict.replace(/[^A-Za-z0-9_-]/g, "-")}">${escape(verdict)}</span>` +
      (fnCandidate ? ` <span class="fn-flag" title="Confirmed_error on a cell the LLM never flagged \u2014 a likely false negative.">missed!</span>` : "");

    // LLM-flag column.
    const llmTd = document.createElement("td");
    llmTd.className = "llm-flag";
    llmTd.textContent = r.was_flagged_by_llm ? "yes" : "no";

    // Canonical column.
    const canTd = document.createElement("td");
    canTd.className = "canonical";
    canTd.textContent = r.canonical_value || "";

    // Evidence column.
    const evTd = document.createElement("td");
    evTd.className = "evidence";
    const evidence = Array.isArray(r.evidence) ? r.evidence : [];
    if (evidence.length === 0) {
      evTd.innerHTML = `<span class="muted">\u2014</span>`;
    } else {
      evTd.innerHTML = evidence.slice(0, 3).map((ev) => {
        const url = String(ev?.url || "");
        const title = String(ev?.title || url);
        if (!url) return "";
        return `<a href="${escapeAttr(url)}" target="_blank" rel="noopener" title="${escapeAttr(title)}">${escape(shortenUrl(url))}</a>`;
      }).filter(Boolean).join(", ")
        + (evidence.length > 3 ? ` (\u2026 +${evidence.length - 3})` : "");
    }

    // Actions column.
    const actTd = document.createElement("td");
    actTd.className = "audit-row-actions";
    if (fnCandidate && rows.length > 0) {
      const fnBtn = document.createElement("button");
      fnBtn.type = "button";
      fnBtn.className = "audit-fn-btn";
      fnBtn.textContent = "Mark as FN";
      fnBtn.title = `Open the cell viewer at row ${rows[0]} so you can mark this cell as FN. The standard cell-popover handles the actual save.`;
      fnBtn.addEventListener("click", () => jumpToCell(detail.source, rows[0], detail.column));
      actTd.appendChild(fnBtn);
    }
    const jumpBtn = document.createElement("button");
    jumpBtn.type = "button";
    jumpBtn.className = "audit-jump-btn";
    jumpBtn.textContent = "Show in CSV";
    jumpBtn.addEventListener("click", () => jumpToCell(detail.source, rows[0], detail.column));
    actTd.appendChild(jumpBtn);

    tr.appendChild(rowsTd);
    tr.appendChild(valTd);
    tr.appendChild(vTd);
    tr.appendChild(llmTd);
    tr.appendChild(canTd);
    tr.appendChild(evTd);
    tr.appendChild(actTd);
    tbody.appendChild(tr);
  }

  // Wire row-link clicks (after the rows are mounted so the relative
  // hits from delegation pick them all up).
  slot.appendChild(table);
  slot.querySelectorAll(".row-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const src = a.getAttribute("data-source") || "";
      const col = a.getAttribute("data-column") || "";
      const row = parseInt(a.getAttribute("data-row") || "", 10);
      if (!Number.isNaN(row)) jumpToCell(src, row, col);
    });
  });
}

function shortenUrl(url) {
  /* Cosmetic URL shortener for the evidence column: strips scheme +
   * any www. prefix and caps to 50 chars. The full URL stays in the
   * link's href + title so the operator can inspect it. */
  let s = String(url);
  s = s.replace(/^https?:\/\//, "").replace(/^www\./, "");
  if (s.length > 50) s = s.slice(0, 47) + "\u2026";
  return s;
}

function escapeAttr(s) {
  /* HTML-attribute escape: same rules as escape() but also kills the
   * single quote so attribute values quoted with either ' or " stay
   * safe. */
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function jumpToCell(source, row, column) {
  /* Navigate the CSV viewer to (source, row, column). The cell viewer
   * is normally driven by ``renderCsv(cluster)`` (which expects a
   * cluster object with at least ``source`` + ``row`` + ``columns``),
   * so we build a synthetic 1-cell cluster anchored on the audit
   * finding and feed it directly to ``renderCsv``. The operator can
   * then click the highlighted cell to open the cell-popover and
   * mark it FN. */
  if (!source || !Number.isInteger(row)) return;
  // Set the source filter so other parts of the UI (queue, exclusions
  // bar, xlsx pills) follow the operator's intent. Don't await
  // ``refreshQueue`` - the CSV viewer renders independently.
  const ctrl = _multiFilters.get(document.getElementById("filter-source"));
  if (ctrl) ctrl.setSelected([source]);
  state.filters.source = [source];
  updateSourceXlsxActions();
  refreshQueue({ keepSelection: false });
  // Synthetic cluster - just enough fields for renderCsv to anchor.
  const synth = {
    cluster_id: `__audit_jump__${source}__${row}__${column}`,
    source,
    row,
    columns: column ? [column] : [],
    detections: [],
    error_types: [],
  };
  if (typeof renderCsv === "function") {
    try { renderCsv(synth); } catch (e) { console.warn("renderCsv jump failed:", e); }
  }
}

function setupColumnAuditBulkButton() {
  /* One-time wire-up of the "Audit all budget-capped columns" button
   * in the effort panel summary. Triggered once per page load (in
   * bootstrap, after refreshLabelingEffort has had a chance to set
   * its visibility). */
  const btn = document.getElementById("audit-all-capped-btn");
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  // Stop the click from also toggling the parent <details>
  // open/closed state - the operator just wants to launch the job.
  btn.addEventListener("click", async (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    if (!state.run) return;
    // Belt-and-braces: even though applyBulkAuditButtonState
    // disables the button while a job is running, double-clicks
    // and recovered jobs racing the click could still slip through.
    if (state._bulkAuditJob
        && (state._bulkAuditJob.status === "running"
            || state._bulkAuditJob.status === "queued")) {
      toast(`bulk audit ${state._bulkAuditJob.jobId} already running`);
      return;
    }
    btn.disabled = true;
    btn.textContent = "starting\u2026";
    try {
      const body = {
        kind: "column_verify",
        run_id: state.run.run_id,
        params: {
          all_budget_capped: true,
          // Bulk mode shares the per-column concurrency budget; the
          // script iterates columns sequentially and each column's
          // distinct values run with this concurrency internally.
          concurrency: 4,
        },
      };
      const job = await api(`/api/launcher/jobs`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      // Track the bulk job under a sentinel key so the badge code
      // doesn't try to refresh a single-pair detail when it returns.
      state._bulkAuditJob = {
        jobId: job.job_id,
        status: job.status || "running",
        startedAt: Date.now(),
      };
      toast(`bulk audit launched (job ${job.job_id})`);
      // Lock the button into its "Auditing…" state and flip every
      // audit-capped row to the running view INSTANTLY, not after
      // the 1.5 s progress poll cadence catches up.
      applyBulkAuditButtonState();
      refreshAllColumnAuditControlStates();
      pollBulkColumnAudit();
      ensureAuditProgressPoller();
    } catch (err) {
      // Launch failed - clear our optimistic state (if any) and
      // restore the idle label.
      toast(`bulk audit launch failed: ${err.message || err}`);
      applyBulkAuditButtonState();
    }
    // No `finally { btn.disabled = false }` here: when the launch
    // succeeds, the button must STAY disabled until the bulk job
    // reaches a terminal status (handled by pollBulkColumnAudit).
  });
}

async function pollBulkColumnAudit() {
  /* Poll the bulk job until it finishes, then refresh the audit
   * index so all newly-audited columns get cached counts on their
   * badges + show up in the panel. */
  const job = state._bulkAuditJob;
  if (!job) return;
  const POLL_MS = 3000;
  const MAX_POLLS = 1200; // ~1h ceiling
  for (let i = 0; i < MAX_POLLS; i += 1) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    let fresh;
    try {
      fresh = await api(`/api/launcher/jobs/${encodeURIComponent(job.jobId)}`);
    } catch (err) {
      console.warn(`bulk-audit job ${job.jobId} poll failed:`, err);
      continue;
    }
    // Reflect intermediate status changes too so the row painter
    // sees ``running -> cancelled`` immediately rather than waiting
    // for the next progress-poll tick.
    if (state._bulkAuditJob) {
      state._bulkAuditJob.status = fresh.status || state._bulkAuditJob.status;
    }
    if (fresh.status === "running" || fresh.status === "queued") continue;
    // Terminal: refresh the index so all per-column badges get the
    // freshly-written counts.
    if (fresh.status === "succeeded") {
      await refreshColumnAuditIndex();
      toast(`bulk audit done`);
    } else {
      toast(`bulk audit ${fresh.status} \u2014 see launcher log`);
    }
    state._bulkAuditJob = null;
    // Switch every row whose progress view was driven by this bulk
    // job back to the idle (button + badge) view, and unlock the
    // "Audit all budget-capped" button.
    refreshAllColumnAuditControlStates();
    applyBulkAuditButtonState();
    return;
  }
}

// =====================================================================
// Live audit progress bar
// =====================================================================
//
// One global poller for the run's column_verifications/_progress.json
// artifact. The script atomically rewrites that file on every column
// transition + value-batch checkpoint so we get smooth, real-time
// progress without having to parse log files. The poller runs as
// long as ANY column-audit job is in-flight (per-column or bulk) and
// hides the bar once the script writes ``"done": true``.
//
// State:
//   state._auditProgressTimer  - active setTimeout id (or null)
//   state._auditProgressLastJobId - most recent in-flight job id we
//                                   noticed; used to (re)start polling
//                                   when a new job begins.

function _anyAuditJobRunning() {
  if (state._bulkAuditJob && state._bulkAuditJob.status !== "succeeded"
      && state._bulkAuditJob.status !== "failed"
      && state._bulkAuditJob.status !== "cancelled") {
    return state._bulkAuditJob.jobId;
  }
  for (const [, job] of state.columnAuditJobs) {
    if (!job) continue;
    if (job.status === "running" || job.status === "queued") {
      return job.jobId;
    }
  }
  return null;
}

function ensureAuditProgressPoller() {
  /* Idempotent: if a poller is already running, do nothing.
   * Otherwise kick off the recurring fetch. The poller self-cancels
   * once no audit job is in-flight AND the artifact reports done. */
  if (state._auditProgressTimer) return;
  pollAuditProgressOnce();
}

async function pollAuditProgressOnce() {
  state._auditProgressTimer = null;
  if (!state.run) return;
  const POLL_MS = 1500;
  let payload;
  try {
    payload = await api(
      `/api/runs/${encodeURIComponent(state.run.run_id)}`
      + `/column-verifications/progress`,
    );
  } catch (err) {
    console.warn("column-audit progress poll failed:", err);
    payload = null;
  }
  // Cache the latest payload so per-row mini progress bars can read
  // it synchronously (without their own fetch).
  state._auditProgressLastPayload = payload || null;
  renderAuditProgress(payload);
  // Refresh per-row controls so any row whose audit job is in-flight
  // shows its progress fill, and rows whose job just finished switch
  // back to the badge view without waiting for the next render.
  refreshAllColumnAuditControlStates();
  // Continue polling whenever EITHER a job is in-flight OR the
  // artifact says we're not done yet (covers the brief window where
  // the job has terminated but the artifact's final flush is still
  // arriving). When both conditions are clear, hibernate.
  const running = _anyAuditJobRunning();
  const inProgress = payload && payload.available && !payload.done;
  if (running || inProgress) {
    state._auditProgressTimer = setTimeout(pollAuditProgressOnce, POLL_MS);
  } else {
    // Hide the bar; index/detail refresh has already been triggered
    // by the per-job poller.
    const el = document.getElementById("audit-progress");
    if (el) el.hidden = true;
  }
}

function renderAuditProgress(payload) {
  const el = document.getElementById("audit-progress");
  if (!el) return;
  const txt = el.querySelector(".audit-progress-text");
  const fill = el.querySelector(".audit-progress-fill");
  const cancelBtn = el.querySelector("#audit-progress-cancel");
  const running = _anyAuditJobRunning();
  // Hide when nothing is in-flight AND the artifact isn't surfacing
  // an unfinished run. The poller re-shows it on the next tick.
  if (!running && !(payload && payload.available && !payload.done)) {
    el.hidden = true;
    return;
  }
  if (!payload || !payload.available) {
    // Job has been launched but the artifact hasn't been written yet
    // (very early in the run, before targets are resolved). Show an
    // indeterminate state so the operator gets immediate feedback.
    el.hidden = false;
    if (txt) txt.textContent = "starting audit\u2026";
    if (fill) {
      fill.style.width = "0%";
      fill.classList.add("indeterminate");
    }
    if (cancelBtn) cancelBtn.hidden = !running;
    return;
  }
  el.hidden = false;
  if (fill) fill.classList.remove("indeterminate");
  const totalCols = Number(payload.total_columns || 0);
  // 1-based column position; 0 means "haven't entered any column yet".
  const curCol = Number(payload.current_column_index || 0);
  const colsDone = Number(payload.columns_done || 0);
  const valTotal = Number(payload.values_total || 0);
  const valDone = Number(payload.values_done || 0);
  const overall = payload.overall || {};
  const fnSoFar = Number(overall.fn_candidates || 0);
  // Estimate overall fraction across the whole run as
  //   (completed columns + current column's value-fraction) / total columns
  // so the bar moves smoothly even on bulk runs with many columns.
  let frac = 0;
  if (totalCols > 0) {
    const colFrac = (valTotal > 0) ? (valDone / valTotal) : 0;
    frac = (colsDone + colFrac) / totalCols;
  } else if (valTotal > 0) {
    frac = valDone / valTotal;
  }
  if (!Number.isFinite(frac)) frac = 0;
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  if (payload.done) frac = 1;
  if (fill) fill.style.width = `${(frac * 100).toFixed(1)}%`;
  // Headline text. Prefer the most informative slot we have:
  //   - bulk (totalCols > 1):  "auditing 3/43: src.col - 20/77 values - 2 FN so far"
  //   - single (totalCols=1):  "src.col - 20/77 values - 0 FN so far"
  //   - done flag:             "done - 5 FN found across N columns"
  if (txt) {
    if (payload.done) {
      const totalRun = Number(overall.total || 0);
      txt.textContent = `done \u2014 ${fnSoFar} FN candidate${fnSoFar === 1 ? "" : "s"}`
        + ` across ${totalRun} verified value${totalRun === 1 ? "" : "s"}`;
    } else {
      const src = payload.current_source || "";
      const col = payload.current_column || "";
      const where = (src || col) ? `${src} \u00B7 ${col}` : "preparing\u2026";
      const valStr = (valTotal > 0)
        ? `${valDone}/${valTotal} value${valTotal === 1 ? "" : "s"}`
        : "";
      const colStr = (totalCols > 1)
        ? `column ${Math.max(curCol, colsDone + (valTotal > 0 ? 1 : 0))}/${totalCols}`
        : "";
      const fnStr = (fnSoFar > 0) ? `${fnSoFar} FN so far` : "";
      txt.textContent = [colStr, where, valStr, fnStr]
        .filter((s) => !!s)
        .join(" \u00B7 ");
    }
  }
  if (cancelBtn) cancelBtn.hidden = !running;
}

function setupAuditProgressCancel() {
  /* Wire the inline "cancel" button next to the progress bar.
   * Calls /jobs/<id>/cancel for whichever audit job is currently
   * in-flight. */
  const btn = document.getElementById("audit-progress-cancel");
  if (!btn || btn.dataset.wired === "1") return;
  btn.dataset.wired = "1";
  btn.addEventListener("click", async (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    const jid = _anyAuditJobRunning();
    if (!jid) {
      toast("no audit job is running");
      return;
    }
    btn.disabled = true;
    try {
      await api(`/api/launcher/jobs/${encodeURIComponent(jid)}/cancel`, {
        method: "POST",
      });
      toast(`cancelled audit job ${jid}`);
    } catch (err) {
      toast(`cancel failed: ${err.message || err}`);
    } finally {
      btn.disabled = false;
    }
  });
}

// Workbook-level xlsx access. When the source filter is set to a
// specific CSV (or to a synthetic ``workbook:<wb>`` group) and the
// run advertises an original .xlsx for that workbook, render
// ``⬇ xlsx`` + ``📄 preview`` pills next to the source dropdown so
// the operator can grab the original workbook without having to
// open a cluster first. Hidden when the selection has no resolvable
// workbook (e.g. ``(all)`` across a multi-workbook run, or a CSV
// from a corpus with no sibling _xlsx dir).
function updateSourceXlsxActions() {
  const el = document.getElementById("filter-source-xlsx");
  if (!el || !state.run) return;
  // Only show the xlsx download/preview pills when the source filter
  // resolves to a single workbook. If the operator picked multiple
  // CSVs / multiple workbooks (or none), the pills don't have a
  // single workbook to point at, so they're hidden.
  const sel = state.filters.source || [];
  let wb = "";
  let hasXlsx = false;
  if (sel.length === 1) {
    const v = sel[0];
    if (v.startsWith("workbook:")) {
      const name = v.slice("workbook:".length);
      const meta = (state.run.workbooks || []).find((w) => w.name === name);
      if (meta) { wb = meta.name; hasXlsx = !!meta.has_xlsx; }
    } else {
      const meta = (state.run.sources || []).find((s) => s.name === v);
      if (meta) {
        wb = meta.table_group || v;
        hasXlsx = !!meta.has_xlsx;
      }
    }
  }
  if (!wb || !hasXlsx) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML =
    `<a class="xlsx-btn" href="${API}/workbooks/${encodeURIComponent(wb)}/xlsx" download="${escape(wb)}" title="Download original .xlsx for ${escape(wb)}">⬇ xlsx</a>` +
    `<a class="xlsx-btn" href="${API}/workbooks/${encodeURIComponent(wb)}/preview" target="_blank" rel="noopener" title="Preview original workbook ${escape(wb)} in new tab">📄 preview</a>`;
}

// Pull fresh per-source coverage counts from the server and update the
// dropdown option text for every source. Cheap O(N) call - used after
// label saves to keep the checked/FN counters in sync without
// tracking deltas locally.
async function refreshSourceMeta() {
  if (!state.run) return;
  try {
    const fresh = await api(`${API}`);
    state.run.sources = fresh.sources;
    const ctrl = _multiFilters.get(document.getElementById("filter-source"));
    if (!ctrl) return;
    const byValue = new Map();
    for (const meta of fresh.sources) {
      const mark = meta.checked ? "\u2713" : "\u25CB";
      const cnt = meta.clusters > 0 ? `(${meta.clusters})` : "(\u2205)";
      const fn = meta.fn_cells ? ` \u00B7 fn:${meta.fn_cells}` : "";
      byValue.set(meta.name, {
        label: `${mark} ${meta.name} ${cnt}${fn}`,
        title: meta.name,
      });
    }
    ctrl.refreshLabels(byValue);
  } catch (_) { /* non-fatal */ }
}

// ----------------------------------------------------------- save-path helpers
//
// On a busy operator (Enter/1-5 on every cluster), the awaited save
// pipeline used to fan out to /labels, /labels/propagate, /sources/
// .../cells, /stats and / (run meta) on every keystroke. Each of the
// last three iterates 22k clusters server-side, and they pile up on
// the single uvicorn worker. Two helpers below cut that fan-out to
// at most one in-flight call each:
//
//   * ``scheduleStatsRefresh`` / ``scheduleSourceMetaRefresh`` debounce
//     the dashboard pills + source dropdown to one call per ~250 ms.
//   * ``bumpCellStateLocal`` mirrors the server-side rank precedence
//     (tp > partial > fp > fn > unsure) so we can repaint the cell
//     viewer immediately after a save without awaiting a /cells round
//     trip. A background ``loadCellState({force:true})`` then
//     reconciles any edge cases (FN shadowing, propagated siblings).

const _debouncers = new Map();
function _debounce(key, fn, ms) {
  const prev = _debouncers.get(key);
  if (prev) clearTimeout(prev);
  _debouncers.set(key, setTimeout(() => {
    _debouncers.delete(key);
    try { fn(); } catch (e) { console.warn(`debounced ${key} failed:`, e); }
  }, ms));
}
function scheduleStatsRefresh() { _debounce("stats", refreshStats, 250); }
function scheduleSourceMetaRefresh() { _debounce("source-meta", refreshSourceMeta, 350); }

// Server-side rank: cells.py's ``rank`` dict.
const _CELL_STATE_RANK = { tp: 5, partial: 4, fp: 3, fn: 2, unsure: 1 };

function bumpCellStateLocal(source, row, col, status, extras = {}) {
  if (!source || !Number.isInteger(row) || !col || !status) return;
  let map = state.cellState.get(source);
  if (!map) {
    map = new Map();
    state.cellState.set(source, map);
  }
  const key = `${row}\x00${col}`;
  const cur = map.get(key);
  const newRank = _CELL_STATE_RANK[status] || 0;
  const curRank = cur ? (_CELL_STATE_RANK[cur.state] || 0) : 0;
  const ce = cur ? { ...cur } : { row, column: col };
  // Track that the (cluster_id) labelled this cell, even when the
  // existing higher-rank state wins.
  if (extras.cluster_ids) {
    ce.cluster_ids = Array.from(new Set([
      ...(ce.cluster_ids || []),
      ...extras.cluster_ids,
    ]));
  }
  if (newRank >= curRank) {
    ce.state = status;
    if (extras.error_types) ce.error_types = extras.error_types.slice();
    if ("note" in extras) ce.note = extras.note || "";
    if ("annotator" in extras) ce.annotator = extras.annotator || "";
    if ("label_source" in extras) ce.label_source = extras.label_source;
    if ("corrected_value" in extras) ce.corrected_value = extras.corrected_value;
    // FN-shadowing: a TP/PARTIAL cluster verdict on a cell with a
    // pre-existing FN cell label "supersedes" the FN. The server
    // re-derives this on next /cells GET; mirror it here so the
    // shadow class paints synchronously.
    if (
      (status === "tp" || status === "partial")
      && cur && cur.label_source === "cell"
      && cur.state === "fn"
    ) {
      ce.superseded_fn_by = (extras.cluster_ids || []).slice(0, 1);
    }
  }
  map.set(key, ce);
}

// ----------------------------------------------------------- queue

async function refreshQueue({ keepSelection }) {
  const qs = new URLSearchParams();
  // Filters are arrays (multi-select). The backend's /clusters
  // endpoint only accepts a single value per facet, so we send a
  // backend-side filter ONLY when exactly one value is picked - this
  // lets large runs benefit from the server-side filter when the
  // operator has narrowed down. With 2+ picks we fetch unfiltered (up
  // to the 5000 cap) and apply the multi-value match client-side.
  const srcFilter = state.filters.source || [];
  const statusFilter = state.filters.status || [];
  const etFilter = state.filters.error_type || [];
  if (srcFilter.length === 1) {
    const v = srcFilter[0];
    if (v.startsWith("workbook:")) {
      qs.set("workbook", v.slice("workbook:".length));
    } else {
      qs.set("source", v);
    }
  }
  if (statusFilter.length === 1) qs.set("status", statusFilter[0]);
  if (etFilter.length === 1) qs.set("error_type", etFilter[0]);
  // Backend cap is 5000; this matches it so the (all)-source view
  // never silently truncates large runs and "unlabeled" counts in the
  // header don't drift away from what's actually in the queue.
  qs.set("limit", "5000");
  const r = await api(`${API}/clusters?${qs.toString()}`);
  let queueItems = r.items.slice();
  if (r.total != null && r.total > r.items.length) {
    toast(`queue truncated: ${r.items.length}/${r.total} (apply a filter to see the rest)`, 4000);
  }

  // Map source filter into "which CSVs are accepted" + "which workbook
  // tokens are accepted" so the client-side check below is O(1) per
  // cluster. Any non-empty selection means we accept ONLY items that
  // match at least one entry (logical OR within a facet).
  const filterWbBySource = {};
  for (const s of state.run?.sources || []) filterWbBySource[s.name] = s.table_group || "";
  const allowedSources = new Set(srcFilter.filter((v) => !v.startsWith("workbook:")));
  const allowedWorkbooks = new Set(
    srcFilter.filter((v) => v.startsWith("workbook:")).map((v) => v.slice("workbook:".length))
  );
  const allowedStatuses = new Set(statusFilter);
  const allowedEts = new Set(etFilter);

  if (srcFilter.length > 1 || statusFilter.length > 1 || etFilter.length > 1) {
    queueItems = queueItems.filter((c) => {
      if (srcFilter.length > 0) {
        const wb = filterWbBySource[c.source || ""] || "";
        const matches = allowedSources.has(c.source || "") || (wb && allowedWorkbooks.has(wb));
        if (!matches) return false;
      }
      if (statusFilter.length > 0) {
        if (!allowedStatuses.has(c.label_status || "unlabeled")) return false;
      }
      if (etFilter.length > 0) {
        const ets = c.error_types || [];
        if (!ets.some((et) => allowedEts.has(et))) return false;
      }
      return true;
    });
  }

  // Merge multi-table cross-sheet findings into the same queue.
  // Filtering is done client-side here (rather than via the
  // multi-table endpoint's query params) so the cluster-vocab
  // ``status`` filter survives the cluster-vs-MT verdict mapping.
  if (state.run?.counts?.multi_table_findings) {
    try {
      const mtRes = await api(`${API}/multi-table-violations?limit=5000`);
      const mtRaw = Array.isArray(mtRes.items) ? mtRes.items : [];
      for (const m of mtRaw) {
        // Reverse-map MT status -> cluster vocab so the dot color
        // and the status filter dropdown stay consistent across
        // the two item kinds.
        const mappedStatus = m.label_status === "unlabeled"
          ? "unlabeled"
          : (MT_TO_CLUSTER_STATUS[m.label_status] || "unsure");
        // Source filter: a finding matches if ANY of its sheets is in
        // the allowed-sources set OR its workbook is in the allowed-
        // workbooks set. With no source filter, the finding always
        // passes the source check.
        if (srcFilter.length > 0) {
          const sheets = m.sheets || [];
          const sheetMatch = sheets.some((s) => allowedSources.has(s));
          const wbMatch = m.workbook && allowedWorkbooks.has(m.workbook);
          if (!sheetMatch && !wbMatch) continue;
        }
        if (statusFilter.length > 0 && !allowedStatuses.has(mappedStatus)) continue;
        // ``inter_table_inconsistency`` is the canonical error_type
        // for MT findings; expose it as the filter facet so the
        // operator can isolate them.
        const et = "inter_table_inconsistency";
        if (etFilter.length > 0 && !allowedEts.has(et)) continue;
        queueItems.push({
          // Cluster-shaped fields the queue/render code reads.
          cluster_id: m.violation_id,
          source: (m.sheets || [])[0] || "",
          row: -1,                              // not meaningful for MT
          columns: m.columns || [],
          error_types: [et],
          label_status: mappedStatus,
          // MT-specific fields the kind-aware code paths read.
          is_mt: true,
          violation_id: m.violation_id,
          workbook: m.workbook,
          group_id: m.group_id,
          group_kind: m.group_kind,
          mt_kind: m.kind,                      // attribute_conflict / ...
          sheets: m.sheets || [],
          n_rows: m.n_rows || 0,
          llm_verdict: m.llm_verdict,
          llm_confidence: m.llm_confidence,
        });
      }
    } catch (e) {
      // Non-fatal: we still show clusters even if the MT endpoint
      // briefly fails (e.g. malformed multi_table_violations.jsonl).
      console.warn("multi-table violations fetch failed:", e);
    }
  }

  // Column-status filter (multi-select). The labelling-effort artifact
  // keys per (source, column), so this is purely client-side: walk
  // every queue item, look up the arm-status set for each of its
  // (source, column) cells, and keep items whose set intersects the
  // operator's selection. Skipped when nothing is selected. Items
  // with no matching cell in the artifact land in the synthetic
  // ``no-estimate`` bucket - that lets the operator either include
  // or exclude them explicitly.
  const colStatusFilter = state.filters.column_status || [];
  if (colStatusFilter.length > 0) {
    const allowedColStatuses = new Set(colStatusFilter);
    queueItems = queueItems.filter((c) => {
      const seen = columnStatusesForItem(c);
      for (const s of seen) if (allowedColStatuses.has(s)) return true;
      return false;
    });
  }

  state.queue = queueItems;

  const listEl = $("#queue");
  listEl.innerHTML = "";
  // Per-source workbook lookup so the queue row can show the parent
  // table when multiple CSVs of the same workbook are interleaved.
  const wbBySource = {};
  if (state.run?.multi_workbook) {
    for (const s of state.run.sources) wbBySource[s.name] = s.table_group || "";
  }

  // Build a 2-level grouping: workbook -> csv -> [{cluster, queueIndex}].
  // Queue items keep their flat ``state.queue`` index so the j/k/Enter
  // and 1-5 shortcuts (which all index ``state.queue`` directly) keep
  // working without any change to the keyboard handlers. We only
  // change how the rows are RENDERED.
  const multiWb = !!state.run?.multi_workbook;
  const groups = new Map(); // wb -> Map(src -> [{c, idx}])
  state.queue.forEach((c, idx) => {
    const src = c.source || "";
    const wb = wbBySource[src] || "";
    if (!groups.has(wb)) groups.set(wb, new Map());
    const bySrc = groups.get(wb);
    if (!bySrc.has(src)) bySrc.set(src, []);
    bySrc.get(src).push({ c, idx });
  });

  // Persist collapsed state across re-renders so labelling a cluster
  // doesn't snap every group back open.
  state.collapsed = state.collapsed || new Set();

  const headerSummary = (items) => {
    const total = items.length;
    let labeled = 0;
    for (const { c } of items) {
      const s = c.label_status;
      if (s && s !== "unlabeled") labeled += 1;
    }
    return `${labeled}/${total}`;
  };

  for (const [wb, bySrc] of groups) {
    let wbBox = listEl;
    if (multiWb && wb) {
      // Workbook-level group (only meaningful for multi-table corpora).
      const wbAll = [];
      for (const items of bySrc.values()) wbAll.push(...items);
      const wbHdr = document.createElement("div");
      wbHdr.className = "group-header wb";
      const wbKey = `wb:${wb}`;
      wbHdr.dataset.wb = wb;
      const collapsedWb = state.collapsed.has(wbKey);
      const caret = `<span class="caret">${collapsedWb ? "\u25B8" : "\u25BE"}</span>`;
      wbHdr.innerHTML =
        caret +
        `<span class="label">${escape(wb)}</span>` +
        `<span class="count">${headerSummary(wbAll)}</span>`;
      listEl.appendChild(wbHdr);
      const wbBody = document.createElement("div");
      wbBody.className = "group-body wb-body";
      if (collapsedWb) wbBody.style.display = "none";
      listEl.appendChild(wbBody);
      wbBox = wbBody;
      wbHdr.addEventListener("click", () => {
        toggleGroup(wbHdr, wbBody, wbKey);
      });
    }

    for (const [src, items] of bySrc) {
      const srcKey = `src:${wb}/${src}`;
      const collapsedSrc = state.collapsed.has(srcKey);
      const srcHdr = document.createElement("div");
      srcHdr.className = "group-header src";
      srcHdr.dataset.wb = wb;
      srcHdr.dataset.src = src;
      // Surface a budget-exhausted warning when ANY cluster in this
      // source had its row_duplication candidate budget capped.
      // Some duplicate clusters may not have been adjudicated, so
      // the operator should consider rerunning with a larger
      // ``max_groups_per_source``.
      let budgetExhausted = false;
      let budgetCap = 0;
      for (const it of items) {
        if (it.c && it.c.cluster_budget_exhausted) {
          budgetExhausted = true;
          if (it.c.max_groups_per_source && it.c.max_groups_per_source > budgetCap) {
            budgetCap = it.c.max_groups_per_source;
          }
        }
      }
      const budgetBadge = budgetExhausted
        ? `<span class="budget-warn-badge" title="Row-duplication candidate budget exhausted${budgetCap ? ` (cap=${budgetCap})` : ""}; some clusters may have been dropped. Rerun with a larger --row-dup-max-groups-per-source.">budget exhausted</span>`
        : "";
      srcHdr.innerHTML =
        `<span class="caret">${collapsedSrc ? "\u25B8" : "\u25BE"}</span>` +
        `<span class="label">${escape(src)}</span>` +
        budgetBadge +
        `<span class="count">${headerSummary(items)}</span>`;
      wbBox.appendChild(srcHdr);
      const srcBody = document.createElement("div");
      srcBody.className = "group-body src-body";
      if (collapsedSrc) srcBody.style.display = "none";
      wbBox.appendChild(srcBody);
      srcHdr.addEventListener("click", () => {
        toggleGroup(srcHdr, srcBody, srcKey);
      });

      // Within a CSV, row_duplication clusters are bucketed by their
      // duplicate-group key (sorted ``affected_rows``) so the operator
      // sees ONE "Duplicate group: rows 5, 12, 27 (×3)" sub-header per
      // candidate group instead of N separate cluster rows (one per
      // member). Non-row_duplication clusters render flat, just like
      // before.
      const dupBuckets = new Map(); // key -> [{c, idx}]
      const flatItems = [];
      for (const entry of items) {
        const c = entry.c;
        const dupRows = Array.isArray(c.affected_rows) ? c.affected_rows : [];
        const isDupCluster = (c.error_types || []).includes("row_duplication") && dupRows.length >= 1;
        if (!isDupCluster) {
          flatItems.push(entry);
          continue;
        }
        const key = dupRows.slice().sort((a, b) => a - b).join(",");
        if (!dupBuckets.has(key)) dupBuckets.set(key, []);
        dupBuckets.get(key).push(entry);
      }

      const renderQueueRow = (entry) => {
        const { c, idx } = entry;
        const row = document.createElement("div");
        row.className = "row" + (c.inferred ? " row-inferred" : "")
          + (isMtItem(c) ? " row-mt" : "");
        row.dataset.queueIndex = String(idx);
        const status = c.label_status || "unlabeled";
        const inferredBadge = c.inferred
          ? `<span class="inferred-badge" title="No primary detection on this row \u2014 surfaced via a sibling detection's affected_rows. Labelling it captures the LLM's full claim and prevents FN drift.">inf</span>`
          : "";
        const dupRows = Array.isArray(c.affected_rows) ? c.affected_rows : [];
        const isDup = (c.error_types || []).includes("row_duplication") && dupRows.length > 1;
        const dupBadge = isDup
          ? `<span class="dup-badge" title="row_duplication group spans ${dupRows.length} rows">×${dupRows.length}</span>`
          : "";
        // Multi-table findings span 2+ sheets; show a "cross-sheet"
        // badge plus the linked sheets so the operator can spot the
        // item kind without expanding the row. The "row" line is
        // omitted because no single row is canonical.
        const mtBadge = isMtItem(c)
          ? `<span class="dup-badge" title="cross-sheet finding (${(c.sheets || []).join(' \u2194 ')})">cross-sheet</span>`
          : "";
        // Inside a dup-bucket the row line shows just THIS cluster's
        // canonical row (the sibling rows are already named in the
        // sub-header above).
        const rowLine = isMtItem(c)
          ? escape((c.sheets || []).join(" \u2194 "))
          : `row ${escape(String(c.row ?? "?"))}`;
        row.innerHTML = `
          <div class="dot ${status}"></div>
          <div class="main">
            <div class="src">${rowLine} ${dupBadge}${mtBadge}${inferredBadge}</div>
            <div class="cols">${escape((c.columns || []).join(", "))}</div>
            <div class="types">${escape((c.error_types || []).join(" \u00B7 "))}</div>
          </div>
        `;
        row.addEventListener("click", () => selectIndex(idx));
        return row;
      };

      // Flat (non-dup) clusters first, in queue order.
      for (const entry of flatItems) srcBody.appendChild(renderQueueRow(entry));

      // Dup buckets, sorted by smallest member row so groups are stable.
      const sortedBuckets = [...dupBuckets.entries()].sort((a, b) => {
        const ra = Number((a[0].split(",")[0] || "").trim());
        const rb = Number((b[0].split(",")[0] || "").trim());
        return (Number.isFinite(ra) ? ra : 0) - (Number.isFinite(rb) ? rb : 0);
      });
      for (const [key, bucket] of sortedBuckets) {
        const dupKey = `dup:${wb}/${src}/${key}`;
        const collapsedDup = state.collapsed.has(dupKey);
        const dupHdr = document.createElement("div");
        const sample = bucket[0]?.c || {};
        // Bucket-level verdict: prefer the strongest verdict in
        // any cluster in the bucket. ``duplicate`` > ``near_duplicate``
        // > <unset>. Used to decorate the header amber for confirmed
        // duplicates and orange for "needs review" near-duplicates.
        const verdictRank = { duplicate: 2, near_duplicate: 1 };
        let bucketVerdict = "";
        let bestRank = -1;
        const strategies = new Set();
        for (const { c } of bucket) {
          const v = String(c.verdict || "").toLowerCase();
          if (verdictRank[v] != null && verdictRank[v] > bestRank) {
            bestRank = verdictRank[v];
            bucketVerdict = v;
          }
          for (const s of c.discovery_strategies || []) strategies.add(s);
        }
        dupHdr.className = "group-header dup" + (
          bucketVerdict === "near_duplicate" ? " near-dup" : ""
        );
        const dupRows = Array.isArray(sample.affected_rows) ? sample.affected_rows : [];
        const previewRows = dupRows.slice(0, 6);
        const moreCount = dupRows.length - previewRows.length;
        const groupNoun = bucketVerdict === "near_duplicate"
          ? "Near-duplicate group"
          : "Duplicate group";
        const labelText = dupRows.length
          ? `${groupNoun} \u2014 rows ${previewRows.join(", ")}${moreCount > 0 ? `, +${moreCount}` : ""}`
          : `${groupNoun} \u2014 ${bucket.length} cluster(s)`;
        // Bucket-level summary: how many of the sibling clusters in
        // this group are still unlabeled.
        let bucketLabeled = 0;
        for (const { c } of bucket) {
          if (c.label_status && c.label_status !== "unlabeled") bucketLabeled += 1;
        }
        const strategyChips = strategies.size
          ? `<span class="strategy-chips" title="Discovered by: ${[...strategies].join(", ")}">${
              [...strategies].map((s) =>
                `<span class="strategy-chip strategy-${escape(s)}">${escape(s)}</span>`
              ).join("")
            }</span>`
          : "";
        const verdictBadge = bucketVerdict === "near_duplicate"
          ? `<span class="verdict-badge near-dup-badge" title="LLM marked as near-duplicate; needs human review">near-dup</span>`
          : "";
        dupHdr.innerHTML =
          `<span class="caret">${collapsedDup ? "\u25B8" : "\u25BE"}</span>` +
          `<span class="label">${escape(labelText)}</span>` +
          verdictBadge +
          strategyChips +
          `<span class="dup-badge" title="row_duplication group spans ${dupRows.length || bucket.length} rows">\u00d7${dupRows.length || bucket.length}</span>` +
          `<span class="count">${bucketLabeled}/${bucket.length}</span>`;
        srcBody.appendChild(dupHdr);
        const dupBody = document.createElement("div");
        dupBody.className = "group-body dup-body";
        if (collapsedDup) dupBody.style.display = "none";
        srcBody.appendChild(dupBody);
        dupHdr.addEventListener("click", () => {
          toggleGroup(dupHdr, dupBody, dupKey);
        });
        for (const entry of bucket) dupBody.appendChild(renderQueueRow(entry));
      }
    }
  }

  // Strikethrough queue rows whose cluster is on an excluded slice.
  paintQueueExclusions();

  if (keepSelection && state.selectedId) {
    const i = state.queue.findIndex((c) => c.cluster_id === state.selectedId);
    if (i >= 0) return selectIndex(i);
  }
  if (state.queue.length) {
    selectIndex(0);
    return;
  }
  // Empty queue: if the source filter targets a single CSV (not a
  // workbook-wide selector), drop into "browse mode" so the operator
  // can scroll through any table - including ones the LLM never
  // flagged - and mark FN cells. This is the headline use case for
  // CSVs that received zero LLM clusters.
  const src = state.filters.source || [];
  const onlyCsv = src.length === 1 && !src[0].startsWith("workbook:") ? src[0] : "";
  if (onlyCsv) {
    enterBrowseMode(onlyCsv);
  } else {
    renderDetail(null);
    state.cluster = null;
    state.selectedId = null;
  }
}

function enterBrowseMode(source) {
  state.cluster = null;
  state.selectedId = null;
  state.index = -1;
  renderDetailBrowse(source);
  // Synthesise a cluster shape so renderCsv can reuse its scroll/paint
  // code path. row=0, no flagged columns - the operator clicks any
  // cell to mark FN.
  renderCsv({ source, row: 0, columns: [] });
}

function renderDetailBrowse(source) {
  const el = $("#detail");
  const meta = state.run?.sources?.find((s) => s.name === source);
  const wb = state.run?.multi_workbook ? (meta?.table_group || "") : "";
  const head = wb ? `${wb} / ${source}` : source;
  const checked = !!meta?.checked;
  const fn = meta?.fn_cells || 0;
  const cl = meta?.clusters || 0;
  el.innerHTML = `
    <div class="block">
      <h2>Browse: ${escape(head)}</h2>
      <div class="cols">${cl} LLM clusters \u00B7 ${fn} FN cells \u00B7 ${checked ? "checked" : "not yet checked"}</div>
      <div class="placeholder" style="margin-top:8px;">
        ${cl === 0
          ? "The LLM did not flag any cells in this table. Browse the rows on the right and click any cell to mark it as a false negative (FN)."
          : "All LLM clusters for this CSV are filtered out (or already labelled). You can still click any cell to mark FN."}
      </div>
    </div>`;
}

function selectIndex(i) {
  if (i < 0 || i >= state.queue.length) return;
  state.index = i;
  const c = state.queue[i];
  state.selectedId = c.cluster_id;
  for (const el of document.querySelectorAll("#queue .row")) el.classList.remove("active");
  // Rows live inside collapsible group bodies, so positional indexing
  // no longer works. Look up by the ``data-queue-index`` attribute we
  // stamp at render time.
  const row = document.querySelector(`#queue .row[data-queue-index="${i}"]`);
  if (row) {
    row.classList.add("active");
    row.scrollIntoView({ block: "nearest" });
  }
  if (isMtItem(c)) {
    loadMtViolation(c);
  } else {
    loadCluster(c.cluster_id);
  }
}

// Toggle a queue group's collapsed state in place. Rebuilding the
// whole queue would (a) round-trip to the API and (b) wipe the user's
// scroll position, so we just flip ``display`` on the body and update
// the caret + persistent ``state.collapsed`` set.
function toggleGroup(headerEl, bodyEl, key) {
  const willCollapse = bodyEl.style.display !== "none";
  bodyEl.style.display = willCollapse ? "none" : "";
  if (willCollapse) state.collapsed.add(key);
  else state.collapsed.delete(key);
  const caret = headerEl.querySelector(".caret");
  if (caret) caret.textContent = willCollapse ? "\u25B8" : "\u25BE";
}

// Apply propagated cluster verdicts to the queue: update each row's
// ``label_status`` and repaint the dot so propagated siblings reflect
// the new status without a full reload. Used after FP/Partial cluster
// propagation so the operator sees the cascading verdict in the
// sidebar immediately.
function applyPropagatedClustersToQueue(propagatedClusters, status) {
  if (!Array.isArray(propagatedClusters) || !propagatedClusters.length) return;
  const ids = new Set(
    propagatedClusters.map((p) => p && p.cluster_id).filter(Boolean),
  );
  if (!ids.size) return;
  for (let i = 0; i < state.queue.length; i++) {
    const q = state.queue[i];
    if (!q || !ids.has(q.cluster_id)) continue;
    q.label_status = status;
    const row = document.querySelector(`#queue .row[data-queue-index="${i}"]`);
    if (row) {
      const dot = row.querySelector(".dot");
      if (dot) dot.className = "dot " + status;
    }
  }
}

// Re-tally ``unlabeled / total`` for every group header from the
// current ``state.queue``. Called after a label save so the counts
// next to each csv/workbook stay in sync without a full re-render.
function updateGroupCounts() {
  const wbBySource = {};
  if (state.run?.multi_workbook) {
    for (const s of state.run.sources) wbBySource[s.name] = s.table_group || "";
  }
  // Tally per (wb, src) and per wb.
  const perSrc = new Map(); // `${wb}\x00${src}` -> {total, labeled}
  const perWb = new Map(); // wb -> {total, labeled}
  for (const c of state.queue) {
    const src = c.source || "";
    const wb = wbBySource[src] || "";
    const isLabeled = !!c.label_status && c.label_status !== "unlabeled";
    const sKey = `${wb}\x00${src}`;
    const sBucket = perSrc.get(sKey) || { total: 0, labeled: 0 };
    sBucket.total += 1;
    if (isLabeled) sBucket.labeled += 1;
    perSrc.set(sKey, sBucket);
    const wBucket = perWb.get(wb) || { total: 0, labeled: 0 };
    wBucket.total += 1;
    if (isLabeled) wBucket.labeled += 1;
    perWb.set(wb, wBucket);
  }
  for (const hdr of document.querySelectorAll("#queue .group-header.src")) {
    const sKey = `${hdr.dataset.wb || ""}\x00${hdr.dataset.src || ""}`;
    const b = perSrc.get(sKey);
    const cnt = hdr.querySelector(".count");
    if (b && cnt) cnt.textContent = `${b.labeled}/${b.total}`;
  }
  for (const hdr of document.querySelectorAll("#queue .group-header.wb")) {
    const b = perWb.get(hdr.dataset.wb || "");
    const cnt = hdr.querySelector(".count");
    if (b && cnt) cnt.textContent = `${b.labeled}/${b.total}`;
  }
}

// Find the next queue item that has no verdict yet. Used by Enter and
// ``j`` so the operator only stops on items that still need a decision;
// already-labeled siblings (incl. auto-skipped row members) are passed
// over. If nothing ahead is unlabeled, wrap around to the start of the
// queue so the operator can sweep up earlier holes (e.g. clusters they
// previously left as ``unsure``) without manually scrolling. If there
// are no unlabeled items at all, fall back to the immediate next index
// so end-of-queue still moves the cursor instead of locking.
function nextUnlabeledIndex(from) {
  const n = state.queue.length;
  for (let i = from; i < n; i++) {
    const s = state.queue[i].label_status;
    if (!s || s === "unlabeled") return i;
  }
  for (let i = 0; i < Math.min(from, n); i++) {
    const s = state.queue[i].label_status;
    if (!s || s === "unlabeled") {
      toast("wrapped: jumped to first unlabeled");
      return i;
    }
  }
  return Math.min(from, n - 1);
}

// ----------------------------------------------------------- detail + CSV

// Center pane has two modes: the classic CSV grid (cluster items) and
// the side-by-side row panels (multi-table items). Switching between
// them is just a matter of toggling ``hidden`` on the two containers.
function showCsvPane() {
  const csv = $("#csv-viewer"); if (csv) csv.hidden = false;
  const mt = $("#mt-compare"); if (mt) mt.hidden = true;
}
function showMtPane() {
  const csv = $("#csv-viewer"); if (csv) csv.hidden = true;
  const mt = $("#mt-compare"); if (mt) mt.hidden = false;
}

async function loadCluster(cid) {
  showCsvPane();
  const [r] = await Promise.all([api(`${API}/clusters/${encodeURIComponent(cid)}`)]);
  state.cluster = r;
  // Render the CSV BEFORE the detail panel so renderCsv's slice fetch
  // populates ``state.csvCache``. ``computeBeforeAfter`` (called from
  // renderDetail) reads the literal cell value out of that cache,
  // and previously this ran before the cache was populated - causing
  // the "Original" line to render empty on first navigation.
  await renderCsv(r.cluster);
  renderDetail(r);
}

// Load a multi-table finding by violation_id and render the side-by-
// side comparison + a cluster-shaped detail pane. The fetched detail
// is stashed under ``state.cluster`` (with a ``_mt`` marker) so the
// existing ``submitLabel`` path can branch off it.
async function loadMtViolation(item) {
  showMtPane();
  state.cluster = null;
  $("#csv-source").textContent = item.workbook
    ? `${item.workbook} · cross-sheet`
    : "(cross-sheet)";
  $("#csv-pos").textContent = `${state.index + 1} / ${state.queue.length}`;
  $("#mt-compare").innerHTML = `<div class="placeholder">loading…</div>`;
  let detail;
  try {
    detail = await api(
      `${API}/multi-table-violations/${encodeURIComponent(item.violation_id)}`
    );
  } catch (e) {
    $("#mt-compare").innerHTML =
      `<div class="placeholder" style="color:var(--sev-high);">failed: ${escape(e.message)}</div>`;
    state.cluster = null;
    return;
  }
  state.cluster = { _mt: true, mt: detail, queueItem: item };
  renderMtCompare(detail);
  renderMtDetail(detail, item);
}

// Build the per-column Original / Fix surface for a cluster. Returns
// a list of ``{column, before, after, changed}`` entries (one per
// column the cluster touches).
//
// ``before`` is the literal cell value at (source, target row, column)
// pulled from the live CSV slice cache. We deliberately never fall
// back to ``detection.observed`` - that field often holds an LLM
// narrative ("minority suffix pattern: '... Reg' (e.g., 'Andover
// Reg', ...) vs other rows using '... Regional' ...") rather than the
// cell's raw content. If the slice isn't cached yet we leave
// ``before`` empty; renderCsv populates the cache before renderDetail
// on the standard navigation path, so the empty state is rare.
//
// ``after`` prefers the per-detection ``suggested_fix``, then the
// cluster's synthesis ``canonical_fix`` (only used for single-column
// clusters where it's unambiguous). The fix may be a literal value or
// a free-form instruction; we render it verbatim.
function computeBeforeAfter(cluster) {
  const cols = (cluster.columns || []).filter(Boolean);
  if (!cols.length) return [];
  const dets = cluster.detections || [];
  // First detection that names this column is canonical for "after".
  const detsByCol = new Map();
  for (const d of dets) {
    const dcols = (d.columns && d.columns.length) ? d.columns
      : (d.column ? [d.column] : []);
    for (const col of dcols) {
      if (!detsByCol.has(col)) detsByCol.set(col, d);
    }
  }
  // Pull the literal cell value out of the CSV slice cache. Scans
  // every cached slice for the source so we still hit when the
  // operator just navigated to a new cluster on a different row of
  // the same CSV.
  const liveValue = (col) => {
    const src = cluster.source;
    const tgt = cluster.row;
    if (!src || !Number.isInteger(tgt)) return "";
    for (const [key, data] of state.csvCache.entries()) {
      if (!key.startsWith(`${src}:`)) continue;
      const r = (data.rows || []).find((x) => x.row === tgt && x.values);
      if (r && Object.prototype.hasOwnProperty.call(r.values, col)) {
        return String(r.values[col] ?? "");
      }
    }
    return "";
  };
  const synFix = (cluster.synthesis?.canonical_fix || "").toString().trim();
  const out = [];
  for (const col of cols) {
    const det = detsByCol.get(col);
    const before = liveValue(col);
    let after = "";
    if (det && det.suggested_fix != null && det.suggested_fix !== "") {
      after = String(det.suggested_fix);
    } else if (synFix && synFix !== "null" && cols.length === 1) {
      // Synthesis-level fix only applies unambiguously when the cluster
      // touches a single column.
      after = synFix;
    }
    // Skip entries where we have neither value nor suggestion to show.
    if (!before && !after) continue;
    out.push({
      column: col,
      before,
      after,
      changed: !!after && before !== after,
    });
  }
  return out;
}

function renderDetail(bundle) {
  const el = $("#detail");
  if (!bundle) {
    el.innerHTML = `<div class="placeholder">No cluster selected.</div>`;
    return;
  }
  const c = bundle.cluster;
  const label = bundle.label;
  const syn = c.synthesis || {};
  const isFP = !!syn.is_likely_false_positive;

  const dets = (c.detections || []).map((d) => {
    // Relation findings put the discovered predicate + rationale in
    // the explanation field, separated by newlines. Preserve those
    // line breaks so the operator can read them; for non-relation
    // detections this is a no-op (single-line explanations).
    const explanationHtml = escape(d.explanation || "").replaceAll("\n", "<br>");
    return `
    <div class="det-row">
      <div class="et">${escape(d.error_type_name || d.error_type)}</div>
      <div class="obs"><b>Observed:</b> ${escape(d.observed || "")}</div>
      <div class="fix"><b>Explanation:</b> ${explanationHtml}</div>
      ${d.suggested_fix ? `<div class="fix"><b>Fix:</b> ${escape(d.suggested_fix)}</div>` : ""}
    </div>
    `;
  }).join("");

  // Relation-finding summary block. Surfaces the discovered
  // predicate(s) at the top of the detail pane so the operator does
  // not have to skim the explanation text to know what rule the row
  // is being judged against. The flat-issue compiler embeds the
  // predicate as the first line of the explanation
  // ("Discovered relation: ..."); we extract those lines verbatim
  // and dedupe across detections that share the same predicate.
  let relationBlockHtml = "";
  const relationDetections = (c.detections || []).filter(
    (d) => typeof d.discovery_strategy === "string"
      && d.discovery_strategy.startsWith("relation:"),
  );
  if (relationDetections.length) {
    const seen = new Set();
    const items = [];
    for (const d of relationDetections) {
      const lines = String(d.explanation || "").split("\n");
      const predicate = (lines.find((ln) => ln.startsWith("Discovered relation:"))
        || "").replace(/^Discovered relation:\s*/, "").trim();
      const rationale = (lines.find((ln) => ln.startsWith("Rationale:"))
        || "").replace(/^Rationale:\s*/, "").trim();
      const reviewer = (lines.find((ln) => ln.startsWith("Reviewer reason:"))
        || "").replace(/^Reviewer reason:\s*/, "").trim();
      const detection = (lines.find((ln) => ln.startsWith("Detection:"))
        || "").replace(/^Detection:\s*/, "").trim();
      const key = `${predicate}::${rationale}`;
      if (seen.has(key)) continue;
      seen.add(key);
      items.push(`
        <div style="margin-top:6px;">
          <div><b>Predicate:</b> ${escape(predicate || "(unknown)")}</div>
          ${rationale ? `<div style="margin-top:2px;"><b>Rationale:</b> ${escape(rationale)}</div>` : ""}
          ${reviewer ? `<div style="margin-top:2px;"><b>Reviewer reason:</b> ${escape(reviewer)}</div>` : ""}
          ${detection ? `<div style="margin-top:2px; color: var(--muted); font-size: 11.5px;">${escape(detection)}</div>` : ""}
        </div>
      `);
    }
    relationBlockHtml = `
      <div class="block">
        <h2>Discovered relation${items.length > 1 ? "s" : ""}</h2>
        ${items.join("")}
      </div>
    `;
  }

  // Duplicate-group summary: only meaningful for row_duplication
  // clusters. We collect the union of every detection's
  // ``affected_rows`` (plus the primary row) and surface the count +
  // row-list so the operator can verify the group at a glance and
  // jump into the context table (where these rows are already
  // amber-highlighted via the .dup-group class).
  let dupBlockHtml = "";
  const isDupCluster = (c.error_types || []).includes("row_duplication");
  if (isDupCluster) {
    const rows = new Set([Number(c.row)]);
    for (const d of c.detections || []) {
      for (const r of d.affected_rows || []) {
        const n = Number(r);
        if (Number.isFinite(n)) rows.add(n);
      }
    }
    const sorted = [...rows].sort((a, b) => a - b);
    const keyCols = (c.columns || []).join(", ");
    dupBlockHtml = `
      <div class="block">
        <h2>Duplicate group (${sorted.length} rows)</h2>
        <div class="cols">key: [${escape(keyCols)}]</div>
        <div style="margin-top:6px; font-size:12px;">
          rows: ${sorted.map((r) => `<span class="sev-chip">${escape(String(r))}</span>`).join(" ")}
        </div>
      </div>
    `;
  }

  const currentStatus = label?.status || "";
  const btn = (s, key) => `<button data-status="${s}" class="${s}${currentStatus === s ? " active" : ""}" title="key ${key}">${s} <kbd>${key}</kbd></button>`;

  // LLM-proposed corrected value. We prefer the cluster's reconciled
  // ``synthesis.canonical_fix`` (Phase B for multi-type clusters,
  // auto-synthesised "richest detection" for single-type ones) and
  // fall back to the longest non-empty per-detection ``suggested_fix``
  // when synthesis didn't fill one in. The label-side input prefills
  // with this value so the operator can press TP/PARTIAL without
  // retyping when the suggestion is already correct - the human is
  // the source of truth, but the LLM's draft saves keystrokes when
  // it gets it right.
  let llmSuggestion = "";
  const synFix = (syn?.canonical_fix || "").toString().trim();
  if (synFix && synFix !== "null") llmSuggestion = synFix;
  if (!llmSuggestion) {
    for (const d of c.detections || []) {
      const f = (d?.suggested_fix || "").toString().trim();
      if (f && f !== "null" && f.length > llmSuggestion.length) {
        llmSuggestion = f;
      }
    }
  }
  const operatorValue = label?.corrected_value || "";
  const correctedInputValue = operatorValue || llmSuggestion;
  const showLlmHint = !operatorValue && !!llmSuggestion;

  // Synthesis fallback. The aggregator only runs an LLM-driven
  // reconciliation pass when a cluster has 2+ detections to merge;
  // single-detection clusters carry no ``canonical_explanation`` /
  // ``canonical_fix``. To keep the Synthesis block useful in that
  // case, surface the lone detection's fields verbatim - they're
  // already the authoritative description of the issue, so there's
  // nothing for the LLM to "synthesise". ``synFix`` was already
  // computed above for the corrected-value prefill; we reuse it here.
  let synPrimaryType = syn.primary_error_type || "";
  let synExplanationToShow = (syn.canonical_explanation || "").toString().trim();
  let synFixToShow = synFix;
  let synFromSingleDetection = false;
  if (!synExplanationToShow && Array.isArray(c.detections) && c.detections.length === 1) {
    const only = c.detections[0] || {};
    synExplanationToShow = (only.explanation || "").toString().trim();
    if (!synFixToShow) synFixToShow = (only.suggested_fix || "").toString().trim();
    if (!synPrimaryType) synPrimaryType = only.error_type_name || only.error_type || "";
    synFromSingleDetection = !!synExplanationToShow;
  }

  // Per-column Original / Fix view. Original sits above Fix so the
  // block remains readable when the LLM's fix is a free-form
  // instruction ("Standardize the suffix to ...") rather than a
  // literal corrected value. Empty for clusters with nothing useful
  // to surface (row_duplication, FN-only); the block is omitted.
  const beforeAfter = computeBeforeAfter(c);
  const beforeAfterHtml = beforeAfter.length
    ? `
      <div class="block before-after">
        <h2>Original / Fix</h2>
        ${beforeAfter.map((row) => `
          <div class="ba-row">
            ${beforeAfter.length > 1 ? `<div class="ba-col">${escape(row.column)}</div>` : ""}
            <div class="ba-line">
              <span class="ba-tag">Original</span>
              <span class="ba-val${row.before ? "" : " ba-empty"}">${
                row.before ? escape(row.before) : '<span class="muted">(empty)</span>'
              }</span>
            </div>
            <div class="ba-line">
              <span class="ba-tag ba-tag-fix">Fix</span>
              <span class="ba-val ba-val-fix${row.after ? "" : " ba-empty"}">${
                row.after ? escape(row.after) : '<span class="muted">(no suggestion)</span>'
              }</span>
            </div>
          </div>
        `).join("")}
      </div>
    `
    : "";

  el.innerHTML = `
    <div class="block">
      <h2>${escape(c.cluster_id)}</h2>
      <div class="cols">${escape(c.source)} · row ${escape(String(c.row))} · [${escape((c.columns || []).join(", "))}]</div>
      <div style="margin-top:6px;">
        ${(c.error_types || []).map((t) => `<span class="sev-chip">${escape(t)}</span>`).join("")}
        ${isFP ? `<span class="sev-chip" style="color: var(--fp); border-color: var(--fp);">likely FP (synth)</span>` : ""}
        ${c.inferred ? `<span class="sev-chip inferred-chip" title="No primary detection filed for this row \u2014 it appears only via affected_rows of a sibling detection. Surfaced for FN-aware benchmark labelling.">inferred from affected_rows</span>` : ""}
      </div>
    </div>

    ${beforeAfterHtml}

    ${synExplanationToShow ? `
      <div class="block">
        <h2>Synthesis${synFromSingleDetection ? ` <span class="muted" style="text-transform:none; letter-spacing:0; font-size:11px; font-weight:400;">(from single detection)</span>` : ""}</h2>
        ${synPrimaryType ? `<div><b>${escape(synPrimaryType)}</b></div>` : ""}
        <div style="margin-top:4px;">${escape(synExplanationToShow)}</div>
        ${synFixToShow ? `<div style="margin-top:4px;"><b>Fix:</b> ${escape(synFixToShow)}</div>` : ""}
      </div>
    ` : ""}

    ${relationBlockHtml}

    ${renderWebVerificationBlock(bundle)}

    <div class="block">
      <h2>Verdict</h2>
      <div class="buttons" id="verdict-buttons">
        ${btn("tp","1")}${btn("fp","2")}${btn("partial","3")}${btn("unsure","4")}${btn("skip","5")}
      </div>
      <label class="field">
        <span>Corrected value (for tp / partial)</span>
        <input id="corrected-value" type="text" value="${escape(correctedInputValue)}" placeholder="(leave blank if unknown)" data-llm-suggestion="${escape(llmSuggestion)}" data-from-llm="${showLlmHint ? "1" : "0"}" />
        ${showLlmHint ? `<div class="muted llm-suggestion-hint" style="margin-top:4px; font-size:11px;">prefilled from LLM suggestion · <button type="button" id="clear-corrected" class="link-btn" style="background:none; border:none; padding:0; color: var(--accent); cursor:pointer; font-size:11px;">clear</button></div>` : ""}
      </label>
      <label class="field" style="flex-direction: row; align-items: center; gap: 6px;">
        <input id="propagate-same-value" type="checkbox" checked />
        <span style="color: var(--muted); font-size: 11.5px;">
          Apply this verdict to other rows with the same value
          (TP/partial → FN siblings; FP → FP sibling clusters)
        </span>
      </label>
      <label class="field">
        <span>Note</span>
        <textarea id="note">${escape(label?.note || "")}</textarea>
      </label>
      ${label ? `<div class="muted" style="margin-top:6px; font-size:11px;">last: ${escape(label.status)} by ${escape(label.annotator)} @ ${escape(label.ts)}</div>` : ""}
    </div>

    ${dupBlockHtml}

    <div class="block">
      <h2>Detections (${(c.detections || []).length})</h2>
      ${dets || '<div class="placeholder">None.</div>'}
    </div>
  `;

  el.querySelectorAll("#verdict-buttons button").forEach((b) => {
    b.addEventListener("click", () => submitLabel(b.dataset.status, { advance: false }));
  });

  // LLM prefill UX: the "from-llm" flag dims the input until the
  // operator clicks/edits it. As soon as they touch it, drop the
  // dimmed state - the value is theirs to keep or change. Also
  // clears the "prefilled" hint once edited.
  const correctedInput = $("#corrected-value");
  if (correctedInput) {
    const dropFromLlm = () => {
      if (correctedInput.dataset.fromLlm === "1") {
        correctedInput.dataset.fromLlm = "0";
        const hint = el.querySelector(".llm-suggestion-hint");
        if (hint) hint.remove();
      }
    };
    correctedInput.addEventListener("input", dropFromLlm);
    correctedInput.addEventListener("focus", () => {
      // Select the prefilled text on first focus so the operator
      // can immediately overwrite (vs. having to ctrl-A).
      if (correctedInput.dataset.fromLlm === "1") {
        correctedInput.select();
      }
    });
  }
  const clearBtn = $("#clear-corrected");
  if (clearBtn && correctedInput) {
    clearBtn.addEventListener("click", () => {
      correctedInput.value = "";
      correctedInput.dataset.fromLlm = "0";
      const hint = el.querySelector(".llm-suggestion-hint");
      if (hint) hint.remove();
      correctedInput.focus();
    });
  }

  // Web-verification "Accept" wiring: copy the canonical_value into
  // the Corrected-value input and pre-select the suggested verdict.
  el.querySelectorAll("[data-web-accept]").forEach((b) => {
    b.addEventListener("click", () => {
      const verdict = b.getAttribute("data-web-accept");
      const canon = b.getAttribute("data-canonical") || "";
      const correctedInput = $("#corrected-value");
      if (correctedInput && canon) {
        correctedInput.value = canon;
        correctedInput.dataset.fromLlm = "0";
        const hint = el.querySelector(".llm-suggestion-hint");
        if (hint) hint.remove();
      }
      if (verdict === "tp" || verdict === "fp") {
        // Pre-select the verdict button so Enter submits it.
        document.querySelectorAll("#verdict-buttons button").forEach((vb) => {
          vb.classList.toggle("active", vb.dataset.status === verdict);
        });
        toast(`accepted: ${verdict}${canon ? ` -> "${canon.slice(0, 40)}"` : ""} - press Enter to save`);
      }
    });
  });

  // "Verify on web" CTA wiring: POST to the labeling app, then
  // re-render the detail pane with the new verification block.
  el.querySelectorAll("[data-web-verify]").forEach((b) => {
    b.addEventListener("click", () => triggerWebVerification(b));
  });
}

// Trigger an on-demand web verification for the currently-displayed
// cluster. Disables the button + shows a spinner while in flight,
// then reloads the cluster (the GET response now carries the new
// web_verification field thanks to the server's mtime-aware reload).
async function triggerWebVerification(btn) {
  const cid = btn.getAttribute("data-web-verify");
  if (!cid) return;
  // ``data-web-retry`` is set on the Retry button rendered when an
  // existing verification has ``error`` set (e.g. an Azure agent
  // timeout). Server-side, ``?retry=1`` allows replacing the prior
  // errored record; without it the server returns 409.
  const isRetry = btn.getAttribute("data-web-retry") === "1";
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = isRetry ? "Retrying..." : "Verifying...";
  btn.classList.add("loading");
  try {
    const url = `${API}/clusters/${encodeURIComponent(cid)}/web-verification`
      + (isRetry ? "?retry=1" : "");
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      // 404 / 409 / 422 / 500 - server returns {detail: "..."}.
      let msg = `verify failed (${res.status})`;
      try {
        const body = await res.json();
        if (body && body.detail) msg = `${msg}: ${body.detail}`;
      } catch (e) { /* response wasn't JSON */ }
      toast(msg);
      btn.disabled = false;
      btn.textContent = original;
      btn.classList.remove("loading");
      return;
    }
    // Reload the cluster so renderDetail picks up the new
    // web_verification field via the cluster bundle.
    await loadCluster(cid);
    toast(isRetry ? "web verification retried" : "web verification complete");
  } catch (err) {
    toast(`verify error: ${err}`);
    btn.disabled = false;
    btn.textContent = original;
    btn.classList.remove("loading");
  }
}

// Render the "Web verification" block. Called from inside renderDetail.
// When the verifier has run, render the verdict + evidence + Accept
// buttons. When it hasn't but the cluster carries LLM detections,
// render a "Verify on web" CTA so the operator can trigger a single-
// cluster verification on demand.
function renderWebVerificationBlock(bundle) {
  const v = bundle?.web_verification;
  const cluster = bundle?.cluster || {};
  const cid = cluster.cluster_id || "";
  const hasDetections = Array.isArray(cluster.detections)
    ? cluster.detections.length > 0
    : false;

  // No verification yet -> offer the manual-verify CTA, but only when
  // the cluster has at least one LLM-flagged detection (the user's
  // requirement: not on bare cells, only on already-flagged ones).
  if (!v) {
    if (!hasDetections) return "";
    return `
      <div class="block web-verify web-verify-cta">
        <h2><span>Web verification</span></h2>
        <div class="muted" style="font-size:12px; margin-bottom:8px;">
          This cluster has not been verified on the web yet.
          Click below to run the Foundry agent (Web Search tool).
        </div>
        <button class="web-verify-btn" data-web-verify="${escape(cid)}">
          Verify on web
        </button>
      </div>
    `;
  }

  const verdict = String(v.verdict || "ambiguous");
  const conf = Number(v.confidence || 0);
  const confPct = `${Math.round(conf * 100)}%`;
  const errored = !!v.error;

  // The backend's anti-hallucination filter drops any evidence URL
  // the agent claimed unless it was also recorded as a real
  // ``url_citation`` annotation by the Web Search tool. When the
  // surviving evidence list is empty AND the agent originally
  // declared confirmed_error, ``_verify_one`` downgrades the verdict
  // to ambiguous and stamps ``parse_error`` with this exact phrase.
  // In that situation the model's narrative reasoning typically
  // contains fabricated source attributions ("Multiple authoritative
  // sources confirm...") -- it sounds confident but is not backed
  // by anything the search tool actually returned. We surface the
  // reasoning behind a prominent warning so the operator does not
  // mistake it for a verified conclusion.
  const reasoningUnreliable = !errored
    && typeof v.parse_error === "string"
    && v.parse_error.includes("no evidence URLs survived");

  // Verdict chip color follows convention: confirmed=accent, fp=fp,
  // ambiguous=muted, errored=warn.
  const chipColor =
    errored        ? "var(--warn, #f59e0b)"
    : verdict === "confirmed_error" ? "var(--accent)"
    : verdict === "false_positive"  ? "var(--fp, #f87171)"
    : "var(--muted)";

  const evidenceHtml = (v.evidence || []).map((e) => `
    <li>
      <a href="${escape(e.url)}" target="_blank" rel="noopener noreferrer">
        ${escape(e.title || e.url)}
      </a>
      ${e.quote ? `<div class="ev-quote">"${escape(e.quote)}"</div>` : ""}
    </li>
  `).join("");

  // Accept button: only meaningful when the verdict is actionable.
  // single_cell + confirmed_error -> set Corrected value + pre-select TP.
  // false_positive  -> just pre-select FP (no corrected value).
  // multi_row / multi_column / composite + confirmed_error -> we
  // currently only offer a TP pre-select (operator applies the
  // structured fix manually using the ``canonical_record`` table
  // shown below), since the row/column targeting differs per
  // detection and we don't have a single "correct value" to
  // splice into the existing TP-flow.
  const pattern = String(v.pattern || "single_cell");
  let acceptBtn = "";
  if (!errored && verdict === "confirmed_error" && v.canonical_value && pattern === "single_cell") {
    acceptBtn = `<button class="web-accept" data-web-accept="tp"
        data-canonical="${escape(v.canonical_value)}"
        title="Use this canonical value as the correction and pre-select TP">
        Accept TP -> "${escape(v.canonical_value)}"
      </button>`;
  } else if (!errored && verdict === "confirmed_error" && pattern !== "single_cell") {
    acceptBtn = `<button class="web-accept" data-web-accept="tp"
        data-canonical=""
        title="Pre-select TP; apply the structured fix from canonical_record manually">
        Accept TP
      </button>`;
  } else if (!errored && verdict === "false_positive") {
    acceptBtn = `<button class="web-accept" data-web-accept="fp"
        data-canonical=""
        title="Pre-select FP">
        Accept FP
      </button>`;
  }

  // ``v.queries`` now holds Bing search URLs the agent fired (api
  // host already rewritten to www.bing.com on the server). Render
  // them as clickable links so an operator can audit the searches.
  const queriesHtml = (v.queries || []).map((q) => {
    const url = String(q || "");
    let label = url;
    try { label = decodeURIComponent(new URL(url).search.replace(/^\?q=/, "")) || url; }
    catch (_) { /* not a URL: fall back to raw text */ }
    return `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer"><code>${escape(label)}</code></a>`;
  }).join(" ");

  const nCalls = Number(v.n_results || 0);
  const nUrls = Number(v.n_unique_urls || 0);
  const callsLabel = `${nUrls} cited url${nUrls === 1 ? "" : "s"} / ${nCalls} web search${nCalls === 1 ? "" : "es"}`;

  // ---- Pattern badge + structured-fix / records-snapshot blocks --
  // Only rendered for non-single_cell verifications so the existing
  // single-cell UI stays compact.
  const patternLabels = {
    single_cell: "single cell",
    multi_row: "multi-row (duplicate check)",
    multi_column: "multi-column (consistency check)",
    composite: "composite (rows x cols)",
  };
  const patternBadge = (pattern && pattern !== "single_cell")
    ? `<span class="verdict-chip" style="border-color:var(--muted);color:var(--muted);margin-left:6px;" title="Cluster shape">${escape(patternLabels[pattern] || pattern)}</span>`
    : "";

  // canonical_record: column -> corrected_value mapping for
  // multi_column / composite confirmed_error verdicts. Render as a
  // small table so the operator can see at a glance which columns
  // need to change.
  let canonicalRecordHtml = "";
  if (v.canonical_record && typeof v.canonical_record === "object") {
    const rows = Object.entries(v.canonical_record);
    if (rows.length) {
      canonicalRecordHtml = `
        <div style="margin-top:8px;">
          <div class="muted" style="font-size:11px; margin-bottom:4px;">
            Suggested per-column fixes:
          </div>
          <table class="canonical-record" style="font-size:12px;border-collapse:collapse;">
            <tbody>
              ${rows.map(([col, val]) => `
                <tr>
                  <td style="padding:2px 8px 2px 0;color:var(--muted);">${escape(col)}</td>
                  <td style="padding:2px 0;"><code>${escape(String(val))}</code></td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }
  }

  // record_snapshots: one entry per record actually shown to the
  // agent. For multi_row / composite this is the list of duplicate-
  // candidate rows; for multi_column it's a single record projected
  // to the columns under review. We render a compact summary so the
  // operator can verify the agent saw the right data.
  let recordsHtml = "";
  const snaps = Array.isArray(v.record_snapshots) ? v.record_snapshots : [];
  if ((pattern === "multi_row" || pattern === "composite") && snaps.length >= 2) {
    const rowsList = Array.isArray(v.rows) ? v.rows : [];
    // Collect the union of keys (column names) across all snapshots
    // so the operator sees one consistent column order.
    const keyOrder = [];
    const seenKey = new Set();
    for (const s of snaps) {
      for (const k of Object.keys(s || {})) {
        if (!seenKey.has(k)) { seenKey.add(k); keyOrder.push(k); }
      }
    }
    const headerHtml = keyOrder.slice(0, 8).map((k) =>
      `<th style="padding:2px 6px;text-align:left;color:var(--muted);font-weight:600;">${escape(k)}</th>`
    ).join("");
    const bodyHtml = snaps.map((s, i) => {
      const r = rowsList[i];
      const rowLabel = (typeof r === "number")
        ? `row ${r}`
        : `record ${i + 1}`;
      const tds = keyOrder.slice(0, 8).map((k) => {
        const val = String((s && s[k]) || "");
        const trunc = val.length > 60 ? val.slice(0, 60) + "\u2026" : val;
        return `<td style="padding:2px 6px;">${escape(trunc)}</td>`;
      }).join("");
      return `<tr><td style="padding:2px 6px;color:var(--muted);white-space:nowrap;">${escape(rowLabel)}</td>${tds}</tr>`;
    }).join("");
    recordsHtml = `
      <div style="margin-top:8px;overflow-x:auto;">
        <div class="muted" style="font-size:11px; margin-bottom:4px;">
          Records compared (${snaps.length}):
        </div>
        <table style="font-size:12px;border-collapse:collapse;">
          <thead><tr><th style="padding:2px 6px;text-align:left;color:var(--muted);">&nbsp;</th>${headerHtml}</tr></thead>
          <tbody>${bodyHtml}</tbody>
        </table>
      </div>
    `;
  }

  return `
    <div class="block web-verify">
      <h2>
        <span>Web verification</span>
        <span class="verdict-chip" style="border-color:${chipColor};color:${chipColor};">
          ${escape(errored ? "errored" : verdict)} · ${confPct}
        </span>
        ${patternBadge}
      </h2>
      ${errored
        ? `<div class="muted" style="font-size:12px;">${escape(v.error || "verifier failed")}</div>
           <div style="margin-top:8px;">
             <button class="web-verify-btn" data-web-verify="${escape(cid)}" data-web-retry="1">
               Retry verification
             </button>
           </div>`
        : `
          ${v.canonical_value
            ? `<div><b>Canonical:</b> <span class="canon">${escape(v.canonical_value)}</span></div>`
            : ""}
          ${v.reasoning
            ? (reasoningUnreliable
              ? `<div class="reasoning reasoning-warn">
                   <div class="reasoning-warn-banner">
                     ⚠ Agent reasoning is not backed by evidence:
                     it claimed sources that the Web Search tool did
                     not actually return. Treat the conclusion below
                     as unreliable; the verdict was downgraded to
                     <b>ambiguous</b>.
                   </div>
                   <div class="reasoning-text">${escape(v.reasoning)}</div>
                 </div>`
              : `<div class="reasoning">${escape(v.reasoning)}</div>`)
            : ""}
          ${canonicalRecordHtml}
          ${recordsHtml}
          ${evidenceHtml
            ? `<ul class="evidence">${evidenceHtml}</ul>`
            : `<div class="muted" style="font-size:12px;">No surviving evidence citations.</div>`}
          <div class="muted web-meta">
            ${queriesHtml ? `searches: ${queriesHtml} · ` : ""}
            ${callsLabel}
            ${v.entity_kind ? ` · kind: ${escape(v.entity_kind)}` : ""}
          </div>
          ${acceptBtn ? `<div style="margin-top:8px;">${acceptBtn}</div>` : ""}
        `}
    </div>
  `;
}

async function renderCsv(cluster) {
  const head = $("#csv-head");
  const viewer = $("#csv-viewer");
  const source = cluster.source;
  const targetRow = cluster.row ?? 0;
  const flagged = new Set(cluster.columns || []);
  // Affected-rows set: union of every detection's ``affected_rows``
  // in this cluster, minus the primary row. Painted in the context
  // table so a row_duplication cluster shows the whole group
  // highlighted, not just the canonical row.
  const affectedRows = new Set();
  for (const d of cluster.detections || []) {
    for (const r of d.affected_rows || []) {
      const n = Number(r);
      if (Number.isFinite(n)) affectedRows.add(n);
    }
  }
  affectedRows.delete(targetRow);
  // For row_duplication clusters, highlight the target row as part
  // of the duplicate group too so the visual band is contiguous and
  // the operator can see at a glance which rows are siblings.
  const isDupCluster_ = (cluster.error_types || []).includes("row_duplication");
  const sourceChanged = state.currentSource !== source;
  state.currentSource = source;
  state.currentTargetRow = targetRow;
  state.currentColumns = flagged;
  // Update the exclusion bar's chip strip / button enable-state
  // whenever the focused CSV changes.
  if (sourceChanged) updateExclusionUi();

  const start = Math.max(0, targetRow - CSV_WINDOW);
  const limit = CSV_WINDOW * 2 + 1;

  // Soft-repaint short-circuit: when the slice we'd fetch is exactly
  // what's already on screen (same source + window), skip the data
  // fetch + full DOM rebuild and just retoggle the highlight classes.
  // Cell-state classes were already applied by ``repaintCellStates``
  // after the most recent label save, and click handlers ride along
  // on the same <td> nodes, so the operator gets a snappy advance
  // even when 22k clusters are loaded.
  const slice = state.currentSlice;
  if (
    slice
    && slice.source === source
    && slice.start === start
    && slice.limit === limit
    && viewer.querySelector("table")
  ) {
    applyCsvHighlights(cluster);
    applyExclusionMarks();
    return;
  }

  const cacheKey = `${source}:${start}:${limit}`;
  let data = state.csvCache.get(cacheKey);
  if (!data) {
    try {
      data = await api(`${API}/sources/${encodeURIComponent(source)}?start=${start}&limit=${limit}`);
      state.csvCache.set(cacheKey, data);
    } catch (e) {
      viewer.innerHTML = `<div class="placeholder">Failed to load ${escape(source)}: ${escape(e.message)}</div>`;
      return;
    }
  }

  // Load cell-state for the source so we can paint per-cell verdicts
  // (TP/FP/PARTIAL/FN/UNSURE) on every visible row, not just the
  // current cluster's row.
  const cellMap = await loadCellState(source);

  // For multi-workbook runs, prefix the file name with its workbook so
  // the operator can see which logical table the row belongs to (and
  // notice when adjacent clusters are from sibling tables of the same
  // workbook - that's where inter-table errors live).
  const wb = state.run?.multi_workbook
    ? (state.run.sources.find((s) => s.name === source)?.table_group || "")
    : "";
  // Original-xlsx buttons: surface a "Download .xlsx" + "Preview" pair
  // when the run config exposes the original workbook (curated CSVs
  // were derived from). The backend's per-source slice carries the
  // resolved verdict so we don't need a second round-trip.
  const wbForXlsx = data.workbook || wb;
  const xlsxHtml = (data.has_xlsx && wbForXlsx)
    ? `<span class="xlsx-actions">
        <a class="xlsx-btn" href="${API}/workbooks/${encodeURIComponent(wbForXlsx)}/xlsx" download="${escape(wbForXlsx)}" title="Download original .xlsx">⬇ xlsx</a>
        <a class="xlsx-btn" href="${API}/workbooks/${encodeURIComponent(wbForXlsx)}/preview" target="_blank" rel="noopener" title="Open original workbook preview in new tab">📄 preview</a>
      </span>`
    : "";
  // Sibling tables strip: jump directly to a sibling CSV. ``data.siblings``
  // is populated for multi-table corpora (curated_pbi / Bing21 reconstructed).
  const siblings = (data.siblings || []);
  const siblingHtml = siblings.length
    ? `<nav class="sibling-tables" aria-label="Sibling tables">
        <span class="label">${escape(wb || "siblings")}:</span>
        <button class="active" disabled>${escape(source)}</button>
        ${siblings.map((s) => `<button data-jump-source="${escape(s)}">${escape(s)}</button>`).join("")}
      </nav>`
    : "";
  $("#csv-source").innerHTML =
    `<span>${escape(wb ? `${wb} / ${source}` : source)}</span>${xlsxHtml}${siblingHtml}`;
  $("#csv-pos").textContent = `rows ${data.start}..${data.end} of ${data.total_rows} · target row ${targetRow}`;

  // Wire sibling-tables jump buttons. Clicking jumps to the first
  // cluster on that sibling, or to row 0 if the sibling has no
  // clusters.
  $("#csv-source").querySelectorAll("[data-jump-source]").forEach((b) => {
    b.addEventListener("click", () => jumpToSource(b.getAttribute("data-jump-source")));
  });

  let html = '<table><thead><tr><th>#</th>';
  for (const col of data.columns) {
    const flag = flagged.has(col);
    html += `<th class="${flag ? "flagged" : ""}" data-col="${escape(col)}">${escape(col)}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (const r of data.rows) {
    const isTarget = r.row === targetRow;
    const isDup = affectedRows.has(r.row) || (isDupCluster_ && isTarget);
    const rowCls = ["row-idx", isTarget ? "target" : "", isDup ? "dup-group" : ""]
      .filter(Boolean)
      .join(" ");
    html += `<tr class="${rowCls}" data-row="${escape(String(r.row))}"><td>${escape(String(r.row))}</td>`;
    for (const col of data.columns) {
      const flag = flagged.has(col) && isTarget;
      const ce = cellMap.get(`${r.row}\x00${col}`);
      const stateCls = ce ? `cell-${ce.state}` : "";
      const shadowed = ce && ce.superseded_fn_by ? "cell-shadowed" : "";
      const value = r.values[col] ?? "";
      const titleParts = [];
      if (ce) {
        titleParts.push(`label: ${ce.state}`);
        if (ce.error_types?.length) titleParts.push(`types: ${ce.error_types.join(", ")}`);
        if (ce.cluster_ids?.length) titleParts.push(`cluster: ${ce.cluster_ids[0]}`);
        if (ce.note) titleParts.push(`note: ${ce.note}`);
        if (ce.annotator) titleParts.push(`by ${ce.annotator}`);
        if (ce.superseded_fn_by) titleParts.push("FN superseded by TP cluster");
      }
      const titleAttr = titleParts.length ? `title="${escape(titleParts.join(" \u00B7 "))}"` : "";
      // Whitespace-anomaly badge: a small dotted underline on cells
      // whose value contains leading/trailing/multi-space, tabs,
      // NBSP, line breaks, or zero-width characters. Cheap regex
      // mirror of ``renderCsvCellHtml``'s detection rules so the CSS
      // class matches the rendered markers.
      const valueStr = value == null ? "" : String(value);
      const hasWs =
        valueStr !== ""
        && (
          /^[ ]|[ ]$|[ ]{2,}/.test(valueStr)
          || /[\t\u00A0\n\r\u2028\u2029\u200B\u200C\u200D\u00AD\uFEFF]/.test(valueStr)
        );
      const wsCls = hasWs ? "has-ws" : "";
      const classes = ["clickable", flag ? "flagged" : "", stateCls, shadowed, wsCls].filter(Boolean).join(" ");
      // Per-(column, error_type) labelling-effort chip. Only painted
      // on cells the LLM annotated (ce present + has error_types) -
      // unannotated cells stay clean. Each chip shows the Chao1
      // projected-missed count for its (column, error_type) arm
      // (rounded). The chip is purely informational; clicking it
      // still fires the cell-popover handler so the chip never
      // intercepts the labelling flow.
      let chipsHtml = "";
      if (ce && ce.error_types && ce.error_types.length) {
        const lookup = effortLookupForCell(source, col);
        if (lookup) {
          const chips = [];
          for (const et of ce.error_types) {
            const e = lookup.byEt.get(et);
            if (!e) continue;
            const missed = Math.round(e.estimated_missed || 0);
            // Render a chip for every (cell, error_type) arm. Saturated/
            // exhausted arms with 0 projected-missed still get a chip
            // (a green "\u2713" check) so the operator can confirm
            // coverage at a glance instead of seeing "no chip = ?".
            const isDone = (e.status === "saturated" || e.status === "exhausted") && missed <= 0;
            const text = isDone ? "\u2713" : `+${missed}`;
            const tip = `${et}: ` +
              (isDone ? "detector converged, " : `+${e.estimated_missed.toFixed(1)} more projected, `) +
              `seen ${e.n_observed}, status ${armStatusLabel(e.status)}, ` +
              `GT mass ${(e.missing_mass * 100).toFixed(1)}%`;
            chips.push(
              `<span class="cell-effort cell-effort-${escape(e.status)}" ` +
              `title="${escape(tip)}">${text}</span>`
            );
          }
          if (chips.length) {
            chipsHtml = `<span class="cell-effort-chips">${chips.join("")}</span>`;
          }
        }
      }
      html += `<td class="${classes}" data-row="${escape(String(r.row))}" data-col="${escape(col)}" ${titleAttr}>${renderCsvCellHtml(value)}${chipsHtml}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  viewer.innerHTML = html;

  // Audit verdict markers: paint a small dot in the corner of every
  // cell in audited columns (loaded from column_verifications/*.jsonl).
  // Also fire async fetches for any audited (source, *) pair we
  // don't yet have detail for - the fetch will repaint when it
  // arrives. Done AFTER innerHTML so markers can be appended without
  // being blown away by the rebuild.
  paintAuditMarkersForSource(source);
  ensureAuditDetailsForSource(source);

  // Cell-click handler: opens a popover anchored to the clicked cell.
  // For LLM-flagged cells (state in {tp, fp, partial, unsure}) the
  // popover lets the operator change the verdict on the underlying
  // cluster. For unlabelled cells it offers an FN action. Cells
  // already TP-via-cluster cannot be FN'd (TP wins).
  viewer.querySelectorAll("td.clickable").forEach((td) => {
    td.addEventListener("click", (e) => {
      // Don't open the popover when the click is the tail of a
      // text-selection drag (so the operator can highlight cell
      // contents and copy with Ctrl+C without the popover stealing
      // focus). A selection that spans more than one character is a
      // strong signal the user is selecting, not labelling.
      const sel = window.getSelection();
      if (sel && !sel.isCollapsed && String(sel).length > 0) return;
      const row = Number(td.getAttribute("data-row"));
      const col = td.getAttribute("data-col");
      openCellPopover(td, source, row, col);
    });
  });

  // Scroll the flagged cell (or, failing that, the target row) into
  // view in BOTH axes. Tables wider/taller than the viewport would
  // otherwise leave the highlighted cell off-screen, forcing the
  // operator to scroll manually to see what was flagged.
  requestAnimationFrame(() => {
    const tr = viewer.querySelector(`tr[data-row="${targetRow}"]`);
    if (!tr) return;
    // Pick the first .flagged cell on the target row; if none (e.g.
    // FN browsing mode), fall back to the row itself.
    const cell = tr.querySelector("td.flagged") || tr;
    cell.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
  });

  // Stamp the slice we just rendered so a follow-up renderCsv can
  // soft-repaint when the (source, start, limit) tuple is identical
  // (e.g. consecutive clusters that share a row, or popover-driven
  // re-renders that don't move the window).
  state.currentSlice = {
    source,
    start,
    limit,
    totalRows: data.total_rows,
  };
  // Apply exclusion strikethroughs to the freshly-rendered table. This
  // runs after the DOM is in place so the per-cell / per-row class
  // toggles touch real nodes.
  applyExclusionMarks();
}

// Repaint just the highlight classes (target row, flagged columns/
// cells, dup-group rows) on the existing CSV DOM. Used when the
// slice we'd render hasn't changed - the rebuild was the dominant
// cost on advance, so this drops the per-save UI work to a few
// classList toggles. Cell-state classes (cell-tp/fp/...) and click
// handlers are preserved across calls because the DOM nodes don't
// move.
function applyCsvHighlights(cluster) {
  const viewer = $("#csv-viewer");
  if (!viewer) return;
  const targetRow = cluster.row ?? 0;
  const flagged = new Set(cluster.columns || []);
  const affectedRows = new Set();
  for (const d of cluster.detections || []) {
    for (const r of d.affected_rows || []) {
      const n = Number(r);
      if (Number.isFinite(n)) affectedRows.add(n);
    }
  }
  affectedRows.delete(targetRow);
  const isDupCluster_ = (cluster.error_types || []).includes("row_duplication");
  state.currentTargetRow = targetRow;
  state.currentColumns = flagged;

  // Header columns.
  for (const th of viewer.querySelectorAll("thead th[data-col]")) {
    th.classList.toggle("flagged", flagged.has(th.getAttribute("data-col")));
  }
  // Body rows + per-cell flagged class on the target row.
  for (const tr of viewer.querySelectorAll("tbody tr[data-row]")) {
    const r = Number(tr.getAttribute("data-row"));
    const isTarget = r === targetRow;
    const isDup = affectedRows.has(r) || (isDupCluster_ && isTarget);
    tr.classList.toggle("target", isTarget);
    tr.classList.toggle("dup-group", isDup);
    for (const td of tr.querySelectorAll("td.clickable")) {
      const col = td.getAttribute("data-col");
      td.classList.toggle("flagged", isTarget && flagged.has(col));
    }
  }
  // Position indicator (window unchanged - just update target row).
  const slice = state.currentSlice;
  if (slice) {
    const pos = $("#csv-pos");
    if (pos) {
      pos.textContent =
        `rows ${slice.start}..${Math.min(slice.start + slice.limit, slice.totalRows)} `
        + `of ${slice.totalRows} \u00B7 target row ${targetRow}`;
    }
  }
  requestAnimationFrame(() => {
    const tr = viewer.querySelector(`tr[data-row="${targetRow}"]`);
    if (!tr) return;
    const cell = tr.querySelector("td.flagged") || tr;
    cell.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
  });
}

// ----------------------------------------------------------- cell state

async function loadCellState(source, { force = false } = {}) {
  if (!force && state.cellState.has(source)) return state.cellState.get(source);
  let map = new Map();
  try {
    const r = await api(`${API}/sources/${encodeURIComponent(source)}/cells`);
    for (const ce of r.cells || []) {
      map.set(`${ce.row}\x00${ce.column}`, ce);
    }
  } catch (_) {
    // Soft failure: cells just won't be coloured.
  }
  state.cellState.set(source, map);
  return map;
}

function repaintCellStates(source) {
  const map = state.cellState.get(source);
  if (!map) return;
  const viewer = $("#csv-viewer");
  viewer.querySelectorAll("td.clickable").forEach((td) => {
    const row = Number(td.getAttribute("data-row"));
    const col = td.getAttribute("data-col");
    const ce = map.get(`${row}\x00${col}`);
    // Strip any prior cell-* / cell-shadowed class before reapplying.
    td.className = td.className.split(/\s+/).filter((c) =>
      !c.startsWith("cell-") || c === "cell-shadowed"
    ).filter((c) => c !== "cell-shadowed").join(" ");
    if (ce) {
      td.classList.add(`cell-${ce.state}`);
      if (ce.superseded_fn_by) td.classList.add("cell-shadowed");
    }
  });
}

// ----------------------------------------------------------- popover

function closeCellPopover() {
  document.querySelectorAll(".cell-popover").forEach((el) => el.remove());
}

// Extract a cell's raw value text from its <td>, ignoring any
// .cell-effort-chips span the labelling-effort renderer inlined inside
// the cell. The CSV renderer paints "<value><chips>" so we need to
// strip the chips before treating the result as the user-visible value.
function rawCellValue(td) {
  if (!td) return "";
  const clone = td.cloneNode(true);
  clone.querySelectorAll(".cell-effort-chips").forEach((n) => n.remove());
  return clone.textContent || "";
}

// Markup for the "apply to other rows with the same value" toggle in
// the cell popover. Disabled (and labelled accordingly) when the anchor
// cell is empty - the propagation endpoint compares values for
// equality, so an empty anchor would match every blank sibling.
function propagateCheckboxHtml(anchorValue, prevCorrectedValue) {
  const trimmed = (anchorValue || "").trim();
  const empty = trimmed === "";
  const label = empty
    ? "Cannot auto-apply: cell is empty"
    : "Also FN-label other rows in this column with the same value";
  const disabled = empty ? "disabled" : "";
  // Default checked when there's a previous correction (rerun of FN
  // update) or always-on for fresh labels - matches the cluster-side
  // propagation behaviour (checked by default).
  void prevCorrectedValue;
  const checked = empty ? "" : "checked";
  return `<label class="propagate-row" style="display:flex;align-items:center;gap:6px;margin-top:6px;font-size:11.5px;color:var(--muted);">
    <input type="checkbox" data-field="propagate-same-value" ${checked} ${disabled} />
    <span>${escape(label)}</span>
  </label>`;
}

function openCellPopover(anchorEl, source, row, col) {
  closeCellPopover();
  const map = state.cellState.get(source) || new Map();
  const ce = map.get(`${row}\x00${col}`);
  const pop = document.createElement("div");
  pop.className = "cell-popover";
  // Anchor cell's raw text value, excluding any labelling-effort chips
  // we render inside the <td>. Used both to display the value in the
  // popover header AND to guard same-value propagation against being
  // fired on empty cells (where _scalar("") would match every empty
  // sibling and FN-spam the column).
  const cellValueRaw = rawCellValue(anchorEl);
  const valueText = cellValueRaw.slice(0, 80);
  const wb = state.run?.multi_workbook
    ? (state.run.sources.find((s) => s.name === source)?.table_group || "")
    : "";
  const headLabel = wb ? `${wb} / ${source}` : source;

  let body = `<h3>Cell verdict</h3>
    <div class="meta">${escape(headLabel)} \u00B7 row ${escape(String(row))} \u00B7 ${escape(col)}</div>
    <div class="meta">value: ${escape(valueText) || "<em>(empty)</em>"}</div>`;

  // Web-audit findings for this cell (from the column-wide audit).
  // Surfaced FIRST (above the GT stats) because when present it's
  // the most actionable signal: a confirmed_error verdict on a cell
  // the LLM never flagged is the headline FN-discovery use case.
  body += renderAuditFindingsForCell(source, row, col);

  // Good-Turing / Chao1 stats for this (source, column). Surfaces the
  // per-error-type projected-missed counts, observed counts, GT
  // missing mass, and arm status. Only rendered when the run has a
  // per-arm artifact AND this column appears in it. Limited to the
  // top 5 error types by projected-missed so the popover stays
  // compact; remainder is accessible via the labelling-effort panel.
  const gtLookup = effortLookupForCell(source, col);
  if (gtLookup) {
    const c = gtLookup.cell;
    const cellEts = new Set(ce && ce.error_types ? ce.error_types : []);
    const rows = (c.by_error_type || []).slice(0, 5).map((e) => {
      const cls = `gt-status-${escape(e.status)}` + (cellEts.has(e.error_type) ? " gt-row-match" : "");
      const marker = cellEts.has(e.error_type) ? "\u25C6 " : "";
      const stTip = armStatusTooltip(e.status);
      const stTitle = stTip ? ` title="${escape(stTip)}"` : "";
      return `<tr class="${cls}">
        <td class="et" title="${escape(e.error_type)}">${marker}${escape(e.error_type)}</td>
        <td class="num">${e.n_observed}</td>
        <td class="num">${e.estimated_total.toFixed(1)}</td>
        <td class="num">+${e.estimated_missed.toFixed(1)}</td>
        <td class="num">${(e.missing_mass * 100).toFixed(1)}%</td>
        <td class="status"${stTitle}>${escape(armStatusLabel(e.status))}</td>
      </tr>`;
    }).join("");
    const more = (c.by_error_type || []).length > 5
      ? `<div class="muted gt-more">+${c.by_error_type.length - 5} more (see Labelling effort panel)</div>`
      : "";
    body += `<div class="gt-stats">
      <div class="gt-head">
        <strong>Good-Turing / Chao1 for this column</strong>
        <span class="muted">labelled ${c.labelled_so_far} \u00B7 seen ${Math.round(c.n_observed)} \u00B7 total est. ${c.chao1_estimated_total.toFixed(1)} \u00B7 missed ${c.chao1_estimated_missed.toFixed(1)}</span>
      </div>
      <table class="gt-table">
        <thead><tr>
          <th title="Error type">type</th>
          <th title="Distinct defect species observed for this (column, error_type) arm">seen</th>
          <th title="Chao1 projected total distinct species (observed + unseen)">total</th>
          <th title="Chao1 projected unseen species - the operator should plan to encounter ~this many more">missed</th>
          <th title="Good-Turing missing mass: probability the next perturbed run surfaces a never-before-seen species">GT</th>
          <th title="Per-arm verdict from the adaptive stop-predicate. Hover individual rows for details. converged = detector reached its ceiling; budget-capped = hit k_max with GT still high; too sparse = too few species to call.">status</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${more}
    </div>`;
  }

  if (ce && (ce.state === "tp" || ce.state === "partial" || ce.state === "fp" || ce.state === "unsure") && ce.label_source === "cluster") {
    // Cluster-level verdict already exists. Offer quick re-label of the
    // underlying cluster (so the operator can correct mistakes spotted
    // mid-review without leaving the cell). FN is intentionally not
    // offered because the cell is already covered by a cluster.
    const cid = (ce.cluster_ids || [])[0] || "";
    body += `<div class="meta">current: <b>${escape(ce.state)}</b> via cluster <code>${escape(cid)}</code>${ce.error_types?.length ? ` \u00B7 ${escape(ce.error_types.join(", "))}` : ""}</div>`;
    // Same-value propagation toggle for popover-driven cluster
    // re-labels. The TP/Partial branch needs a correction to FN-spread
    // siblings; the popover doesn't collect one, so propagation only
    // fires for FP here. We keep the checkbox visible (and checked)
    // for all four verdicts, but submitClusterStatusForCell skips
    // propagation when status != fp.
    body += propagateCheckboxHtml(cellValueRaw, "");
    body += `<div class="actions">
      <button class="tp"      data-cluster-status="tp"     >TP</button>
      <button class="partial" data-cluster-status="partial">Partial</button>
      <button class="fp"      data-cluster-status="fp"     >FP</button>
      <button class="unsure"  data-cluster-status="unsure" >Unsure</button>
      <button class="close"   data-action="close">Close</button>
    </div>
    <div class="hint">Tip: changing the verdict updates the source cluster <code>${escape(cid)}</code>; all cells of that cluster are repainted.</div>`;
    pop.innerHTML = body;
    pop.dataset.clusterId = cid;
  } else if (ce && ce.state === "fn") {
    // Existing FN cell. Let the operator clear it (to FP-on-cell, which
    // we treat as "not actually an error after all") or edit the note.
    // Same-value propagation is offered on Update so the operator can
    // retroactively mirror an FN to all matching siblings discovered
    // after the initial label.
    body += `<div class="meta">current: <b>fn</b>${ce.annotator ? ` by ${escape(ce.annotator)}` : ""}</div>
      <label>Note <textarea data-field="note" rows="2">${escape(ce.note || "")}</textarea></label>
      ${propagateCheckboxHtml(cellValueRaw, ce.corrected_value)}
      <div class="actions">
        <button class="fn"     data-cell-status="fn"  data-update="1">Update</button>
        <button class="fp"     data-cell-status="fp">Clear FN</button>
        <button class="close"  data-action="close">Close</button>
      </div>`;
    pop.innerHTML = body;
  } else {
    // Unlabelled cell. Offer FN (the headline feature). Note input
    // captures the rationale.
    const blocked = ce && (ce.state === "tp" || ce.state === "partial");
    if (blocked) {
      const cid = (ce.cluster_ids || [])[0] || "";
      body += `<div class="hint warn">Already labelled <b>${escape(ce.state)}</b> via cluster <code>${escape(cid)}</code> - cannot mark as FN.</div>
        <div class="actions"><button class="close" data-action="close">Close</button></div>`;
    } else {
      body += `<label>Note <textarea data-field="note" rows="2" placeholder="why is this an error the LLM missed?"></textarea></label>
        <label>Corrected value <input type="text" data-field="corrected_value" placeholder="(optional)" /></label>
        ${propagateCheckboxHtml(cellValueRaw, "")}
        <div class="actions">
          <button class="fn"    data-cell-status="fn">Mark as FN</button>
          <button class="close" data-action="close">Cancel</button>
        </div>
        <div class="hint">FN labels the cell as a real error the LLM missed. If a TP cluster is later created on this cell, the FN claim is auto-superseded.</div>`;
    }
    pop.innerHTML = body;
  }

  // Anchor next to the clicked cell.
  document.body.appendChild(pop);
  const r = anchorEl.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let left = r.right + 8, top = r.top;
  if (left + pw > window.innerWidth) left = Math.max(8, r.left - pw - 8);
  if (top + ph > window.innerHeight) top = Math.max(8, window.innerHeight - ph - 8);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;

  // Wire actions.
  pop.querySelectorAll("[data-action='close']").forEach((b) =>
    b.addEventListener("click", closeCellPopover)
  );
  pop.querySelectorAll("[data-cluster-status]").forEach((b) =>
    b.addEventListener("click", async () => {
      const cid = pop.dataset.clusterId;
      const status = b.getAttribute("data-cluster-status");
      // Propagate flag is honoured only when status === 'fp' (the
      // cluster-status propagation endpoint requires no corrected
      // value); other verdicts ignore it because the popover doesn't
      // collect a corrected value to FN-spread to siblings.
      const propagate = !!pop.querySelector("[data-field='propagate-same-value']")?.checked;
      await submitClusterStatusForCell(cid, status, {
        propagate, anchorValue: cellValueRaw,
      });
      closeCellPopover();
    })
  );
  pop.querySelectorAll("[data-cell-status]").forEach((b) =>
    b.addEventListener("click", async () => {
      const status = b.getAttribute("data-cell-status");
      const note = pop.querySelector("[data-field='note']")?.value || "";
      const corrected = pop.querySelector("[data-field='corrected_value']")?.value || "";
      // Propagation is only offered (and honoured) for FN actions; the
      // backend endpoint always writes FN labels, so 'fp' (Clear FN)
      // and other statuses ignore this flag.
      const propagate = !!pop.querySelector("[data-field='propagate-same-value']")?.checked;
      await submitCellLabel(source, row, col, status, {
        note, corrected, propagate, anchorValue: cellValueRaw,
      });
      closeCellPopover();
    })
  );

  // Close on outside click / escape.
  setTimeout(() => {
    document.addEventListener("click", outsideClick, true);
    document.addEventListener("keydown", popoverKey, true);
  }, 0);
  function outsideClick(ev) {
    if (!pop.contains(ev.target) && ev.target !== anchorEl) cleanup();
  }
  function popoverKey(ev) {
    if (ev.key === "Escape") cleanup();
  }
  function cleanup() {
    document.removeEventListener("click", outsideClick, true);
    document.removeEventListener("keydown", popoverKey, true);
    closeCellPopover();
  }
}

async function submitCellLabel(source, row, col, status, { note, corrected, propagate, anchorValue }) {
  try {
    let propagateResult = null;
    await api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cell",
        target_id: `cell:${source}:${row}:${col}`,  // server canonicalises
        source, row, column: col,
        status,
        corrected_value: (corrected || "").trim() || null,
        note: note || "",
      }),
    });
    // Same-value propagation: when the operator marks a cell as FN
    // and supplies a corrected value, mirror the FN onto every other
    // row in this column whose value equals the anchor's. The
    // backend skips cells already covered by a cluster verdict or an
    // existing cell label, so we never silently overwrite prior work.
    // Guard rails: only on FN, only when corrected is non-empty, and
    // only when the anchor cell itself has a non-blank value (empty
    // anchors would match every blank sibling and FN-spam the column).
    let propagatedCount = 0;
    const trimmedCorrected = (corrected || "").trim();
    const anchorTrimmed = (anchorValue || "").trim();
    if (propagate && status === "fn" && trimmedCorrected && anchorTrimmed) {
      try {
        const r = await api(`${API}/labels/propagate-same-value`, {
          method: "POST",
          body: JSON.stringify({
            source, row, column: col,
            corrected_value: trimmedCorrected,
            note: "(auto: same-value as labelled FN cell)",
          }),
        });
        propagatedCount = r.propagated || 0;
        propagateResult = r;
      } catch (e) {
        // Non-fatal: the primary cell label already saved.
        toast(`propagation failed: ${e.message}`, 3000);
      }
    } else if (propagate && status === "fn" && trimmedCorrected && !anchorTrimmed) {
      toast("skip propagation: anchor cell is empty");
    }
    const parts = [`saved: cell ${status}`];
    if (propagatedCount) parts.push(`+${propagatedCount} fn auto-applied`);
    toast(parts.join(" \u00B7 "));
    // Local cell-state update + immediate repaint so the operator
    // sees the new colour without waiting for a /cells round-trip.
    // The status ``fn`` (and ``fp`` to clear an FN) are the only
    // verdicts the backend accepts on a cell label.
    bumpCellStateLocal(source, row, col, status, {
      label_source: "cell",
      note: note || "",
      corrected_value: (corrected || "").trim() || null,
    });
    if (propagateResult && Array.isArray(propagateResult.cells)) {
      for (const pc of propagateResult.cells) {
        bumpCellStateLocal(source, pc.row, pc.column, "fn", {
          label_source: "cell",
          corrected_value: trimmedCorrected,
          note: "(auto: same-value as labelled FN cell)",
        });
      }
    }
    if (state.currentSource === source) repaintCellStates(source);
    scheduleStatsRefresh();
    scheduleSourceMetaRefresh();
    // Background reconciliation: catch any shadow/precedence edge
    // cases the local mirror missed. Not awaited - the UI is
    // already in a good state.
    loadCellState(source, { force: true })
      .then(() => { if (state.currentSource === source) repaintCellStates(source); })
      .catch(() => {});
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  }
}

async function submitClusterStatusForCell(clusterId, status, opts = {}) {
  if (!clusterId) return;
  const { propagate = false, anchorValue = "" } = opts;
  try {
    // Re-fetch the cluster so we have its source/row/columns for the
    // payload (we may not be looking at it in the detail pane).
    const bundle = await api(`${API}/clusters/${encodeURIComponent(clusterId)}`);
    const c = bundle.cluster;
    const cols = Array.isArray(c.columns) ? c.columns.filter(Boolean) : [];
    await api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: clusterId,
        source: c.source,
        row: c.row,
        column: cols[0] || null,
        columns: cols,
        status,
        corrected_value: null,
        note: "",
      }),
    });
    // Same-value propagation: only fp from the popover (the popover
    // doesn't collect a corrected value, so we can't FN-spread on
    // tp/partial). Skip when the seed cell was empty - the column
    // would match every blank sibling and FP-spam other clusters.
    let propagated = 0;
    let propagateResult = null;
    const anchorTrimmed = (anchorValue || "").trim();
    if (
      propagate
      && status === "fp"
      && c.source && Number.isInteger(c.row) && (cols[0] || null)
      && anchorTrimmed
    ) {
      try {
        const r = await api(`${API}/clusters/propagate-status`, {
          method: "POST",
          body: JSON.stringify({
            source: c.source,
            row: c.row,
            column: cols[0],
            status: "fp",
            note: `(auto: same-value as ${clusterId})`,
          }),
        });
        propagated = r.propagated || 0;
        propagateResult = r;
        // Repaint queue dots for siblings cascaded by the backend.
        applyPropagatedClustersToQueue(r.clusters || [], "fp");
      } catch (e) {
        toast(`propagation failed: ${e.message}`, 3000);
      }
    }
    const parts = [`cluster saved: ${status}`];
    if (propagated) parts.push(`+${propagated} fp auto-applied`);
    toast(parts.join(" \u00B7 "));
    // If the cluster being edited is also the active queue selection,
    // reflect the new verdict in the queue dot + verdict buttons.
    const qi = state.queue.findIndex((q) => q.cluster_id === clusterId);
    if (qi >= 0) {
      state.queue[qi].label_status = status;
      const row = document.querySelector(`#queue .row[data-queue-index="${qi}"]`);
      if (row) row.querySelector(".dot").className = "dot " + status;
    }
    updateGroupCounts();
    if (state.cluster?.cluster?.cluster_id === clusterId) {
      document.querySelectorAll("#verdict-buttons button").forEach((b) => {
        b.classList.toggle("active", b.dataset.status === status);
      });
    }
    // Local cell-state mirror: bump the seed cluster's cells AND any
    // siblings the backend just cascaded, so the CSV grid repaints
    // immediately without a /cells GET.
    if (c.source && Number.isInteger(c.row)) {
      for (const col of cols) {
        bumpCellStateLocal(c.source, c.row, col, status, {
          cluster_ids: [clusterId],
          error_types: c.error_types || [],
          label_source: "cluster",
        });
      }
    }
    if (propagateResult && Array.isArray(propagateResult.clusters)) {
      for (const pc of propagateResult.clusters) {
        if (!pc || !Number.isInteger(pc.row)) continue;
        for (const pcol of (pc.columns || [])) {
          bumpCellStateLocal(c.source, pc.row, pcol, status, {
            cluster_ids: [pc.cluster_id],
            label_source: "cluster",
          });
        }
      }
    }
    if (state.currentSource === c.source) repaintCellStates(c.source);
    scheduleStatsRefresh();
    scheduleSourceMetaRefresh();
    // Background reconciliation - non-blocking.
    loadCellState(c.source, { force: true })
      .then(() => { if (state.currentSource === c.source) repaintCellStates(c.source); })
      .catch(() => {});
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  }
}

async function jumpToSource(name) {
  // Find the first cluster on the target source in the current queue;
  // fall back to refreshing with source filter pre-set.
  let idx = state.queue.findIndex((c) => c.source === name);
  if (idx < 0) {
    state.filters.source = [name];
    const ctrl = _multiFilters.get(document.getElementById("filter-source"));
    if (ctrl) ctrl.setSelected([name]);
    await refreshQueue({ keepSelection: false });
    idx = state.queue.findIndex((c) => c.source === name);
  }
  if (idx >= 0) selectIndex(idx);
}

// ----------------------------------------------------------- labelling

// When the operator skips one cluster, cascade the verdict to every
// other still-unlabeled cluster on the same (source, row) so the whole
// row is dismissed in one keystroke. Sibling clusters that already
// carry a verdict (tp/fp/partial/unsure) are left alone to avoid
// clobbering prior work.
async function skipRowSiblings(currentCluster) {
  const src = currentCluster.source;
  const row = currentCluster.row;
  if (src == null || row == null) return 0;
  const siblings = state.queue.filter((q) =>
    q.cluster_id !== currentCluster.cluster_id &&
    q.source === src &&
    q.row === row &&
    (!q.label_status || q.label_status === "unlabeled")
  );
  if (!siblings.length) return 0;

  const results = await Promise.allSettled(siblings.map((sib) => {
    const sibCols = Array.isArray(sib.columns) ? sib.columns.filter(Boolean) : [];
    return api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: sib.cluster_id,
        source: sib.source,
        row: sib.row,
        column: sibCols[0] || null,
        columns: sibCols,
        status: "skip",
        corrected_value: null,
        note: "(auto-skipped: row skipped)",
      }),
    }).then((rec) => ({ sib, rec }));
  }));

  let n = 0;
  for (const r of results) {
    if (r.status !== "fulfilled") continue;
    const { sib, rec } = r.value;
    state.latest[sib.cluster_id] = rec;
    const idx = state.queue.findIndex((q) => q.cluster_id === sib.cluster_id);
    if (idx >= 0) {
      state.queue[idx].label_status = "skip";
      const el = document.querySelector(`#queue .row[data-queue-index="${idx}"]`);
      const dot = el?.querySelector(".dot");
      if (dot) dot.className = "dot skip";
    }
    n += 1;
  }
  if (n > 0) updateGroupCounts();
  return n;
}

// ---------------------------------------------------- multi-table render

// Render the side-by-side row panels for a multi-table finding. Logic
// is lifted from the legacy mt_label.js (renderCompare): walk the
// group's column_mapping (canonical -> per-sheet raw name) so two
// sheets with different headers ("country" vs "Country") still align
// to the same table row, then mark cells that disagree across panels.
function renderMtCompare(d) {
  const compare = $("#mt-compare");
  if (!d.rows || d.rows.length === 0) {
    compare.innerHTML = `<div class="placeholder">No row data available.</div>`;
    return;
  }
  const flagged = (d.rows[0]?.flagged_columns || []).slice();
  const pk = (d.group?.primary_key || []).slice();
  const colMap = d.group?.column_mapping || {};
  const ordered = [];
  const seen = new Set();
  const push = (c) => {
    if (!c || seen.has(c)) return;
    seen.add(c); ordered.push(c);
  };
  flagged.forEach(push);
  pk.forEach(push);
  for (const r of d.rows) {
    if (Array.isArray(r.columns) && r.columns.length) {
      r.columns.forEach((c) => {
        const canonical = mtCanonicalFor(c, r.sheet, colMap);
        push(canonical || c);
      });
    } else {
      Object.keys(r.values || {}).forEach(push);
    }
  }
  const panelData = d.rows.map((r) => ({
    sheet: r.sheet, row: r.row,
    values: r.values || {}, snapshot: r.snapshot || {},
  }));
  const cellByCol = ordered.map((canonical) => ({
    canonical,
    cells: panelData.map((p) => mtValueFor(canonical, p, colMap)),
  }));
  const fragment = document.createDocumentFragment();
  d.rows.forEach((r, paneIdx) => {
    const pane = document.createElement("div");
    pane.className = "mt-pane";
    const head = document.createElement("div");
    head.className = "mt-pane-head";
    head.innerHTML = `
      <span class="mt-pane-sheet">${escape(r.sheet || "?")}</span>
      <span class="mt-pane-row">row ${r.row != null && r.row >= 0 ? r.row : "?"}</span>
    `;
    pane.appendChild(head);
    if (!cellByCol.length) {
      const empty = document.createElement("div");
      empty.className = "mt-pane-empty";
      empty.textContent = "(no comparable columns)";
      pane.appendChild(empty);
    } else {
      const tbl = document.createElement("table");
      tbl.className = "mt-pane-table";
      const tbody = document.createElement("tbody");
      const flaggedSet = new Set(flagged);
      const pkSet = new Set(pk);
      for (const { canonical, cells } of cellByCol) {
        const tr = document.createElement("tr");
        if (flaggedSet.has(canonical)) tr.classList.add("flagged");
        const th = document.createElement("th");
        th.textContent = canonical;
        if (pkSet.has(canonical)) th.classList.add("pk");
        tr.appendChild(th);
        const myVal = cells[paneIdx];
        const td = document.createElement("td");
        td.textContent = myVal == null ? "—" : myVal;
        const norms = cells
          .filter((v) => v != null && String(v).trim() !== "")
          .map((v) => String(v).trim().toLowerCase());
        const distinct = new Set(norms);
        if (distinct.size > 1) td.classList.add("differ");
        else td.classList.add("agree");
        tr.appendChild(td);
        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      pane.appendChild(tbl);
    }
    fragment.appendChild(pane);
  });
  compare.replaceChildren(fragment);
}

function mtCanonicalFor(rawCol, sheet, colMap) {
  if (!colMap) return null;
  for (const [canonical, perSheet] of Object.entries(colMap)) {
    if (perSheet && perSheet[sheet] === rawCol) return canonical;
  }
  return null;
}

function mtValueFor(canonical, panel, colMap) {
  const perSheet = colMap?.[canonical];
  const rawCol = (perSheet && perSheet[panel.sheet]) || canonical;
  if (Object.prototype.hasOwnProperty.call(panel.values, rawCol)) return panel.values[rawCol];
  if (Object.prototype.hasOwnProperty.call(panel.values, canonical)) return panel.values[canonical];
  if (Object.prototype.hasOwnProperty.call(panel.snapshot, rawCol)) return panel.snapshot[rawCol];
  if (Object.prototype.hasOwnProperty.call(panel.snapshot, canonical)) return panel.snapshot[canonical];
  return null;
}

// Render the right-hand detail pane for a multi-table finding.
// Re-uses the cluster-vocab verdict buttons (tp / fp / partial /
// unsure) and writes through the cluster-vs-MT mapping in
// submitLabel. ``skip`` is intentionally absent because the MT label
// schema has no equivalent verdict.
function renderMtDetail(d, item) {
  const el = $("#detail");
  const linked = (d.rows || []).map(
    (r) => `${r.sheet}:${r.row != null && r.row >= 0 ? r.row : "?"}`,
  ).join(" \u2194 ");
  const flagged = ((d.rows[0] || {}).flagged_columns || []).join(", ");
  const labelStatus = d.label?.status || "";
  const displayedStatus = MT_TO_CLUSTER_STATUS[labelStatus] || "";
  const btn = (s, key) => `<button data-status="${s}" class="${s}${displayedStatus === s ? " active" : ""}" title="key ${key}">${s} <kbd>${key}</kbd></button>`;
  const llm = d.llm_verdict || {};
  const llmHtml = llm.verdict ? `
    <div class="block">
      <h2>LLM verdict</h2>
      <div>
        <span class="sev-chip">${escape((llm.verdict || "").replace(/_/g, " "))}</span>
        ${llm.confidence ? `<span class="muted" style="margin-left:6px;">${escape(llm.confidence)}</span>` : ""}
      </div>
      ${llm.rationale ? `<div style="margin-top:6px;">${escape(llm.rationale)}</div>` : ""}
    </div>` : "";
  const groupHtml = d.group ? `
    <div class="block">
      <h2>Group context</h2>
      ${(d.group.primary_key || []).length
        ? `<div class="muted">primary key: ${escape((d.group.primary_key || []).join(", "))}</div>` : ""}
      ${(d.group.sheets || []).length
        ? `<div class="muted">sheets: ${escape((d.group.sheets || []).join(", "))}</div>` : ""}
      ${d.group.rationale ? `<div style="margin-top:6px;">${escape(d.group.rationale)}</div>` : ""}
    </div>` : "";
  el.innerHTML = `
    <div class="block">
      <h2>${escape(d.violation_id)} <span class="sev-chip" title="cross-sheet finding">cross-sheet</span></h2>
      <div class="cols">${escape(d.workbook || "?")}${
        d.group_kind ? ` · group ${escape(d.group_kind)}` : ""
      }${d.kind ? ` · ${escape(d.kind)}` : ""}</div>
      <div class="cols" style="margin-top:4px;">rows: ${escape(linked)}</div>
      ${flagged ? `<div class="cols">columns: ${escape(flagged)}</div>` : ""}
      ${d.rationale ? `<div style="margin-top:6px;">${escape(d.rationale)}</div>` : ""}
    </div>

    ${llmHtml}

    <div class="block">
      <h2>Verdict</h2>
      <div class="buttons" id="verdict-buttons">
        ${btn("tp","1")}${btn("fp","2")}${btn("partial","3")}${btn("unsure","4")}
      </div>
      <div class="muted" style="font-size:11px; margin-top:4px;">
        tp = confirmed inconsistency · fp = not a violation · partial = benign variation
      </div>
      <label class="field">
        <span>Note</span>
        <textarea id="note">${escape(d.label?.note || "")}</textarea>
      </label>
      ${d.label ? `<div class="muted" style="margin-top:6px; font-size:11px;">last: ${escape(d.label.status)} by ${escape(d.label.annotator)} @ ${escape(d.label.ts)}</div>` : ""}
    </div>

    ${groupHtml}
  `;
  el.querySelectorAll("#verdict-buttons button").forEach((b) => {
    b.addEventListener("click", () => submitLabel(b.dataset.status, { advance: false }));
  });
}

// POST a multi-table label using the cluster-vocab status the user
// pressed. Mirrors the legacy mt_label.js submit() path but updates
// the unified queue + dot the same way submitLabel does for clusters.
async function submitMtLabel(displayStatus, { advance }) {
  const wrapper = state.cluster;
  if (!wrapper || !wrapper._mt) return;
  const d = wrapper.mt;
  const mtStatus = CLUSTER_TO_MT_STATUS[displayStatus];
  if (!mtStatus) {
    toast(`'${displayStatus}' is not a valid cross-sheet verdict (use tp/fp/partial/unsure)`);
    return;
  }
  if (submitLabel._inFlight) return;
  submitLabel._inFlight = true;
  document.body.classList.add("saving");
  const note = ($("#note")?.value || "").trim();
  try {
    const res = await api(`${API}/multi-table-labels`, {
      method: "POST",
      body: JSON.stringify({
        violation_id: d.violation_id,
        workbook: d.workbook || "",
        group_id: d.group_id || "",
        kind: d.kind || "",
        status: mtStatus,
        note,
      }),
    });
    d.label = res.label;
    d.history = res.history;
    const q = state.queue[state.index];
    if (q) {
      q.label_status = displayStatus;
    }
    const row = document.querySelector(`#queue .row[data-queue-index="${state.index}"]`);
    if (row) {
      const dot = row.querySelector(".dot");
      if (dot) dot.className = "dot " + displayStatus;
    }
    updateGroupCounts();
    document.querySelectorAll("#verdict-buttons button").forEach((b) => {
      b.classList.toggle("active", b.dataset.status === displayStatus);
    });
    toast(`saved: ${displayStatus} (${mtStatus.replace(/_/g, " ")})`);
    if (advance) selectIndex(nextUnlabeledIndex(state.index + 1));
  } catch (e) {
    toast(`save failed: ${e.message}`, 3000);
  } finally {
    submitLabel._inFlight = false;
    document.body.classList.remove("saving");
    // If the operator pressed Enter while the digit's POST was
    // still in flight, honour that queued advance now that the
    // save has settled. Skip when this call already advanced -
    // otherwise mashing Enter would skip two clusters per save.
    if (submitLabel._pendingAdvance) {
      submitLabel._pendingAdvance = false;
      if (!advance) selectIndex(nextUnlabeledIndex(state.index + 1));
    }
  }
}

async function submitLabel(status, { advance = true } = {}) {
  if (!state.cluster) return;
  // Multi-table findings use a separate label store + verdict
  // vocabulary; route them through the dedicated POST path. The
  // ``skip`` button is hidden in the MT detail pane so we don't expect
  // it here, but we guard against bound keyboard shortcuts firing it.
  if (state.cluster._mt) {
    if (status === "skip") {
      toast("skip is not available for cross-sheet findings");
      return;
    }
    return submitMtLabel(status, { advance });
  }
  // Re-entrancy guard: a fast operator can press Enter (or 1-5) again
  // while the previous POST is still in flight, which would race two
  // /api/labels writes on the same cluster and double-fire the
  // same-value propagation. We drop the second click silently - the
  // verdict for the same cluster is idempotent and `advance` will
  // happen at the end of the first call anyway.
  if (submitLabel._inFlight) return;
  submitLabel._inFlight = true;
  document.body.classList.add("saving");
  const c = state.cluster.cluster;
  const corrected = ($("#corrected-value")?.value ?? "").trim();
  const note = ($("#note")?.value ?? "");
  try {
    const cols = Array.isArray(c.columns) ? c.columns.filter(Boolean) : [];
    // Mirror the scalar correction across every column the cluster
    // covers so the per-column map stays the source of truth even
    // when the UI only collects one input.
    const correctedMap = (corrected && cols.length)
      ? Object.fromEntries(cols.map((col) => [col, corrected]))
      : {};
    const record = await api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: c.cluster_id,
        source: c.source,
        row: c.row,
        column: cols[0] || null,
        columns: cols,
        status,
        corrected_value: corrected || null,
        corrected_values: correctedMap,
        note,
      }),
    });
    state.latest[c.cluster_id] = record;
    // Update the queue dot without a round-trip.
    const q = state.queue[state.index];
    if (q) q.label_status = status;
    const row = document.querySelector(`#queue .row[data-queue-index="${state.index}"]`);
    if (row) {
      const dot = row.querySelector(".dot");
      dot.className = "dot " + status;
    }
    updateGroupCounts();
    document.querySelectorAll("#verdict-buttons button").forEach((b) => {
      b.classList.toggle("active", b.dataset.status === status);
    });
    // Skip cascades to the whole row: when the operator marks one
    // cluster as ``skip``, every other cluster on the same (source, row)
    // that is still unlabeled also gets skipped, so we can move past
    // an unusable row in a single click.
    let cascaded = 0;
    if (status === "skip") {
      cascaded = await skipRowSiblings(c);
    }
    // Same-value propagation: branch on verdict.
    //   TP / Partial: walk the column and FN-label other rows holding
    //                 the identical erroneous value (the LLM missed
    //                 those siblings). Requires a corrected_value.
    //   FP:          walk the column and FP every OTHER cluster on
    //                 (source, column) whose anchor cell value matches
    //                 - if "AcmeCo" is correct here, it's correct
    //                 wherever else the LLM flagged it. No corrected
    //                 value needed.
    //   Unsure / Skip: never propagate (uncertainty shouldn't spread;
    //                  skip has its own row-cascade above).
    let propagated = 0;
    let propagateFnResult = null;
    let propagateClusterResult = null;
    const propagateChk = document.getElementById("propagate-same-value");
    const wantPropagate = !!(propagateChk && propagateChk.checked);
    const firstCol = (c.columns || [])[0];
    if (
      wantPropagate
      && (status === "tp" || status === "partial")
      && c.source && Number.isInteger(c.row) && firstCol
      && corrected
    ) {
      try {
        const r = await api(`${API}/labels/propagate-same-value`, {
          method: "POST",
          body: JSON.stringify({
            source: c.source,
            row: c.row,
            column: firstCol,
            corrected_value: corrected,
            note: `(auto: same-value as ${c.cluster_id})`,
          }),
        });
        propagated = r.propagated || 0;
        propagateFnResult = r;
      } catch (e) {
        // Non-fatal: the cluster verdict is already saved.
        toast(`propagation failed: ${e.message}`, 3000);
      }
    } else if (
      wantPropagate
      && status === "fp"
      && c.source && Number.isInteger(c.row) && firstCol
    ) {
      try {
        const r = await api(`${API}/clusters/propagate-status`, {
          method: "POST",
          body: JSON.stringify({
            source: c.source,
            row: c.row,
            column: firstCol,
            status: "fp",
            note: `(auto: same-value as ${c.cluster_id})`,
          }),
        });
        propagated = r.propagated || 0;
        propagateClusterResult = r;
        // Repaint queue dots for siblings cascaded by the backend so
        // the sidebar reflects their new fp verdict without a reload.
        applyPropagatedClustersToQueue(r.clusters || [], "fp");
      } catch (e) {
        toast(`propagation failed: ${e.message}`, 3000);
      }
    }
    const parts = [`saved: ${status}`];
    if (cascaded) parts.push(`+${cascaded} skipped on row`);
    if (propagated) {
      const verb = status === "fp" ? "fp auto-applied" : "fn auto-applied";
      parts.push(`+${propagated} ${verb}`);
    }
    toast(parts.join(" · "));
    scheduleStatsRefresh();
    scheduleSourceMetaRefresh();
    // Local cell-state mirror so the CSV grid repaints synchronously
    // (the awaited /cells GET is now dispatched in the background).
    if (c.source && Number.isInteger(c.row)) {
      const cols = Array.isArray(c.columns) ? c.columns.filter(Boolean) : [];
      for (const col of cols) {
        bumpCellStateLocal(c.source, c.row, col, status, {
          cluster_ids: [c.cluster_id],
          error_types: c.error_types || [],
          note,
          corrected_value: corrected || null,
          label_source: "cluster",
        });
      }
      if (propagateFnResult && Array.isArray(propagateFnResult.cells)) {
        for (const pc of propagateFnResult.cells) {
          bumpCellStateLocal(c.source, pc.row, pc.column, "fn", {
            label_source: "cell",
            corrected_value: corrected || null,
          });
        }
      }
      if (propagateClusterResult && Array.isArray(propagateClusterResult.clusters)) {
        for (const pc of propagateClusterResult.clusters) {
          if (!pc || !Number.isInteger(pc.row)) continue;
          for (const pcol of (pc.columns || [])) {
            bumpCellStateLocal(c.source, pc.row, pcol, "fp", {
              cluster_ids: [pc.cluster_id],
              label_source: "cluster",
            });
          }
        }
      }
      if (state.currentSource === c.source) repaintCellStates(c.source);
      // Background reconciliation - non-blocking.
      loadCellState(c.source, { force: true })
        .then(() => { if (state.currentSource === c.source) repaintCellStates(c.source); })
        .catch(() => {});
    }
    if (advance) selectIndex(nextUnlabeledIndex(state.index + 1));
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  } finally {
    submitLabel._inFlight = false;
    document.body.classList.remove("saving");
    // Honour an Enter press that arrived during this in-flight
    // save (the keydown handler set ``_pendingAdvance`` instead
    // of firing a second POST that would race ours). Skip when
    // this call already advanced - otherwise mashing Enter would
    // skip two clusters per save.
    if (submitLabel._pendingAdvance) {
      submitLabel._pendingAdvance = false;
      if (!advance) selectIndex(nextUnlabeledIndex(state.index + 1));
    }
  }
}

// ----------------------------------------------------------- keyboard

// Helper: paint the verdict-button .active state synchronously so
// the next Enter press (which reads .active) sees the user's most
// recent choice without waiting for the POST that ``submitLabel``
// fires off to round-trip and complete. Without this, pressing 1
// then Enter quickly would race: Enter fires before submitLabel's
// post-await ``classList.toggle`` runs, so it sees no .active and
// silently does nothing - the operator then has to press Enter a
// second time after the save settles.
function setActiveVerdictButton(status) {
  const buttons = document.querySelectorAll("#verdict-buttons button");
  if (!buttons.length) return;
  buttons.forEach((b) => {
    b.classList.toggle("active", b.dataset.status === status);
  });
}

function onKeydown(e) {
  const t = e.target;
  const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
  if (e.key === "Escape" && typing) { t.blur(); return; }
  if (typing) return;
  // Don't swallow OS shortcuts (Ctrl+C copy, Ctrl+V paste, Ctrl+F find,
  // Cmd+R reload, Alt+Tab, etc.) - only single-key bindings are ours.
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  const map = { "1": "tp", "2": "fp", "3": "partial", "4": "unsure", "5": "skip" };
  if (map[e.key]) {
    // Paint .active synchronously so a fast follow-up Enter sees
    // the right verdict (the post-await paint inside submitLabel
    // doesn't run until the POST resolves).
    setActiveVerdictButton(map[e.key]);
    submitLabel(map[e.key], { advance: false });
    e.preventDefault();
    return;
  }
  if (e.key === "Enter") {
    // Enter saves with the currently-selected verdict (or tp default
    // when nothing is active yet) and advances. Three cases:
    //   (a) A previous submitLabel from a digit press is still in
    //       flight: don't fire a second POST that would race the
    //       first - just queue the advance, ``submitLabel``'s finally
    //       block will pick it up.
    //   (b) An ``.active`` button exists (operator pressed a digit
    //       OR the cluster already has a saved verdict): submit it
    //       with advance.
    //   (c) Nothing active and nothing in flight: default to tp so
    //       Enter is never a silent no-op (matches the keyboard hint
    //       in the footer).
    if (submitLabel._inFlight) {
      submitLabel._pendingAdvance = true;
      e.preventDefault();
      return;
    }
    let status = document
      .querySelector("#verdict-buttons button.active")?.dataset.status;
    if (!status) {
      status = "tp";
      setActiveVerdictButton(status);
    }
    submitLabel(status, { advance: true });
    e.preventDefault();
    return;
  }
  if (e.key === "j") { selectIndex(nextUnlabeledIndex(state.index + 1)); e.preventDefault(); }
  else if (e.key === "k") { selectIndex(state.index - 1); e.preventDefault(); }
  else if (e.key === "c") { $("#corrected-value")?.focus(); e.preventDefault(); }
  else if (e.key === "n") { $("#note")?.focus(); e.preventDefault(); }
}

// ----------------------------------------------------------- util

function escape(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// Render a CSV cell value as HTML, surfacing whitespace anomalies so
// the operator can spot them at a glance.
//
// Bugs we flag:
//   - Leading or trailing regular spaces ("  Smith" / "Smith  ").
//   - Runs of >= 2 consecutive internal spaces ("Smith  John").
//   - Tabs (\t) - rare in CSV cells, almost always a paste bug.
//   - Non-breaking space (\u00A0) - looks like a space but isn't.
//   - Newlines / carriage returns (\n, \r, \u2028, \u2029) inside a
//     "single-line" cell - usually a stray import.
//   - Zero-width characters (\u200B, \u200C, \u200D, \u00AD, \uFEFF)
//     - completely invisible without a marker.
//
// Rendering rules:
//   - Regular spaces are PRESERVED as space characters inside a marker
//     span so copy/paste still yields the literal whitespace - we
//     just paint a colored background to make their location visible.
//   - Other special characters are REPLACED with a glyph (->, NBSP,
//     newline arrow, dot) inside a marker span, since the underlying
//     character has no visible width and a faithful copy is rarely
//     what the operator wants.
//   - Single internal spaces are left unchanged so well-formed cells
//     don't get visually polluted.
//
// The caller is expected to insert the result via innerHTML.
function renderCsvCellHtml(raw) {
  const s = raw == null ? "" : String(raw);
  if (s === "") return "";
  // Special non-space whitespace: replaced with a visible glyph.
  // ``cls`` selects the colour scheme; ``title`` is the tooltip the
  // operator sees on hover. ``preserve`` says "keep the original char
  // in the DOM so copy/paste survives" - we currently use the glyph
  // for everything in this map, since these characters have no
  // visible width and a faithful round-trip via copy is rarely the
  // operator's intent.
  const SPECIAL = {
    "\t":     { glyph: "\u2192",       cls: "ws-special",  title: "tab" },
    "\n":     { glyph: "\u21B5",       cls: "ws-newline",  title: "newline (\\n)" },
    "\r":     { glyph: "\u240D",       cls: "ws-newline",  title: "carriage return (\\r)" },
    "\u00A0": { glyph: "\u2423",       cls: "ws-special",  title: "non-breaking space (NBSP)" },
    "\u200B": { glyph: "\u00B7",       cls: "ws-zw",       title: "zero-width space (ZWSP)" },
    "\u200C": { glyph: "\u00B7",       cls: "ws-zw",       title: "zero-width non-joiner (ZWNJ)" },
    "\u200D": { glyph: "\u00B7",       cls: "ws-zw",       title: "zero-width joiner (ZWJ)" },
    "\u00AD": { glyph: "\u00B7",       cls: "ws-zw",       title: "soft hyphen" },
    "\uFEFF": { glyph: "\u00B7",       cls: "ws-zw",       title: "byte-order mark (BOM)" },
    "\u2028": { glyph: "\u21B5",       cls: "ws-newline",  title: "line separator (U+2028)" },
    "\u2029": { glyph: "\u21B5",       cls: "ws-newline",  title: "paragraph separator (U+2029)" },
  };

  // Locate leading + trailing runs of regular spaces so we can mark
  // them differently from internal multi-space runs (edge whitespace
  // is almost always a bug; internal multi-space is suspicious).
  const leadMatch = s.match(/^[ ]+/);
  const trailMatch = s.match(/[ ]+$/);
  const leadEnd = leadMatch ? leadMatch[0].length : 0;
  // Avoid marking the trailing run twice when the whole cell is
  // spaces (leadEnd would already cover it).
  const trailStart = trailMatch
    ? Math.max(s.length - trailMatch[0].length, leadEnd)
    : s.length;

  const escChar = (ch) => {
    if (ch === "&") return "&amp;";
    if (ch === "<") return "&lt;";
    if (ch === ">") return "&gt;";
    if (ch === '"') return "&quot;";
    if (ch === "'") return "&#39;";
    return ch;
  };

  let out = "";
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === " ") {
      // Coalesce a contiguous run of regular spaces into a single
      // span so very long runs render with one tooltip rather than
      // N nested spans.
      let j = i;
      while (j < s.length && s[j] === " ") j++;
      const run = j - i;
      const isLeading = i < leadEnd;
      const isTrailing = i >= trailStart;
      const isEdge = isLeading || isTrailing;
      // Internal single space - render as a literal space, no marker.
      // This is what the original ``escape(value)`` would have done;
      // we keep it that way for healthy cells so the view stays calm.
      if (!isEdge && run === 1) {
        out += " ";
        i = j;
        continue;
      }
      const cls = isEdge
        ? (isLeading && isTrailing ? "ws-edge ws-edge-both"
           : isLeading ? "ws-edge ws-edge-leading"
           : "ws-edge ws-edge-trailing")
        : "ws-internal";
      const where = isLeading ? "leading" : isTrailing ? "trailing" : "internal";
      const title = `${run} ${run === 1 ? "space" : "spaces"} (${where})`;
      // Keep the literal space chars inside the span so the operator
      // can copy "Smith   " and paste it intact. The colored
      // background reveals the location.
      out += `<span class="ws-marker ${cls}" title="${title}">${" ".repeat(run)}</span>`;
      i = j;
      continue;
    }
    const def = SPECIAL[ch];
    if (def) {
      out += `<span class="ws-marker ${def.cls}" title="${def.title}">${def.glyph}</span>`;
      i++;
      continue;
    }
    out += escChar(ch);
    i++;
  }
  return out;
}

bootstrap().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#f87171;">bootstrap failed: ${escape(e.stack || e.message)}</pre>`;
});
