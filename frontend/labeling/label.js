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
  filters: { source: "", status: "", error_type: "" },
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

// ----------------------------------------------------------- bootstrap

async function bootstrap() {
  const run = await api(`${API}`);
  state.run = run;

  document.getElementById("export-link").href = `${API}/export/benchmark`;

  $("#run-meta").textContent =
    `${RUN_ID} · annotator ${run.annotator} · ` +
    `${run.counts.clusters} clusters · ${run.counts.issues} raw issues`;

  // Populate source filter. For multi-workbook runs (e.g. curated_pbi /
  // Bing21 reconstructed) group sources under <optgroup> tags by their
  // parent-dir basename, and prepend each group with an "(all in <wb>)"
  // entry whose value is the synthetic ``workbook:<wb>`` token. The
  // queue-fetch translates that token into a ``?workbook=`` query so
  // every CSV in the workbook is shown together (essential for
  // inter-table errors).
  const srcSel = $("#filter-source");
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
  // ``optionFor`` formats one <option> for a source with a checkmark
  // prefix so the operator can see at a glance which CSVs they have
  // already touched. ``\u2713`` = touched (cluster verdict OR FN cell
  // exists), ``\u25CB`` = untouched. The ``(N)`` count shows clusters;
  // ``\u2205`` flags CSVs the LLM never flagged (zero clusters) so
  // the operator knows they need to browse for FN candidates.
  const optionFor = (s) => {
    const o = document.createElement("option");
    o.value = s.name;
    const mark = s.checked ? "\u2713" : "\u25CB";
    const cnt = s.clusters > 0 ? `(${s.clusters})` : "(\u2205)";
    const fn = s.fn_cells ? ` \u00B7 fn:${s.fn_cells}` : "";
    o.textContent = `${mark} ${shortSource(s.name)} ${cnt}${fn}`;
    o.title = s.name;
    o.dataset.checked = s.checked ? "1" : "0";
    o.dataset.clusters = String(s.clusters);
    return o;
  };
  if (multiWb && Array.isArray(run.workbooks) && run.workbooks.length) {
    const grouped = new Set();
    for (const wb of run.workbooks) {
      const og = document.createElement("optgroup");
      const wbChecked = wb.sources.filter(
        (n) => run.sources.find((x) => x.name === n)?.checked
      ).length;
      og.label = `${wb.name}  \u2014  ${wbChecked}/${wb.sources.length} checked, ${wb.clusters} clusters`;
      const allOpt = document.createElement("option");
      allOpt.value = `workbook:${wb.name}`;
      allOpt.textContent = `(all in ${wb.name})`;
      og.appendChild(allOpt);
      for (const name of wb.sources) {
        const s = run.sources.find((x) => x.name === name);
        if (!s) continue;
        og.appendChild(optionFor(s));
        grouped.add(name);
      }
      srcSel.appendChild(og);
    }
    // Anything not in a multi-CSV workbook (singletons) goes under a
    // catch-all optgroup so they remain reachable.
    const orphans = run.sources.filter((s) => !grouped.has(s.name));
    if (orphans.length) {
      const og = document.createElement("optgroup");
      og.label = "(other)";
      for (const s of orphans) og.appendChild(optionFor(s));
      srcSel.appendChild(og);
    }
  } else {
    for (const s of run.sources) srcSel.appendChild(optionFor(s));
  }

  // Populate error-type filter from summary.
  const ets = Object.keys(run.summary?.issues_by_error_type || {});
  const etSel = $("#filter-error-type");
  ets.sort().forEach((et) => {
    const o = document.createElement("option");
    o.value = et; o.textContent = et;
    etSel.appendChild(o);
  });

  ["source", "status", "error-type"].forEach((k) => {
    $(`#filter-${k}`).addEventListener("change", (e) => {
      const key = k === "error-type" ? "error_type" : k;
      state.filters[key] = e.target.value;
      if (k === "source") updateSourceXlsxActions();
      if (k === "source") renderLabelingEffort();
      refreshQueue({ keepSelection: false });
    });
  });
  updateSourceXlsxActions();

  await Promise.all([refreshLatest(), refreshStats(), refreshLabelingEffort()]);
  await refreshQueue({ keepSelection: false });

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
  if (!payload || !payload.available) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  state.labelingEffort = payload;

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

  renderLabelingEffort();
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

function effortLookupForCell(source, column) {
  const ef = state.labelingEffort;
  if (!ef || !ef.available || !Array.isArray(ef.cells)) return null;
  // Linear scan is fine: per-arm artifacts are O(num_columns *
  // num_error_types), typically <500 rows. The lookup is also only
  // hit per-cell during renderCsv (~CSV_WINDOW rows).
  const cell = ef.cells.find((c) => c.source === source && c.column === column);
  if (!cell) return null;
  const byEt = new Map();
  for (const e of (cell.by_error_type || [])) {
    byEt.set(e.error_type, e);
  }
  return { cell, byEt };
}

function renderLabelingEffort() {
  const tbody = document.getElementById("effort-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  const cells = (state.labelingEffort && state.labelingEffort.cells) || [];
  if (!cells.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">(no per-column estimates available yet)</td>`;
    tbody.appendChild(tr);
    return;
  }
  // Filter by selected source (matches the queue's filter so the two
  // panels feel coordinated). Workbook-token filters fall through to
  // "no filter" here because the per-arm artifact keys by source CSV.
  const filterSource = state.filters?.source || "";
  const visible = cells.filter((c) =>
    !filterSource || filterSource.startsWith("workbook:") || c.source === filterSource
  );
  if (!visible.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">(no estimates for the current source filter)</td>`;
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
    tr.addEventListener("click", () => {
      const sel = document.getElementById("filter-source");
      if (!sel) return;
      // Only do anything if the option exists (it should: the queue
      // and effort table share the same source list).
      const target = c.source;
      if ([...sel.options].some((o) => o.value === target)) {
        sel.value = target;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
      }
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
    tbody.appendChild(tr);
  }
  if (overflow) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">(\u2026 +${overflow} more cells; widen the source filter to drill in)</td>`;
    tbody.appendChild(tr);
  }
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
  const sel = state.filters.source || "";
  let wb = "";
  let hasXlsx = false;
  if (sel.startsWith("workbook:")) {
    const name = sel.slice("workbook:".length);
    const meta = (state.run.workbooks || []).find((w) => w.name === name);
    if (meta) { wb = meta.name; hasXlsx = !!meta.has_xlsx; }
  } else if (sel) {
    const meta = (state.run.sources || []).find((s) => s.name === sel);
    if (meta) {
      wb = meta.table_group || sel;
      hasXlsx = !!meta.has_xlsx;
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
    const sel = document.getElementById("filter-source");
    if (!sel) return;
    const byName = new Map(fresh.sources.map((s) => [s.name, s]));
    for (const o of sel.querySelectorAll("option")) {
      const meta = byName.get(o.value);
      if (!meta) continue;
      const mark = meta.checked ? "\u2713" : "\u25CB";
      const cnt = meta.clusters > 0 ? `(${meta.clusters})` : "(\u2205)";
      const fn = meta.fn_cells ? ` \u00B7 fn:${meta.fn_cells}` : "";
      o.textContent = `${mark} ${meta.name} ${cnt}${fn}`;
      o.dataset.checked = meta.checked ? "1" : "0";
    }
  } catch (_) { /* non-fatal */ }
}

// ----------------------------------------------------------- queue

async function refreshQueue({ keepSelection }) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) {
    if (!v) continue;
    // Synthetic ``workbook:<name>`` token (added by the optgroup
    // builder for multi-table corpora) maps to the ``workbook=`` query
    // param so every CSV under that workbook is included.
    if (k === "source" && v.startsWith("workbook:")) {
      qs.set("workbook", v.slice("workbook:".length));
    } else {
      qs.set(k, v);
    }
  }
  // Backend cap is 5000; this matches it so the (all)-source view
  // never silently truncates large runs and "unlabeled" counts in the
  // header don't drift away from what's actually in the queue.
  qs.set("limit", "5000");
  const r = await api(`${API}/clusters?${qs.toString()}`);
  state.queue = r.items;
  if (r.total != null && r.total > r.items.length) {
    toast(`queue truncated: ${r.items.length}/${r.total} (apply a filter to see the rest)`, 4000);
  }

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
      srcHdr.innerHTML =
        `<span class="caret">${collapsedSrc ? "\u25B8" : "\u25BE"}</span>` +
        `<span class="label">${escape(src)}</span>` +
        `<span class="count">${headerSummary(items)}</span>`;
      wbBox.appendChild(srcHdr);
      const srcBody = document.createElement("div");
      srcBody.className = "group-body src-body";
      if (collapsedSrc) srcBody.style.display = "none";
      wbBox.appendChild(srcBody);
      srcHdr.addEventListener("click", () => {
        toggleGroup(srcHdr, srcBody, srcKey);
      });

      for (const { c, idx } of items) {
        const row = document.createElement("div");
        row.className = "row" + (c.inferred ? " row-inferred" : "");
        row.dataset.queueIndex = String(idx);
        const status = c.label_status || "unlabeled";
        const inferredBadge = c.inferred
          ? `<span class="inferred-badge" title="No primary detection on this row \u2014 surfaced via a sibling detection's affected_rows. Labelling it captures the LLM's full claim and prevents FN drift.">inf</span>`
          : "";
        row.innerHTML = `
          <div class="dot ${status}"></div>
          <div class="main">
            <div class="src">row ${escape(String(c.row ?? "?"))} ${inferredBadge}</div>
            <div class="cols">${escape((c.columns || []).join(", "))}</div>
            <div class="types">${escape((c.error_types || []).join(" \u00B7 "))}</div>
          </div>
        `;
        row.addEventListener("click", () => selectIndex(idx));
        srcBody.appendChild(row);
      }
    }
  }

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
  const src = state.filters.source;
  if (src && !src.startsWith("workbook:")) {
    enterBrowseMode(src);
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
  loadCluster(c.cluster_id);
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

async function loadCluster(cid) {
  const [r] = await Promise.all([api(`${API}/clusters/${encodeURIComponent(cid)}`)]);
  state.cluster = r;
  renderDetail(r);
  renderCsv(r.cluster);
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

  const dets = (c.detections || []).map((d) => `
    <div class="det-row">
      <div class="et">${escape(d.error_type_name || d.error_type)}</div>
      <div class="obs"><b>Observed:</b> ${escape(d.observed || "")}</div>
      <div class="fix"><b>Explanation:</b> ${escape(d.explanation || "")}</div>
      ${d.suggested_fix ? `<div class="fix"><b>Fix:</b> ${escape(d.suggested_fix)}</div>` : ""}
    </div>
  `).join("");

  const currentStatus = label?.status || "";
  const btn = (s, key) => `<button data-status="${s}" class="${s}${currentStatus === s ? " active" : ""}" title="key ${key}">${s} <kbd>${key}</kbd></button>`;

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

    ${syn.canonical_explanation ? `
      <div class="block">
        <h2>Synthesis</h2>
        <div><b>${escape(syn.primary_error_type || "")}</b></div>
        <div style="margin-top:4px;">${escape(syn.canonical_explanation)}</div>
        ${syn.canonical_fix ? `<div style="margin-top:4px;"><b>Fix:</b> ${escape(syn.canonical_fix)}</div>` : ""}
      </div>
    ` : ""}

    ${renderWebVerificationBlock(bundle)}

    <div class="block">
      <h2>Verdict</h2>
      <div class="buttons" id="verdict-buttons">
        ${btn("tp","1")}${btn("fp","2")}${btn("partial","3")}${btn("unsure","4")}${btn("skip","5")}
      </div>
      <label class="field">
        <span>Corrected value (for tp / partial)</span>
        <input id="corrected-value" type="text" value="${escape(label?.corrected_value || "")}" placeholder="(leave blank if unknown)" />
      </label>
      <label class="field" style="flex-direction: row; align-items: center; gap: 6px;">
        <input id="propagate-same-value" type="checkbox" checked />
        <span style="color: var(--muted); font-size: 11.5px;">
          On TP/partial: also FN-label other rows in this column with the same value
        </span>
      </label>
      <label class="field">
        <span>Note</span>
        <textarea id="note">${escape(label?.note || "")}</textarea>
      </label>
      ${label ? `<div class="muted" style="margin-top:6px; font-size:11px;">last: ${escape(label.status)} by ${escape(label.annotator)} @ ${escape(label.ts)}</div>` : ""}
    </div>

    <div class="block">
      <h2>Detections (${(c.detections || []).length})</h2>
      ${dets || '<div class="placeholder">None.</div>'}
    </div>
  `;

  el.querySelectorAll("#verdict-buttons button").forEach((b) => {
    b.addEventListener("click", () => submitLabel(b.dataset.status, { advance: false }));
  });

  // Web-verification "Accept" wiring: copy the canonical_value into
  // the Corrected-value input and pre-select the suggested verdict.
  el.querySelectorAll("[data-web-accept]").forEach((b) => {
    b.addEventListener("click", () => {
      const verdict = b.getAttribute("data-web-accept");
      const canon = b.getAttribute("data-canonical") || "";
      const correctedInput = $("#corrected-value");
      if (correctedInput && canon) correctedInput.value = canon;
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
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Verifying...";
  btn.classList.add("loading");
  try {
    const res = await fetch(
      `${API}/clusters/${encodeURIComponent(cid)}/web-verification`,
      { method: "POST" },
    );
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
    toast("web verification complete");
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
          Click below to run a one-shot search + LLM check.
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
  // confirmed_error -> set Corrected value + pre-select TP.
  // false_positive  -> just pre-select FP (no corrected value).
  let acceptBtn = "";
  if (!errored && verdict === "confirmed_error" && v.canonical_value) {
    acceptBtn = `<button class="web-accept" data-web-accept="tp"
        data-canonical="${escape(v.canonical_value)}"
        title="Use this canonical value as the correction and pre-select TP">
        Accept TP -> "${escape(v.canonical_value)}"
      </button>`;
  } else if (!errored && verdict === "false_positive") {
    acceptBtn = `<button class="web-accept" data-web-accept="fp"
        data-canonical=""
        title="Pre-select FP">
        Accept FP
      </button>`;
  }

  const queriesHtml = (v.queries || []).map((q) =>
    `<code>${escape(q)}</code>`
  ).join(" ");

  return `
    <div class="block web-verify">
      <h2>
        <span>Web verification</span>
        <span class="verdict-chip" style="border-color:${chipColor};color:${chipColor};">
          ${escape(errored ? "errored" : verdict)} · ${confPct}
        </span>
      </h2>
      ${errored
        ? `<div class="muted" style="font-size:12px;">${escape(v.error || "verifier failed")}</div>`
        : `
          ${v.canonical_value
            ? `<div><b>Canonical:</b> <span class="canon">${escape(v.canonical_value)}</span></div>`
            : ""}
          ${v.reasoning
            ? `<div class="reasoning">${escape(v.reasoning)}</div>`
            : ""}
          ${evidenceHtml
            ? `<ul class="evidence">${evidenceHtml}</ul>`
            : `<div class="muted" style="font-size:12px;">No surviving evidence citations.</div>`}
          <div class="muted web-meta">
            ${queriesHtml ? `queries: ${queriesHtml} · ` : ""}
            ${v.n_unique_urls || 0} unique urls / ${v.n_results || 0} results
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
  state.currentSource = source;
  state.currentTargetRow = targetRow;
  state.currentColumns = flagged;

  const start = Math.max(0, targetRow - CSV_WINDOW);
  const limit = CSV_WINDOW * 2 + 1;
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
    html += `<th class="${flag ? "flagged" : ""}">${escape(col)}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (const r of data.rows) {
    const isTarget = r.row === targetRow;
    html += `<tr class="row-idx${isTarget ? " target" : ""}" data-row="${escape(String(r.row))}"><td>${escape(String(r.row))}</td>`;
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
      const classes = ["clickable", flag ? "flagged" : "", stateCls, shadowed].filter(Boolean).join(" ");
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
      html += `<td class="${classes}" data-row="${escape(String(r.row))}" data-col="${escape(col)}" ${titleAttr}>${escape(value)}${chipsHtml}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  viewer.innerHTML = html;

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

function openCellPopover(anchorEl, source, row, col) {
  closeCellPopover();
  const map = state.cellState.get(source) || new Map();
  const ce = map.get(`${row}\x00${col}`);
  const pop = document.createElement("div");
  pop.className = "cell-popover";
  const valueText = (anchorEl.textContent || "").slice(0, 80);
  const wb = state.run?.multi_workbook
    ? (state.run.sources.find((s) => s.name === source)?.table_group || "")
    : "";
  const headLabel = wb ? `${wb} / ${source}` : source;

  let body = `<h3>Cell verdict</h3>
    <div class="meta">${escape(headLabel)} \u00B7 row ${escape(String(row))} \u00B7 ${escape(col)}</div>
    <div class="meta">value: ${escape(valueText) || "<em>(empty)</em>"}</div>`;

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
    body += `<div class="meta">current: <b>fn</b>${ce.annotator ? ` by ${escape(ce.annotator)}` : ""}</div>
      <label>Note <textarea data-field="note" rows="2">${escape(ce.note || "")}</textarea></label>
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
      await submitClusterStatusForCell(cid, status);
      closeCellPopover();
    })
  );
  pop.querySelectorAll("[data-cell-status]").forEach((b) =>
    b.addEventListener("click", async () => {
      const status = b.getAttribute("data-cell-status");
      const note = pop.querySelector("[data-field='note']")?.value || "";
      const corrected = pop.querySelector("[data-field='corrected_value']")?.value || "";
      await submitCellLabel(source, row, col, status, { note, corrected });
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

async function submitCellLabel(source, row, col, status, { note, corrected }) {
  try {
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
    toast(`saved: cell ${status}`);
    await loadCellState(source, { force: true });
    repaintCellStates(source);
    refreshStats();
    refreshSourceMeta();
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  }
}

async function submitClusterStatusForCell(clusterId, status) {
  if (!clusterId) return;
  try {
    // Re-fetch the cluster so we have its source/row/columns for the
    // payload (we may not be looking at it in the detail pane).
    const bundle = await api(`${API}/clusters/${encodeURIComponent(clusterId)}`);
    const c = bundle.cluster;
    await api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: clusterId,
        source: c.source,
        row: c.row,
        column: (c.columns || [])[0] || null,
        status,
        corrected_value: null,
        note: "",
      }),
    });
    toast(`cluster saved: ${status}`);
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
    await loadCellState(c.source, { force: true });
    repaintCellStates(c.source);
    refreshStats();
    refreshSourceMeta();
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  }
}

async function jumpToSource(name) {
  // Find the first cluster on the target source in the current queue;
  // fall back to refreshing with source filter pre-set.
  let idx = state.queue.findIndex((c) => c.source === name);
  if (idx < 0) {
    state.filters.source = name;
    document.getElementById("filter-source").value = name;
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

  const results = await Promise.allSettled(siblings.map((sib) =>
    api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: sib.cluster_id,
        source: sib.source,
        row: sib.row,
        column: (sib.columns || [])[0] || null,
        status: "skip",
        corrected_value: null,
        note: "(auto-skipped: row skipped)",
      }),
    }).then((rec) => ({ sib, rec }))
  ));

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

async function submitLabel(status, { advance = true } = {}) {
  if (!state.cluster) return;
  const c = state.cluster.cluster;
  const corrected = ($("#corrected-value")?.value ?? "").trim();
  const note = ($("#note")?.value ?? "");
  try {
    const record = await api(`${API}/labels`, {
      method: "POST",
      body: JSON.stringify({
        target: "cluster",
        target_id: c.cluster_id,
        source: c.source,
        row: c.row,
        column: (c.columns || [])[0] || null,
        status,
        corrected_value: corrected || null,
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
    // Same-value propagation: when the operator marks a single-cell
    // cluster as TP (or PARTIAL) and supplies a corrected value, walk
    // the column for other rows holding the identical erroneous value
    // and FN-label them with the same correction. Skipped silently
    // when the cluster has no usable column / row, or no correction
    // was provided (the backend requires one to know the fix).
    let propagated = 0;
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
      } catch (e) {
        // Non-fatal: the cluster verdict is already saved.
        toast(`propagation failed: ${e.message}`, 3000);
      }
    }
    const parts = [`saved: ${status}`];
    if (cascaded) parts.push(`+${cascaded} skipped on row`);
    if (propagated) parts.push(`+${propagated} fn auto-applied`);
    toast(parts.join(" · "));
    refreshStats();
    refreshSourceMeta();
    // Per-cell view depends on cluster verdicts; refresh paint without
    // a full CSV reload.
    if (c.source) {
      await loadCellState(c.source, { force: true });
      if (state.currentSource === c.source) repaintCellStates(c.source);
    }
    if (advance) selectIndex(nextUnlabeledIndex(state.index + 1));
  } catch (e) {
    toast(`error: ${e.message}`, 3000);
  }
}

// ----------------------------------------------------------- keyboard

function onKeydown(e) {
  const t = e.target;
  const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
  if (e.key === "Escape" && typing) { t.blur(); return; }
  if (typing) return;
  // Don't swallow OS shortcuts (Ctrl+C copy, Ctrl+V paste, Ctrl+F find,
  // Cmd+R reload, Alt+Tab, etc.) - only single-key bindings are ours.
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  const map = { "1": "tp", "2": "fp", "3": "partial", "4": "unsure", "5": "skip" };
  if (map[e.key]) { submitLabel(map[e.key], { advance: false }); e.preventDefault(); return; }
  if (e.key === "Enter") {
    // Enter saves with current verdict (or tp default) and advances.
    const active = document.querySelector("#verdict-buttons button.active");
    const status = active?.dataset.status;
    if (status) submitLabel(status, { advance: true });
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

bootstrap().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#f87171;">bootstrap failed: ${escape(e.stack || e.message)}</pre>`;
});
