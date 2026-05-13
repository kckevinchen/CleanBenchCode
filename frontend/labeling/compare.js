// Compare-runs panel.
// Shows a per-bucket overlap table + paginated list of cells where
// the two runs disagree at the bucket level. Click any disagreement
// to open a side-drawer with full per-side issue details + a small
// CSV slice with the cell highlighted.
//
// Backed by /api/compare, /api/compare/cells, /api/compare/cell.

import { $, $$, api, escapeHtml as escape } from "./shared/utils.js";

const state = {
  runs: [],
  loadedPickers: false,
  diff: null,                // last /api/compare result
  page: { offset: 0, limit: 50, total: 0 },
  filters: { side: "", bucket: "", source: "" },
};

// --------------------------------------------------------- run pickers

async function loadRuns() {
  if (state.loadedPickers) return;
  let data;
  try {
    data = await api("/api/runs");
  } catch (err) {
    console.error(err);
    return;
  }
  state.runs = data.runs || [];
  for (const id of ["cmp-run-a", "cmp-run-b"]) {
    const sel = $("#" + id);
    sel.innerHTML = `<option value="">(pick a run)</option>` + state.runs
      .map((r) => {
        const meta = [];
        // Dataset is the most useful disambiguator; show it first.
        if (r.dataset) {
          meta.push(r.n_sources > 1 ? `${r.dataset} (${r.n_sources} files)` : r.dataset);
        }
        if (r.model) meta.push(r.model);
        if (r.normalize_input) meta.push("normalized");
        if (r.issues != null) meta.push(`${r.issues} issues`);
        const tag = meta.length ? ` — ${meta.join(" · ")}` : "";
        return `<option value="${escape(r.run_id)}">${escape(r.run_id)}${escape(tag)}</option>`;
      }).join("");
  }
  state.loadedPickers = true;
  updateButtonState();
}

function updateButtonState() {
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  $("#cmp-load").disabled = !(a && b && a !== b);
}

function swap() {
  const a = $("#cmp-run-a");
  const b = $("#cmp-run-b");
  const tmp = a.value;
  a.value = b.value;
  b.value = tmp;
  updateButtonState();
}

// --------------------------------------------------------- summary + bucket table

async function loadCompare() {
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  if (!a || !b || a === b) return;
  $("#cmp-meta").textContent = "Loading…";
  $("#cmp-results").hidden = true;
  $("#cmp-warn").hidden = true;
  let data;
  try {
    data = await api(`/api/compare?run_a=${encodeURIComponent(a)}&run_b=${encodeURIComponent(b)}`);
  } catch (err) {
    $("#cmp-meta").textContent = "Error: " + err.message;
    return;
  }
  state.diff = data;
  state.page = { offset: 0, limit: 50, total: 0 };
  state.filters = { side: "", bucket: "", source: "" };
  renderSummary(data);
  renderBucketTable(data);
  populateBucketFilter(data);
  $("#cmp-results").hidden = false;
  await loadCells();
}

function renderSummary(d) {
  const j = d.sources.alignment_jaccard;
  const cs = d.sources.common.length;
  const tagFor = (runId, n_issues) => {
    const r = state.runs.find((x) => x.run_id === runId);
    const bits = [];
    if (r?.dataset) bits.push(r.dataset);
    if (r?.model) bits.push(r.model);
    if (r?.normalize_input) bits.push("normalized");
    bits.push(`${n_issues} issues`);
    return bits.join(" · ");
  };
  $("#cmp-meta").innerHTML =
    `Run A: <b>${escape(d.run_a)}</b> <span class="muted">(${escape(tagFor(d.run_a, d.n_a_issues))})</span><br>` +
    `Run B: <b>${escape(d.run_b)}</b> <span class="muted">(${escape(tagFor(d.run_b, d.n_b_issues))})</span><br>` +
    `<b>${d.n_diff_cells}</b> disagreeing cells across ${cs} shared source(s) ` +
    `(source overlap Jaccard ${j.toFixed(2)}).`;
  if (j < 0.5) {
    $("#cmp-warn").hidden = false;
    $("#cmp-warn").innerHTML =
      `⚠ The two runs share only ${(j * 100).toFixed(0)}% of source files. ` +
      `Cell-level diff is meaningful only on the ${cs} shared CSVs; cells from ` +
      `non-overlapping sources will appear as one-sided no matter what.`;
  } else if (cs === 0) {
    $("#cmp-warn").hidden = false;
    $("#cmp-warn").innerHTML =
      `⚠ The two runs have NO source CSVs in common. The diff is meaningless. ` +
      `Re-run both detectors against the same dataset.`;
  }
}

function renderBucketTable(d) {
  const tbody = $("#cmp-bucket-table tbody");
  tbody.innerHTML = d.by_bucket.map((row) => {
    const click = (side, bucket, n) => n > 0
      ? `<span class="clickable" data-side="${side}" data-bucket="${escape(bucket)}">${escape(String(n))}</span>`
      : `<span>${escape(String(n))}</span>`;
    return `<tr>
      <td><code>${escape(row.bucket)}</code></td>
      <td class="num">${escape(String(row.a_hits))}</td>
      <td class="num">${escape(String(row.b_hits))}</td>
      <td class="num">${escape(String(row.intersection))}</td>
      <td class="num a-only">${click("a_only", row.bucket, row.a_only)}</td>
      <td class="num b-only">${click("b_only", row.bucket, row.b_only)}</td>
      <td class="num">${escape(Number(row.jaccard).toFixed(3))}</td>
    </tr>`;
  }).join("");
  // Wire click-to-filter shortcuts.
  tbody.querySelectorAll(".clickable").forEach((el) => {
    el.addEventListener("click", () => {
      $("#cmp-filter-side").value = el.dataset.side;
      $("#cmp-filter-bucket").value = el.dataset.bucket;
      $("#cmp-filter-source").value = "";
      applyFilters();
    });
  });
}

function populateBucketFilter(d) {
  const sel = $("#cmp-filter-bucket");
  sel.innerHTML = `<option value="">(all)</option>` +
    d.by_bucket.map((r) => `<option value="${escape(r.bucket)}">${escape(r.bucket)}</option>`).join("");
}

// --------------------------------------------------------- diff-cell list

async function loadCells() {
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  if (!a || !b) return;
  const params = new URLSearchParams({
    run_a: a, run_b: b,
    offset: String(state.page.offset),
    limit: String(state.page.limit),
  });
  if (state.filters.side) params.set("side", state.filters.side);
  if (state.filters.bucket) params.set("bucket", state.filters.bucket);
  if (state.filters.source) params.set("source", state.filters.source);
  let data;
  try {
    data = await api("/api/compare/cells?" + params.toString());
  } catch (err) {
    console.error(err);
    return;
  }
  state.page.total = data.total;
  renderCells(data.cells);
  renderPager();
}

function renderCells(cells) {
  const tbody = $("#cmp-cells-table tbody");
  if (!cells.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="cmp-issue-empty">No diff cells match the current filters.</td></tr>`;
    return;
  }
  tbody.innerHTML = cells.map((c) => {
    const aSnip = c.a_issues.length ? snippet(c.a_issues[0]) : `<span class="cmp-issue-empty">—</span>`;
    const bSnip = c.b_issues.length ? snippet(c.b_issues[0]) : `<span class="cmp-issue-empty">—</span>`;
    const bucketLabel = (c.buckets && c.buckets.length) ? c.buckets.join(", ") : (c.bucket || "");
    // Whole-row issues (e.g. ``row_duplication``) come back from the
    // backend with a sentinel column ``(whole row)`` so the diff list
    // can show them. Render that as muted italic prose - it isn't a
    // real column header.
    const isWholeRow = c.column === "(whole row)";
    const colCell = isWholeRow
      ? `<span class="muted" style="font-style:italic;">whole row</span>`
      : `<code>${escape(c.column)}</code>`;
    return `<tr>
      <td><span class="cmp-side-badge cmp-side-${c.side}">${c.side === "a_only" ? "A only" : c.side === "b_only" ? "B only" : "both"}</span></td>
      <td><code>${escape(bucketLabel)}</code></td>
      <td><code>${escape(c.source)}</code></td>
      <td class="num">${escape(String(c.row))}</td>
      <td>${colCell}</td>
      <td class="cmp-issue-snippet">${aSnip}</td>
      <td class="cmp-issue-snippet">${bSnip}</td>
      <td><button type="button" class="cmp-view-btn"
        data-source="${escape(c.source)}" data-row="${escape(String(c.row))}" data-column="${escape(c.column)}">View</button></td></td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".cmp-view-btn").forEach((btn) => {
    btn.addEventListener("click", () => openDrawer(btn.dataset.source, +btn.dataset.row, btn.dataset.column));
  });
  $("#cmp-diff-count").textContent = `(${state.page.total} disagreements)`;
}

function snippet(it) {
  const obs = it.observed ? `observed: <code>${escape(String(it.observed).slice(0, 120))}</code>` : "";
  const why = it.explanation ? ` — ${escape(String(it.explanation).slice(0, 160))}` : "";
  return `<b>${escape(it.error_type || "")}</b> ${obs}${why}`;
}

function renderPager() {
  const { offset, limit, total } = state.page;
  const start = total ? offset + 1 : 0;
  const end = Math.min(offset + limit, total);
  $("#cmp-page-info").textContent = total
    ? `${start}–${end} of ${total}`
    : "0 of 0";
  $("#cmp-prev").disabled = offset <= 0;
  $("#cmp-next").disabled = end >= total;
}

function applyFilters() {
  state.filters = {
    side: $("#cmp-filter-side").value,
    bucket: $("#cmp-filter-bucket").value,
    source: $("#cmp-filter-source").value.trim(),
  };
  state.page.offset = 0;
  loadCells();
}

// --------------------------------------------------------- drawer

async function openDrawer(source, row, column) {
  const a = $("#cmp-run-a").value;
  const b = $("#cmp-run-b").value;
  $("#cmp-drawer").hidden = false;
  $("#cmp-drawer-title").textContent = `${source}  ·  row ${row}  ·  ${column}`;
  $("#cmp-drawer-body").innerHTML = `<div class="cmp-issue-empty">Loading…</div>`;
  let detail, frame;
  try {
    [detail, frame] = await Promise.all([
      api(`/api/compare/cell?run_a=${encodeURIComponent(a)}&run_b=${encodeURIComponent(b)}` +
          `&source=${encodeURIComponent(source)}&row=${row}&column=${encodeURIComponent(column)}`),
      // Pull a small slice of the CSV around the target row from run A
      // (frames are identical across runs that share the dataset).
      api(`/api/runs/${encodeURIComponent(a)}/sources/${encodeURIComponent(source)}` +
          `?start=${Math.max(0, row - 3)}&limit=7`),
    ]);
  } catch (err) {
    $("#cmp-drawer-body").innerHTML = `<div class="cmp-issue-empty">Error: ${escape(err.message)}</div>`;
    return;
  }
  $("#cmp-drawer-body").innerHTML = renderDrawer(detail, frame, row, column);
}

function renderDrawer(detail, frame, targetRow, targetCol) {
  const ctx = renderFrameContext(frame, targetRow, targetCol);
  const renderSide = (label, items) => {
    if (!items.length) {
      return `<div class="cmp-drawer-section"><h4>${label}</h4><div class="cmp-issue-empty">(no issues)</div></div>`;
    }
    const blocks = items.map((it) => `
      <div class="cmp-drawer-issue">
        <dl>
          <dt>error_type</dt><dd><code>${escape(it.error_type || "")}</code> <span class="cmp-issue-empty">(bucket: ${escape(it.bucket || "")})</span></dd>
          ${it.observed ? `<dt>observed</dt><dd><code>${escape(it.observed)}</code></dd>` : ""}
          ${it.explanation ? `<dt>explanation</dt><dd>${escape(it.explanation)}</dd>` : ""}
          ${it.suggested_fix ? `<dt>suggested_fix</dt><dd>${escape(it.suggested_fix)}</dd>` : ""}
          ${it.severity ? `<dt>severity</dt><dd>${escape(it.severity)}</dd>` : ""}
          ${it.detector ? `<dt>detector</dt><dd>${escape(it.detector)}</dd>` : ""}
        </dl>
      </div>`).join("");
    return `<div class="cmp-drawer-section"><h4>${label} (${items.length})</h4>${blocks}</div>`;
  };
  return `
    <div class="cmp-drawer-section">
      <h4>CSV context</h4>
      <div class="cmp-drawer-context">${ctx}</div>
    </div>
    ${renderSide("Run A", detail.a_issues)}
    ${renderSide("Run B", detail.b_issues)}
  `;
}

function renderFrameContext(frame, targetRow, targetCol) {
  if (!frame.rows?.length) return `<span class="cmp-issue-empty">(no rows)</span>`;
  const cols = frame.columns;
  const colWidth = 18;
  const fmt = (v) => {
    const s = String(v ?? "");
    return s.length > colWidth ? s.slice(0, colWidth - 1) + "…" : s.padEnd(colWidth, " ");
  };
  // Whole-row issues highlight the entire target row rather than a
  // single cell (there isn't a single suspect column).
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

function closeDrawer() {
  $("#cmp-drawer").hidden = true;
}

// --------------------------------------------------------- wiring

function init() {
  $("#cmp-run-a").addEventListener("change", updateButtonState);
  $("#cmp-run-b").addEventListener("change", updateButtonState);
  $("#cmp-swap").addEventListener("click", swap);
  $("#cmp-load").addEventListener("click", loadCompare);
  $("#cmp-apply-filters").addEventListener("click", applyFilters);
  $("#cmp-prev").addEventListener("click", () => {
    state.page.offset = Math.max(0, state.page.offset - state.page.limit);
    loadCells();
  });
  $("#cmp-next").addEventListener("click", () => {
    state.page.offset += state.page.limit;
    loadCells();
  });
  $("#cmp-drawer-close").addEventListener("click", closeDrawer);
  // Close drawer on Esc.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#cmp-drawer").hidden) closeDrawer();
  });
  // Lazy-load the run list when the Compare tab is first shown.
  $$(".tab").forEach((t) => {
    if (t.dataset.tab === "compare") {
      t.addEventListener("click", loadRuns);
    }
  });
  // If the page restored to the Compare tab on reload, also load.
  if (localStorage.getItem("dq-ws-tab") === "compare") loadRuns();
}

init();
