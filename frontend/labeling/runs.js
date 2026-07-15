// Runs dashboard + launcher + jobs panel.
// One SPA, three tabs. State is held here; server is the source of truth.

// Override styles.css's `html, body { overflow: hidden; height: 100vh; }`
// which is correct for the labelling view but kills scrolling here.
document.documentElement.style.height = "auto";
document.documentElement.style.overflow = "auto";

import { $, $$, api, escapeHtml as escape } from "./shared/utils.js";

function toast(msg, ms = 1800) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

// --------------------------------------------------------- tabs

function showTab(name) {
  $$(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.setAttribute("tabindex", on ? "0" : "-1");
  });
  $$(".panel").forEach((p) => {
    const on = p.dataset.panel === name;
    p.classList.toggle("hidden", !on);
    p.setAttribute("aria-hidden", on ? "false" : "true");
  });
  localStorage.setItem("dq-ws-tab", name);
  // The Step-2 discovery-run picker depends on the run registry; refresh
  // it whenever the operator opens the relations tab so a newly-finished
  // discovery run shows up without a page reload.
  if (name === "relations" && typeof populateDetectDiscoveryRuns === "function") {
    try { populateDetectDiscoveryRuns(); } catch (_) {}
  }
}
$$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// --------------------------------------------------------- state

const state = {
  workspace: null,
  launcherConfig: null,
  selectedErrorTypes: new Set(),
  activeJobId: null,
  logOffset: 0,
  jobPoller: null,
  // Runs-tab UI state. ``selectedRuns`` is the bulk-select set;
  // ``filter`` is one of "all" / "starred" / "unstarred"; ``query`` is
  // the live search box value (lower-cased). All three persist across
  // refreshes so re-clicking Refresh doesn't lose the operator's
  // narrowed view.
  selectedRuns: new Set(),
  runsFilter: localStorage.getItem("dq-ws-runs-filter") || "all",
  runsQuery: "",
  // Virtual folders (see /api/folders). ``folders`` is the server-side
  // definition list (newest first); membership is each run's
  // ``folder_id``. ``unfiledCollapsed`` is local-only UI state for the
  // synthetic "Unfiled" group (which has no server record).
  folders: [],
  unfiledCollapsed: localStorage.getItem("dq-ws-unfiled-collapsed") === "1",
};

// --------------------------------------------------------- bootstrap

async function bootstrap() {
  await refreshWorkspace();
  await loadLauncherConfig();
  await refreshJobs();

  // Restore last tab.
  const last = localStorage.getItem("dq-ws-tab") || "runs";
  showTab(last);

  $("#refresh-runs").addEventListener("click", refreshWorkspace);
  $("#refresh-jobs").addEventListener("click", refreshJobs);
  $("#show-cmd").addEventListener("click", previewCommand);
  $("#launch-form").addEventListener("submit", onSubmit);
  $("#mode").addEventListener("change", onModeChange);

  initRunsToolbar();
  initDataPrep();
  initRelationsForm();
  initRelationDetectionForm();

  // Poll jobs list every 3s (cheap - one small JSON).
  setInterval(refreshJobs, 3000);
}

// --------------------------------------------------------- runs

async function refreshWorkspace() {
  const ws = await api("/api/runs");
  state.workspace = ws;
  state.folders = ws.folders || [];
  $("#ws-meta").textContent = `${ws.project_root} · annotator ${ws.annotator} · ${ws.runs.length} runs`;
  // Drop any stale bulk-selection ids that no longer exist on disk
  // (e.g. another tab deleted them) so the toolbar count stays honest.
  const live = new Set(ws.runs.map((r) => r.run_id));
  for (const id of [...state.selectedRuns]) {
    if (!live.has(id)) state.selectedRuns.delete(id);
  }
  renderRuns(ws.runs);
}

// ---- Runs toolbar wiring ----

function initRunsToolbar() {
  const tb = $("#runs-toolbar");
  if (!tb) return;
  const search = $("#runs-search");
  search.addEventListener("input", () => {
    state.runsQuery = search.value.trim().toLowerCase();
    renderRuns(state.workspace?.runs || []);
  });
  $$("#runs-toolbar .runs-chip").forEach((c) => {
    c.addEventListener("click", () => {
      state.runsFilter = c.dataset.filter;
      localStorage.setItem("dq-ws-runs-filter", state.runsFilter);
      $$("#runs-toolbar .runs-chip").forEach((x) => {
        const on = x === c;
        x.classList.toggle("active", on);
        x.setAttribute("aria-selected", on ? "true" : "false");
      });
      renderRuns(state.workspace?.runs || []);
    });
  });
  // Restore persisted filter chip selection.
  const active = $$("#runs-toolbar .runs-chip").find((c) => c.dataset.filter === state.runsFilter);
  if (active) {
    $$("#runs-toolbar .runs-chip").forEach((x) => {
      const on = x === active;
      x.classList.toggle("active", on);
      x.setAttribute("aria-selected", on ? "true" : "false");
    });
  }
  $("#runs-bulk-clear").addEventListener("click", () => {
    state.selectedRuns.clear();
    renderRuns(state.workspace?.runs || []);
  });
  $("#runs-bulk-delete").addEventListener("click", onBulkDelete);
  $("#runs-bulk-duplicate").addEventListener("click", onBulkDuplicate);

  // "New folder" (empty) - always available from the toolbar.
  $("#runs-new-folder")?.addEventListener("click", () => createFolder(null));

  // Bulk "Move to…" picker. It's a one-shot <select>: pick a target,
  // we act, then snap it back to the placeholder so it reads as a verb,
  // not a persistent state. Options are rebuilt on every render so a
  // freshly-created folder shows up without a reload.
  const move = $("#runs-bulk-move");
  if (move) {
    move.addEventListener("change", () => {
      const v = move.value;
      move.value = "";
      if (!v) return;
      if (v === "__new__") createFolder([...state.selectedRuns]);
      else if (v === "__unfiled__") assignSelectedToFolder(null);
      else if (v.startsWith("folder:")) assignSelectedToFolder(v.slice("folder:".length));
    });
  }
}

function applyRunsFilter(runs) {
  const q = state.runsQuery;
  const f = state.runsFilter;
  return runs.filter((r) => {
    if (f === "starred" && !r.starred) return false;
    if (f === "unstarred" && r.starred) return false;
    if (!q) return true;
    const hay = (
      (r.run_id || "") + " " +
      (r.display_name || "") + " " +
      (r.model || "") + " " +
      (r.mode || "") + " " +
      (r.dataset || "")
    ).toLowerCase();
    return hay.includes(q);
  });
}

function sortRuns(runs) {
  // Starred first; within each group preserve server order (= newest
  // first by mtime, see RunRegistry.list). A stable sort is mandated
  // by the JS spec since ES2019.
  return [...runs].sort((a, b) => {
    if (a.starred && !b.starred) return -1;
    if (!a.starred && b.starred) return 1;
    return 0;
  });
}

function renderRunRow(r) {
  if (r.kind === "relations") return renderRelationsRunRow(r);
  return renderDetectionRunRow(r);
}

function renderRuns(runs) {
  $("#runs-count").textContent = `(${runs.length})`;
  const tb = $("#runs-toolbar");
  if (tb) tb.hidden = runs.length === 0;
  updateBulkBar();
  populateBulkMove();

  const box = $("#runs-list");
  if (!runs.length) {
    box.innerHTML = `
      <div class="empty-state">
        <h3>No runs yet</h3>
        <p>Launch your first detection run to get started.</p>
        <button class="btn" data-goto-launch="1">Launch new run →</button>
      </div>`;
    box.querySelector("[data-goto-launch]")?.addEventListener("click", () => showTab("launch"));
    return;
  }
  const visible = sortRuns(applyRunsFilter(runs));
  if (!visible.length) {
    box.innerHTML = `<div class="runs-empty-filtered">No runs match the current filter.</div>`;
    return;
  }
  // No folders defined → preserve the original flat list verbatim so
  // the dashboard looks identical until the operator opts into folders.
  if (!state.folders.length) {
    box.innerHTML = visible.map(renderRunRow).join("");
    wireRowActions(box);
    return;
  }
  renderGroupedRuns(box, visible);
  wireFolderActions(box);
  wireRowActions(box);
}

// Partition the (already filtered + sorted) runs into their folders plus
// the synthetic "Unfiled" bucket. A run whose ``folder_id`` doesn't match
// any live folder is treated as unfiled, so a stale pointer (e.g. after a
// folder was deleted in another tab) degrades gracefully.
function groupRunsByFolder(visible) {
  const live = new Set(state.folders.map((f) => f.id));
  const byFolder = new Map();   // folder_id -> runs[]
  const unfiled = [];
  for (const r of visible) {
    const fid = r.folder_id || "";
    if (fid && live.has(fid)) {
      if (!byFolder.has(fid)) byFolder.set(fid, []);
      byFolder.get(fid).push(r);
    } else {
      unfiled.push(r);
    }
  }
  return { byFolder, unfiled };
}

function renderGroupedRuns(box, visible) {
  const { byFolder, unfiled } = groupRunsByFolder(visible);
  // When a search query or non-"all" star filter is active, hide folders
  // that have no matching runs so the result stays focused. In the
  // default unfiltered view we always show every folder (even empty
  // ones) so they remain manageable.
  const filtering = !!state.runsQuery || state.runsFilter !== "all";
  const parts = [];
  // Run-ids hidden purely because their group is collapsed (NOT because
  // they were filtered out). We prune these from the bulk selection
  // below so "Delete/Move selected" can never act on a run the operator
  // can't currently see.
  const hiddenByCollapse = [];
  for (const f of state.folders) {
    const rows = byFolder.get(f.id) || [];
    if (filtering && rows.length === 0) continue;
    // While filtering, force-expand any folder that HAS matches so the
    // search results aren't swallowed by a persisted collapse state.
    const open = !f.collapsed || (filtering && rows.length > 0);
    parts.push(renderFolderHeader(f, rows.length, open));
    if (open) {
      parts.push(`<div class="folder-body" id="folder-body-${escape(f.id)}" role="group">`);
      parts.push(
        rows.length
          ? rows.map(renderRunRow).join("")
          : `<div class="folder-empty">Empty folder — select runs and use “Move to…” to file them here.</div>`,
      );
      parts.push(`</div>`);
    } else {
      for (const r of rows) hiddenByCollapse.push(r.run_id);
    }
  }
  // "Unfiled" group: only shown when it actually holds runs. Force-expand
  // while filtering for the same reason as folders above.
  if (unfiled.length) {
    const open = !state.unfiledCollapsed || filtering;
    parts.push(renderUnfiledHeader(unfiled.length, open));
    if (open) {
      parts.push(`<div class="folder-body" id="folder-body-unfiled" role="group">`);
      parts.push(unfiled.map(renderRunRow).join(""));
      parts.push(`</div>`);
    } else {
      for (const r of unfiled) hiddenByCollapse.push(r.run_id);
    }
  }
  box.innerHTML = parts.join("");

  // Drop now-hidden runs from the selection so the bulk bar count and
  // the destructive bulk actions stay honest about what's visible.
  let pruned = false;
  for (const id of hiddenByCollapse) {
    if (state.selectedRuns.delete(id)) pruned = true;
  }
  if (pruned) updateBulkBar();
}

function renderFolderHeader(f, count, open) {
  const id = escape(f.id);
  const name = f.name || "(unnamed folder)";
  const caret = open ? "▾" : "▸";
  const cls = f.name ? "folder-name" : "folder-name placeholder";
  return `
    <div class="folder-head ${open ? "" : "collapsed"}" data-folder="${id}">
      <button class="folder-toggle" data-action="folder-toggle" data-folder="${id}"
        aria-expanded="${open ? "true" : "false"}" aria-controls="folder-body-${id}"
        title="${open ? "Collapse folder" : "Expand folder"}">${caret}</button>
      <span class="folder-glyph" aria-hidden="true">🗀</span>
      <span class="${cls}" role="button" tabindex="0"
        data-action="folder-rename" data-folder="${id}" title="Click to rename">
        ${escape(name)}<span class="pencil" aria-hidden="true">✎</span>
      </span>
      <span class="folder-count">${escape(String(count))} run${count === 1 ? "" : "s"}</span>
      <span class="folder-head-actions">
        <button data-action="folder-rename" data-folder="${id}" title="Rename folder">Rename</button>
        <button class="danger" data-action="folder-delete" data-folder="${id}"
          title="Delete folder (its runs are kept and moved to Unfiled)">Delete</button>
      </span>
    </div>`;
}

function renderUnfiledHeader(count, open) {
  const caret = open ? "▾" : "▸";
  return `
    <div class="folder-head unfiled ${open ? "" : "collapsed"}">
      <button class="folder-toggle" data-action="unfiled-toggle"
        aria-expanded="${open ? "true" : "false"}" aria-controls="folder-body-unfiled"
        title="${open ? "Collapse" : "Expand"}">${caret}</button>
      <span class="folder-glyph" aria-hidden="true">▤</span>
      <span class="folder-name">Unfiled</span>
      <span class="folder-count">${escape(String(count))} run${count === 1 ? "" : "s"}</span>
    </div>`;
}

// Returns the leading `<button class="row-select">…</button>
// <button class="row-star">…</button> <div class="name-block">…</div>`
// HTML that's shared between detection + relations row renderers. The
// rest of each row's grid cells (counts, actions) is renderer-specific.
function renderRowLead(r, extraTags = "") {
  const id = escape(r.run_id);
  const selected = state.selectedRuns.has(r.run_id);
  const checkbox = `
    <div class="row-select">
      <input type="checkbox" data-action="row-select" data-run="${id}"
        ${selected ? "checked" : ""}
        aria-label="Select run ${id}" />
    </div>`;
  const starBtn = `
    <button class="row-star ${r.starred ? "is-starred" : ""}"
      data-action="toggle-star" data-run="${id}"
      title="${r.starred ? "Unstar (allow deletion)" : "Star to pin this run"}"
      aria-pressed="${r.starred ? "true" : "false"}">
      ${r.starred ? "★" : "☆"}
    </button>`;
  const created = r.created
    ? escape(r.created.slice(0, 19).replace("T", " ")) + " UTC"
    : "";
  const dn = r.display_name || "";
  const displayName = dn
    ? `<div class="display-name" data-action="rename" data-run="${id}" title="Click to rename">
         <span>${escape(dn)}</span><span class="pencil">✎</span>
       </div>`
    : `<div class="display-name placeholder" data-action="rename" data-run="${id}" title="Click to add a friendly name">
         <span>(unnamed)</span><span class="pencil">✎</span>
       </div>`;
  const nameBlock = `
    <div class="name-block">
      ${displayName}
      <div class="id">${id}</div>
      <div class="sub">${extraTags}${created ? `<span>${created}</span>` : ""}</div>
    </div>`;
  return { checkbox, starBtn, nameBlock, selected };
}

function renderDetectionRunRow(r) {
  // Primary "Label" CTA: the single entry point into labelling.
  // ``label.html`` is the unified surface that hosts every error
  // type a run can produce - per-cell typo / row-duplication /
  // semantic-incompatible clusters, multi-table findings, AND
  // inconsistency clusters merged in from ``/column-clusters``
  // (with LLM drafts surfaced as queue items). The legacy column
  // board (``column_board.html``) is still reachable via the deep
  // link in the right-pane detail when an operator needs bulk
  // editing, but it is no longer the run-list CTA.
  const hasColumnClusters =
    r.has_column_cluster_drafts || r.has_column_cluster_labels;
  const hasCellLevel = r.has_clusters || r.has_multi_table;
  const hasAnyFindings = hasColumnClusters || hasCellLevel;
  // Fold every error type's finding count into the single "clusters"
  // numeric column instead of tacking "+N MT / +N inconsistency"
  // suffixes onto the Label button. ``r.clusters`` is the cell-level
  // total (typo / semantic / row-dup …); inconsistency column-clusters
  // and multi-table findings are counted separately by the registry.
  // The column shows the grand total with a hover breakdown so the
  // per-type detail isn't lost.
  const cellClusters = Number(r.clusters ?? 0);
  const mtFindings = Number(r.n_multi_table_findings ?? 0);
  const ccTotal =
    Number(r.n_column_cluster_drafts ?? 0) + Number(r.n_column_cluster_labels ?? 0);
  const clustersTotal = cellClusters + ccTotal + mtFindings;
  const breakdown = [
    cellClusters ? `${cellClusters} cell` : "",
    ccTotal ? `${ccTotal} inconsistency` : "",
    mtFindings ? `${mtFindings} MT` : "",
  ].filter(Boolean).join(" + ");
  // Only attach the tooltip when the total is actually a mix; a lone
  // "812 cell" hover over "812" is just noise. ``breakdown`` is always
  // non-empty when either of these is positive, so the flag is enough.
  const showBreakdown = ccTotal > 0 || mtFindings > 0;
  const clustersCell =
    `<div class="num"${showBreakdown ? ` title="${escape(breakdown)}"` : ""}>` +
    `<span class="label">clusters</span>${escape(String(clustersTotal))}</div>`;
  const labelLink = hasAnyFindings
    ? `<a class="primary" href="./label.html?run=${encodeURIComponent(r.run_id)}" title="Label every finding in this run (cells + multi-table + inconsistency clusters)">Label</a>`
    : `<span class="muted">(no findings)</span>`;
  const reportLink = r.has_report
    ? `<a href="/api/runs/${encodeURIComponent(r.run_id)}/report" target="_blank" rel="noopener">Report</a>`
    : `<button data-action="build-report" data-run="${escape(r.run_id)}">Build report</button>`;
  const verifyBtn = r.has_clusters
    ? (r.has_web_verifications
        ? `<button data-action="verify-web" data-run="${escape(r.run_id)}" title="Verify all labelled cells on the web (re-run / extend coverage)" class="verify-web-btn"><span class="verify-web-icon">⌕</span> Verified <span class="verify-web-count">${escape(String(r.n_web_verifications ?? 0))}</span></button>`
        : `<button data-action="verify-web" data-run="${escape(r.run_id)}" title="Run experiments.run_web_verify on every gate-eligible cluster" class="verify-web-btn"><span class="verify-web-icon">⌕</span> Verify on web</button>`)
    : "";
  const tagSpan = (cls, value, title) => value
    ? `<span class="tag ${cls}"${title ? ` title="${escape(title)}"` : ""}>${escape(value)}</span>`
    : "";
  const tags = [
    tagSpan("dataset", r.dataset, `dataset: ${r.dataset || ""}`),
    tagSpan("model", r.model),
  ].join("");
  const { checkbox, starBtn, nameBlock, selected } = renderRowLead(r, tags);
  const rowClass = [
    "row",
    r.starred ? "starred" : "",
    selected ? "row-selected" : "",
  ].filter(Boolean).join(" ");
  return `
    <div class="${rowClass}">
      ${checkbox}
      ${starBtn}
      ${nameBlock}
      ${clustersCell}
      <div class="num"><span class="label">issues</span>${escape(String(r.issues ?? 0))}</div>
      <div class="num"><span class="label">labelled</span>${r.has_labels ? "yes" : "–"}</div>
      <div class="actions">
        ${labelLink}
        ${reportLink}
        ${verifyBtn}
        <button class="danger" data-action="delete-run" data-run="${escape(r.run_id)}" title="Delete this run">Delete</button>
      </div>
    </div>
  `;
}

// --------------------------------------------------------- runs: row actions

function findRun(runId) {
  return (state.workspace?.runs || []).find((r) => r.run_id === runId);
}

function replaceRunInState(updated) {
  const list = state.workspace?.runs || [];
  const i = list.findIndex((r) => r.run_id === updated.run_id);
  if (i >= 0) list[i] = { ...list[i], ...updated };
}

function wireRowActions(box) {
  box.querySelectorAll('[data-action="build-report"]').forEach((b) => {
    b.addEventListener("click", async () => {
      try {
        const job = await api("/api/launcher/jobs", {
          method: "POST",
          body: JSON.stringify({ kind: "report", run_id: b.dataset.run }),
        });
        toast(`build_report started: ${job.job_id}`);
        showTab("jobs");
        openJob(job.job_id);
        refreshJobs();
      } catch (e) {
        toast(`error: ${e.message}`, 3500);
      }
    });
  });

  box.querySelectorAll('[data-action="build-relation-report"]').forEach((b) => {
    b.addEventListener("click", async () => {
      try {
        const job = await api("/api/launcher/jobs", {
          method: "POST",
          body: JSON.stringify({ kind: "relation_report", run_id: b.dataset.run }),
        });
        toast(`relation report build started: ${job.job_id}`);
        showTab("jobs");
        openJob(job.job_id);
        refreshJobs();
      } catch (e) {
        toast(`error: ${e.message}`, 3500);
      }
    });
  });

  box.querySelectorAll('[data-action="validate-relations"]').forEach((b) => {
    b.addEventListener("click", async () => {
      try {
        const job = await api("/api/launcher/jobs", {
          method: "POST",
          body: JSON.stringify({
            kind: "relation_validation",
            run_id: b.dataset.run,
            params: {},
          }),
        });
        toast(`relation validation started: ${job.job_id}`);
        showTab("jobs");
        openJob(job.job_id);
        refreshJobs();
      } catch (e) {
        toast(`error: ${e.message}`, 3500);
      }
    });
  });

  box.querySelectorAll('[data-action="delete-run"]').forEach((b) => {
    b.addEventListener("click", () => deleteSingleRun(b));
  });

  box.querySelectorAll('[data-action="verify-web"]').forEach((b) => {
    b.addEventListener("click", () => {
      const runId = b.dataset.run;
      const run = findRun(runId);
      openWebVerifyModal(run || { run_id: runId });
    });
  });

  // Star toggle (★/☆) - optimistic UI: flip the icon immediately and
  // roll back if the PATCH fails. The starred state pins the run to
  // the top of the list AND guards it against accidental deletion.
  box.querySelectorAll('[data-action="toggle-star"]').forEach((b) => {
    b.addEventListener("click", async () => {
      const runId = b.dataset.run;
      const r = findRun(runId);
      if (!r) return;
      const next = !r.starred;
      r.starred = next;             // optimistic
      renderRuns(state.workspace.runs);
      try {
        const updated = await api(`/api/runs/${encodeURIComponent(runId)}`, {
          method: "PATCH",
          body: JSON.stringify({ starred: next }),
        });
        replaceRunInState(updated);
        renderRuns(state.workspace.runs);
      } catch (e) {
        r.starred = !next;          // rollback
        renderRuns(state.workspace.runs);
        toast(`star failed: ${e.message}`, 3500);
      }
    });
  });

  // Inline rename. Click the display-name area to swap it for an
  // <input>; Enter commits, Esc cancels, blur commits. We rebuild
  // just the input so we don't re-render the whole row mid-edit.
  box.querySelectorAll('[data-action="rename"]').forEach((el) => {
    el.addEventListener("click", (ev) => {
      // Ignore clicks on a pre-existing input (re-entrancy guard).
      if (ev.target.tagName === "INPUT") return;
      startInlineRename(el);
    });
  });

  // Bulk-select checkboxes.
  box.querySelectorAll('[data-action="row-select"]').forEach((cb) => {
    cb.addEventListener("change", () => {
      const runId = cb.dataset.run;
      if (cb.checked) state.selectedRuns.add(runId);
      else state.selectedRuns.delete(runId);
      cb.closest(".row")?.classList.toggle("row-selected", cb.checked);
      updateBulkBar();
    });
  });
}

function startInlineRename(el) {
  const runId = el.dataset.run;
  const r = findRun(runId);
  if (!r) return;
  const current = r.display_name || "";
  el.innerHTML = `<input type="text" maxlength="120" value="${escape(current)}" placeholder="Friendly name (optional)" />`;
  const input = el.querySelector("input");
  input.focus();
  input.select();
  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    const next = input.value.trim();
    if (next === current) {
      renderRuns(state.workspace.runs);
      return;
    }
    try {
      const updated = await api(`/api/runs/${encodeURIComponent(runId)}`, {
        method: "PATCH",
        body: JSON.stringify({ display_name: next }),
      });
      replaceRunInState(updated);
      toast(next ? `renamed: ${next}` : "name cleared");
    } catch (e) {
      toast(`rename failed: ${e.message}`, 3500);
    }
    renderRuns(state.workspace.runs);
  };
  const cancel = () => {
    if (done) return;
    done = true;
    renderRuns(state.workspace.runs);
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); commit(); }
    else if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
  });
  input.addEventListener("blur", commit);
}

async function deleteSingleRun(btn) {
  const runId = btn.dataset.run;
  const r = findRun(runId);
  const isStarred = !!r?.starred;
  const niceName = r?.display_name ? `"${r.display_name}" (${runId})` : `"${runId}"`;
  let msg = `Delete run ${niceName}?\n\nThis will permanently remove its outputs/${runId}/ directory (clusters, summary, report, labels). This cannot be undone.`;
  if (isStarred) {
    msg = `⚠ This run is STARRED.\n\n${msg}\n\nDelete the starred run anyway?`;
  }
  if (!confirm(msg)) return;
  btn.disabled = true;
  btn.textContent = "Deleting…";
  try {
    const url = `/api/runs/${encodeURIComponent(runId)}${isStarred ? "?force=true" : ""}`;
    await api(url, { method: "DELETE" });
    toast(`deleted run ${runId}`);
    state.selectedRuns.delete(runId);
    refreshWorkspace();
  } catch (e) {
    toast(`delete failed: ${e.message}`, 4000);
    btn.disabled = false;
    btn.textContent = "Delete";
  }
}

// Bulk "Duplicate" (top-bar action). Acts on the current bulk
// selection; for a single selected run it reads like the old per-row
// Duplicate. The clean-slate vs. keep-labels choice is asked once and
// applied to every selected run. The backend ``/duplicate`` endpoint is
// single-run, so we fan out sequentially (the N here is a small UI
// selection) and report a clean "M duplicated" summary.
async function onBulkDuplicate() {
  const ids = [...state.selectedRuns];
  if (!ids.length) return;
  const runs = ids.map(findRun).filter(Boolean);
  if (!runs.length) return;
  const label = runs.length === 1
    ? (runs[0].display_name ? `"${runs[0].display_name}" (${runs[0].run_id})` : `"${runs[0].run_id}"`)
    : `${runs.length} runs`;
  // OK = keep existing labels; Cancel leads to a second confirm so we
  // never silently discard annotation work.
  const resetLabels = !confirm(
    `Duplicate ${label}?\n\n` +
    "Click OK to copy WITH existing labels, or Cancel to choose a clean-slate copy (no labels carried over)."
  );
  if (resetLabels && !confirm(
    `Create clean-slate ${runs.length === 1 ? "copy" : "copies"} of ${label} (no labels carried over)?`
  )) {
    return;
  }
  const btn = $("#runs-bulk-duplicate");
  btn.disabled = true;
  btn.textContent = "Duplicating…";
  let okN = 0, failN = 0, lastId = null;
  for (const r of runs) {
    try {
      const res = await api(`/api/runs/${encodeURIComponent(r.run_id)}/duplicate`, {
        method: "POST",
        body: JSON.stringify({ reset_labels: resetLabels }),
      });
      okN += 1;
      lastId = res.run_id;
    } catch (e) {
      failN += 1;
    }
  }
  btn.disabled = false;
  btn.textContent = "Duplicate";
  toast(
    okN === 1 && !failN ? `duplicated → ${lastId}` : `duplicated ${okN}${failN ? ` · ${failN} failed` : ""}`,
    failN ? 4000 : 1800,
  );
  state.selectedRuns.clear();
  refreshWorkspace();
}

function updateBulkBar() {
  const bar = $("#runs-bulk");
  if (!bar) return;
  const n = state.selectedRuns.size;
  if (n === 0) { bar.hidden = true; return; }
  bar.hidden = false;
  $("#runs-bulk-count").textContent = `${n} selected`;
}

async function onBulkDelete() {
  const ids = [...state.selectedRuns];
  if (!ids.length) return;
  const runs = ids.map(findRun).filter(Boolean);
  const starred = runs.filter((r) => r.starred);
  const deletable = runs.filter((r) => !r.starred);
  if (!deletable.length) {
    toast(`All ${ids.length} selected runs are starred - unstar first`, 3500);
    return;
  }
  let msg = `Delete ${deletable.length} run${deletable.length === 1 ? "" : "s"}?\n\n`
    + deletable.slice(0, 8).map((r) => "  • " + (r.display_name ? `${r.display_name} (${r.run_id})` : r.run_id)).join("\n")
    + (deletable.length > 8 ? `\n  …and ${deletable.length - 8} more` : "");
  if (starred.length) {
    msg += `\n\n${starred.length} starred run${starred.length === 1 ? " was" : "s were"} skipped.`;
  }
  msg += "\n\nThis cannot be undone.";
  if (!confirm(msg)) return;
  const btn = $("#runs-bulk-delete");
  btn.disabled = true;
  btn.textContent = "Deleting…";
  let okN = 0, failN = 0;
  // Sequential deletes - keeps server load low and lets us show a
  // clean "N/M deleted" toast at the end. The N here is small (UI
  // bulk-selection), so latency is fine.
  for (const r of deletable) {
    try {
      await api(`/api/runs/${encodeURIComponent(r.run_id)}`, { method: "DELETE" });
      okN += 1;
      state.selectedRuns.delete(r.run_id);
    } catch (e) {
      failN += 1;
    }
  }
  btn.disabled = false;
  btn.textContent = "Delete selected";
  toast(`deleted ${okN}${failN ? ` · ${failN} failed` : ""}`);
  refreshWorkspace();
}

// --------------------------------------------------------- runs: folders
//
// Folders are a virtual grouping layer: creating one and filing runs
// into it never moves a run directory on disk (run-ids stay valid for
// labels / reports). The server holds folder definitions in
// outputs/folders.json; each run carries a ``folder_id`` pointer.

// Rebuild the bulk "Move to…" picker from the current folder list. Runs
// as a one-shot verb control: choosing a target acts immediately and the
// caller snaps the <select> back to its placeholder.
function populateBulkMove() {
  const sel = $("#runs-bulk-move");
  if (!sel) return;
  const opts = [`<option value="" selected>Move to…</option>`];
  for (const f of state.folders) {
    opts.push(`<option value="folder:${escape(f.id)}">${escape(f.name || "(unnamed folder)")}</option>`);
  }
  opts.push(`<option value="__unfiled__">— Remove from folder —</option>`);
  opts.push(`<option value="__new__">＋ New folder from selection…</option>`);
  sel.innerHTML = opts.join("");
}

// Create a folder. ``runIds`` may be an array of ids to file into the new
// folder immediately (bulk-bar path) or ``null`` for an empty folder
// (toolbar path).
async function createFolder(runIds) {
  const name = (prompt("New folder name:", "") || "").trim();
  if (name === "") return;   // empty / cancelled -> no-op
  try {
    const body = { name };
    if (Array.isArray(runIds) && runIds.length) body.run_ids = runIds;
    const res = await api("/api/folders", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const moved = res.moved || 0;
    toast(moved ? `folder “${name}” created · ${moved} moved` : `folder “${name}” created`);
    if (moved) state.selectedRuns.clear();
    await refreshWorkspace();
  } catch (e) {
    toast(`create folder failed: ${e.message}`, 3500);
  }
}

// Move the bulk-selected runs into ``folderId`` (null/"" -> Unfiled).
async function assignSelectedToFolder(folderId) {
  const ids = [...state.selectedRuns];
  if (!ids.length) return;
  try {
    const res = await api("/api/folders/assign", {
      method: "POST",
      body: JSON.stringify({ run_ids: ids, folder_id: folderId || null }),
    });
    toast(folderId ? `moved ${res.moved} to folder` : `removed ${res.moved} from folder`);
    state.selectedRuns.clear();
    await refreshWorkspace();
  } catch (e) {
    toast(`move failed: ${e.message}`, 3500);
  }
}

function findFolder(folderId) {
  return state.folders.find((f) => f.id === folderId);
}

function wireFolderActions(box) {
  // Collapse / expand a folder. Optimistic: flip locally + re-render,
  // then persist the collapse flag server-side so it survives reload.
  box.querySelectorAll('[data-action="folder-toggle"]').forEach((b) => {
    b.addEventListener("click", async () => {
      const f = findFolder(b.dataset.folder);
      if (!f) return;
      const next = !f.collapsed;
      f.collapsed = next;             // optimistic
      renderRuns(state.workspace?.runs || []);
      try {
        await api(`/api/folders/${encodeURIComponent(f.id)}`, {
          method: "PATCH",
          body: JSON.stringify({ collapsed: next }),
        });
      } catch (e) {
        f.collapsed = !next;          // rollback
        renderRuns(state.workspace?.runs || []);
        toast(`folder update failed: ${e.message}`, 3000);
      }
    });
  });

  // The synthetic "Unfiled" group has no server record - its collapse
  // state is local-only (persisted to localStorage).
  box.querySelectorAll('[data-action="unfiled-toggle"]').forEach((b) => {
    b.addEventListener("click", () => {
      state.unfiledCollapsed = !state.unfiledCollapsed;
      localStorage.setItem("dq-ws-unfiled-collapsed", state.unfiledCollapsed ? "1" : "0");
      renderRuns(state.workspace?.runs || []);
    });
  });

  box.querySelectorAll('[data-action="folder-rename"]').forEach((el) => {
    el.addEventListener("click", () => renameFolder(el.dataset.folder));
  });
  // The folder-name span is a role="button" (not a real <button>), so it
  // needs explicit Enter/Space handling to be keyboard-operable. The real
  // Rename <button> already activates on Enter/Space natively.
  box.querySelectorAll('.folder-name[data-action="folder-rename"]').forEach((el) => {
    el.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        renameFolder(el.dataset.folder);
      }
    });
  });

  box.querySelectorAll('[data-action="folder-delete"]').forEach((b) => {
    b.addEventListener("click", () => deleteFolder(b.dataset.folder));
  });
}

async function renameFolder(folderId) {
  const f = findFolder(folderId);
  if (!f) return;
  const name = prompt("Rename folder:", f.name || "");
  if (name === null) return;                 // cancelled
  const next = name.trim();
  if (next === (f.name || "")) return;       // unchanged
  try {
    await api(`/api/folders/${encodeURIComponent(folderId)}`, {
      method: "PATCH",
      body: JSON.stringify({ name: next }),
    });
    toast(`renamed folder`);
    await refreshWorkspace();
  } catch (e) {
    toast(`rename failed: ${e.message}`, 3500);
  }
}

async function deleteFolder(folderId) {
  const f = findFolder(folderId);
  if (!f) return;
  const label = f.name ? `“${f.name}”` : "this folder";
  if (!confirm(
    `Delete folder ${label}?\n\nThe runs inside it are NOT deleted — they move back to “Unfiled”.`
  )) return;
  try {
    const res = await api(`/api/folders/${encodeURIComponent(folderId)}`, {
      method: "DELETE",
    });
    toast(`folder deleted${res.unfiled ? ` · ${res.unfiled} unfiled` : ""}`);
    await refreshWorkspace();
  } catch (e) {
    toast(`delete folder failed: ${e.message}`, 3500);
  }
}

// --------------------------------------------------------- web-verify modal
//
// Lets the operator launch ``experiments.run_web_verify`` over every
// gate-eligible cluster in a finished detection run. Modal lives in
// ``index.html`` (#wv-modal); we just wire up open/close + submit.

const WV_DEFAULT_TYPES = [
  "typo",
  "inconsistency",
  "semantic_incompatible",
  "row_duplication",
  "fd_violation",
  "inter_table_inconsistency",
];

const WV_NON_VERIFIABLE = new Set([
  "row_calculation_inconsistency",
  "string_calculation_inconsistency",
  "numeric_outlier",
]);

const wvState = {
  runId: null,
  selectedTypes: new Set(),
  // Cache the last fetched verifications per run so re-opening the
  // modal feels instant.
  statsCache: new Map(),
};

function openWebVerifyModal(run) {
  const modal = $("#wv-modal");
  if (!modal) return;
  wvState.runId = run.run_id;
  wvState.selectedTypes = new Set();

  // Headline: which run + clusters total.
  $("#wv-modal-run").textContent = run.run_id;
  $("#wv-stat-clusters").textContent =
    run.clusters != null ? String(run.clusters) : "–";
  $("#wv-stat-verified").textContent = "…";
  $("#wv-stat-confirmed").textContent = "…";
  $("#wv-stat-fp").textContent = "…";

  renderWvErrorTypes();

  // Reset form state on every open so a previous run's settings
  // don't bleed in.
  $("#wv-concurrency").value = 4;
  $("#wv-max-clusters").value = "";
  $("#wv-reasoning-effort").value = "";
  $("#wv-model").value = "";
  $("#wv-force").checked = false;
  $("#wv-submit").disabled = false;
  $("#wv-submit").textContent = "Launch verification";

  modal.hidden = false;
  // Tiny defer so transitions can fire if we add any later.
  requestAnimationFrame(() => modal.classList.add("open"));

  // Pull existing verification stats in the background so the
  // headline numbers fill in. Doesn't block the modal.
  void loadWvStats(run.run_id);
}

function closeWebVerifyModal() {
  const modal = $("#wv-modal");
  if (!modal) return;
  modal.classList.remove("open");
  modal.hidden = true;
}

function renderWvErrorTypes() {
  // Use the launcher config's error_types when available so chips
  // stay in sync with the cleaning launcher; fall back to the
  // hard-coded web-verifiable defaults if config hasn't loaded yet.
  const box = $("#wv-error-types");
  if (!box) return;
  box.innerHTML = "";
  const cfg = state.launcherConfig || {};
  const all = (cfg.error_types || []).map((et) => ({
    key: et.key,
    name: et.name || et.key,
  }));
  // Hide non-web-verifiable types entirely so the operator can't
  // pick a useless filter.
  const items = (all.length ? all : WV_DEFAULT_TYPES.map((k) => ({ key: k, name: k })))
    .filter((et) => !WV_NON_VERIFIABLE.has(et.key));

  for (const et of items) {
    const label = document.createElement("label");
    label.className = "chip wv-chip";
    label.title = et.key;
    label.innerHTML = `<input type="checkbox" value="${escape(et.key)}"><span>${escape(et.name)}</span>`;
    const input = label.querySelector("input");
    input.addEventListener("change", () => {
      label.classList.toggle("active", input.checked);
      if (input.checked) wvState.selectedTypes.add(et.key);
      else wvState.selectedTypes.delete(et.key);
    });
    box.appendChild(label);
  }
}

async function loadWvStats(runId) {
  // Cache hit: instant repaint.
  const cached = wvState.statsCache.get(runId);
  if (cached) {
    paintWvStats(cached);
  }
  try {
    const data = await api(
      `/api/runs/${encodeURIComponent(runId)}/web-verifications`,
    );
    wvState.statsCache.set(runId, data);
    // Only paint if the modal is still showing this run.
    if (wvState.runId === runId) paintWvStats(data);
  } catch (e) {
    // Silent: 404 just means "no verifications yet".
    if (wvState.runId === runId && !cached) {
      paintWvStats({ counts: {}, items: [] });
    }
  }
}

function paintWvStats(data) {
  const items = data.items || [];
  const counts = data.counts || {};
  $("#wv-stat-verified").textContent = String(items.length);
  $("#wv-stat-confirmed").textContent = String(counts.confirmed_error || 0);
  $("#wv-stat-fp").textContent = String(counts.false_positive || 0);
}

async function submitWvForm(ev) {
  ev.preventDefault();
  if (!wvState.runId) return;
  const params = {};
  const concurrency = parseInt($("#wv-concurrency").value, 10);
  if (Number.isFinite(concurrency) && concurrency > 0) {
    params.concurrency = concurrency;
  }
  const maxClusters = parseInt($("#wv-max-clusters").value, 10);
  if (Number.isFinite(maxClusters) && maxClusters > 0) {
    params.max_clusters = maxClusters;
  }
  const re = $("#wv-reasoning-effort").value.trim();
  if (re) params.reasoning_effort = re;
  const model = $("#wv-model").value.trim();
  if (model) params.model = model;
  if (wvState.selectedTypes.size > 0) {
    params.error_types = Array.from(wvState.selectedTypes).join(",");
  }
  if ($("#wv-force").checked) params.force = true;

  const submit = $("#wv-submit");
  submit.disabled = true;
  submit.textContent = "Launching…";
  try {
    const job = await api("/api/launcher/jobs", {
      method: "POST",
      body: JSON.stringify({
        kind: "web_verify",
        run_id: wvState.runId,
        params,
      }),
    });
    toast(`web verification started: ${job.job_id}`);
    closeWebVerifyModal();
    showTab("jobs");
    if (typeof openJob === "function") openJob(job.job_id);
    refreshJobs();
  } catch (e) {
    toast(`error: ${e.message}`, 3500);
    submit.disabled = false;
    submit.textContent = "Launch verification";
  }
}

// Wire up modal close + submit listeners. ``runs.js`` loads as
// ``type="module"`` which is implicitly deferred, so the DOM is ready
// by the time we get here -- no DOMContentLoaded gymnastics needed.
function initWebVerifyModal() {
  const modal = document.getElementById("wv-modal");
  if (!modal) return;
  modal.querySelectorAll("[data-wv-close]").forEach((el) => {
    el.addEventListener("click", closeWebVerifyModal);
  });
  const form = document.getElementById("wv-modal-form");
  if (form) form.addEventListener("submit", submitWvForm);
  // Esc key closes the modal when it's open.
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !modal.hidden) closeWebVerifyModal();
  });
}
initWebVerifyModal();

// --------------------------------------------------------- launcher config

async function loadLauncherConfig() {
  const cfg = await api("/api/launcher/config");
  state.launcherConfig = cfg;

  // Dataset dropdown: dir entries + file entries.
  const sel = $("#csv-dir");
  sel.innerHTML = `<option value="">(use config default)</option>`;
  for (const ds of cfg.datasets || []) {
    const o = document.createElement("option");
    o.value = ds.path;
    o.dataset.type = ds.type;
    o.textContent = ds.type === "dir"
      ? `${ds.name}/  (${ds.csv_count} CSVs)`
      : `${ds.name}  (1 file)`;
    sel.appendChild(o);
  }

  // Error-type chips, grouped by primary category (the part of the
  // taxonomy category before the first "/"). E.g. categories
  // "Single-Column / Typo" and "Single-Column / Inconsistency" both
  // collapse into the "Single-Column" group. Scope is inferred from
  // these selections server-side, so we don't expose a separate scope UI.
  const etBox = $("#error-types");
  etBox.innerHTML = "";
  const groups = new Map();   // primary category -> [et, ...]
  for (const et of cfg.error_types || []) {
    const raw = (et.category || "Other").trim();
    const cat = raw.split("/")[0].trim() || "Other";
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat).push(et);
  }
  for (const [cat, items] of groups) {
    const group = document.createElement("div");
    group.className = "et-group";
    const head = document.createElement("div");
    head.className = "et-group-head";
    head.innerHTML = `
      <span class="et-group-title">${escape(cat)}</span>
      <span class="et-group-count" data-count>0 / ${items.length}</span>
      <button type="button" class="et-group-all" data-act="toggle">Select all</button>
    `;
    const body = document.createElement("div");
    body.className = "et-group-body";
    const chips = [];
    for (const et of items) {
      const chip = makeChip(et.key, state.selectedErrorTypes, et.name || et.key);
      chip.title = et.key + (et.name && et.name !== et.key ? ` \u00b7 ${et.name}` : "");
      body.appendChild(chip);
      chips.push(chip);
    }
    const countEl = head.querySelector("[data-count]");
    const toggleBtn = head.querySelector('[data-act="toggle"]');
    const refreshGroup = () => {
      const checked = chips.filter((c) => c.querySelector("input").checked).length;
      countEl.textContent = `${checked} / ${items.length}`;
      countEl.classList.toggle("has-selected", checked > 0);
      toggleBtn.textContent = checked === items.length ? "Clear all" : "Select all";
    };
    toggleBtn.addEventListener("click", () => {
      const allOn = chips.every((c) => c.querySelector("input").checked);
      chips.forEach((c) => {
        const input = c.querySelector("input");
        const wantOn = !allOn;
        if (input.checked !== wantOn) {
          input.checked = wantOn;
          input.dispatchEvent(new Event("change"));
        }
      });
      refreshGroup();
    });
    chips.forEach((c) => {
      c.querySelector("input").addEventListener("change", refreshGroup);
    });
    refreshGroup();
    group.appendChild(head);
    group.appendChild(body);
    etBox.appendChild(group);
  }

  // Pre-fill every form field from config/experiment.yaml so the user
  // sees the actual default that will be used (instead of a vague
  // "(default)" label). Placeholders keep the input visually empty but
  // show the value - submitting with an empty field omits the flag and
  // the CLI / YAML defaults kick in on the server side. Selects,
  // checkboxes and chip sets get explicitly pre-selected.
  applyConfigDefaults(cfg.defaults || {});

  onModeChange();
}

/** Display-helper: "value" -> string suitable for an input placeholder. */
function fmtDefault(v) {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.join(",");
  return String(v);
}

/** Mark the <option> whose value === val as "(default)" and select it. */
function markDefaultOption(selectId, val) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  const needle = val === null || val === undefined ? "" : String(val);
  for (const opt of sel.options) {
    // Strip any previous " (default)" suffix so reloads are idempotent.
    opt.textContent = opt.textContent.replace(/\s+\(default\)\s*$/, "");
    if (opt.value === needle && needle !== "") {
      opt.textContent = `${opt.textContent} (default)`;
      sel.value = opt.value;
    }
  }
}

function applyConfigDefaults(defaults) {
  const data = defaults.data || {};
  const llm = defaults.llm || {};
  const prompts = defaults.prompts || {};
  const exp = defaults.experiment || {};
  const gt = defaults.good_turing || {};

  // ----- Data -----
  setPlaceholder("csv-glob", data.csv_glob);
  setPlaceholder("chunk-size", data.chunk_size);
  setPlaceholder("max-chunks", data.max_chunks == null ? "all" : data.max_chunks);
  setPlaceholder("columns", Array.isArray(data.columns) ? data.columns.join(",") : (data.columns || "all"));
  markDefaultOption("column-mode", data.column_mode);
  // If the configured csv_dir (or csv_path) matches a dataset row,
  // preselect it.
  const dsSel = document.getElementById("csv-dir");
  if (dsSel) {
    const target = data.csv_dir || data.csv_path;
    if (target) {
      for (const opt of dsSel.options) {
        // Strip previous suffix for idempotency.
        opt.textContent = opt.textContent.replace(/\s+\(default\)\s*$/, "");
      }
      for (const opt of dsSel.options) {
        if (opt.value && (opt.value === target || opt.value.endsWith(`/${target}`))) {
          opt.textContent = `${opt.textContent} (default)`;
          dsSel.value = opt.value;
          break;
        }
      }
    }
  }

  // ----- LLM -----
  setPlaceholder("model", llm.model);
  setPlaceholder("concurrency", llm.concurrency == null ? "auto" : llm.concurrency);
  markDefaultOption("reasoning-effort", llm.reasoning_effort);
  markDefaultOption("synthesis-effort", llm.synthesis_effort);

  // ----- Experiment + GT -----
  markDefaultOption("mode", exp.mode);
  markDefaultOption("gt-mode", gt.mode);
  setPlaceholder("gt-num-runs", gt.num_runs);
  setPlaceholder("gt-min-runs", gt.min_runs);
  setPlaceholder("gt-target-missing-mass", gt.target_missing_mass);
  markDefaultOption("gt-stop-scope", gt.stop_scope);
  setPlaceholder("gt-base-seed", gt.base_seed);
  setChecked("gt-shuffle-rows", gt.shuffle_rows === true);
  setChecked("gt-shuffle-columns", gt.shuffle_columns === true);
  setChecked("gt-prompt-ensemble", gt.prompt_ensemble === true);
  setChecked("gt-skip-identity-run", gt.skip_identity_run === true);
  markDefaultOption("gt-variant-effort", gt.variant_generation_effort);

  // Per-arm defaults (good_turing.per_arm.defaults). Fall back to
  // ArmConfig.default() literals when the YAML block is absent so the
  // user always sees what the run will actually use. Note: epsilon /
  // k_min / k_max are NOT shown here -- they're driven by the
  // top-level target_missing_mass / min_runs / num_runs fields.
  const perArm = (gt.per_arm && gt.per_arm.defaults) || {};
  if (perArm.require_two_consecutive != null) {
    markDefaultOption(
      "gt-per-arm-two-consecutive",
      perArm.require_two_consecutive ? "true" : "false",
    );
  } else {
    markDefaultOption("gt-per-arm-two-consecutive", "true");
  }
  // stop_when: defaults to "gt" via ArmConfig.default() when absent.
  markDefaultOption("gt-per-arm-stop-when", perArm.stop_when || "gt");
  // epsilon_chao1: defaults to 0.05 via ArmConfig.default() when absent.
  setPlaceholder(
    "gt-per-arm-epsilon-chao1",
    perArm.epsilon_chao1 != null ? perArm.epsilon_chao1 : 0.05,
  );

  // ----- Chip pre-selection (error_types) -----
  preselectChips("error-types", prompts.error_types, state.selectedErrorTypes);
  refreshRowDupBlock();
  refreshMultiTableBlock();
}

function setChecked(id, on) {
  const el = document.getElementById(id);
  if (!el) return;
  el.checked = !!on;
}

function preselectChips(containerId, values, set) {
  if (!values || values.length === 0) return;
  const box = document.getElementById(containerId);
  if (!box) return;
  const wanted = new Set(values);
  box.querySelectorAll(".chip").forEach((chip) => {
    const key = chip.dataset.key;
    if (!key || !wanted.has(key)) return;
    const input = chip.querySelector("input");
    if (input && !input.checked) {
      input.checked = true;
      chip.classList.add("active");
      set.add(key);
    }
  });
}

function makeChip(key, set, labelText) {
  const label = document.createElement("label");
  label.className = "chip";
  label.dataset.key = key;
  label.innerHTML = `<input type="checkbox" value="${escape(key)}"><span>${escape(labelText || key)}</span>`;
  const input = label.querySelector("input");
  input.addEventListener("change", () => {
    label.classList.toggle("active", input.checked);
    if (input.checked) set.add(key);
    else set.delete(key);
    if (set === state.selectedErrorTypes) {
      refreshRowDupBlock();
      refreshMultiTableBlock();
    }
  });
  return label;
}

// Show the row-duplication-specific fieldset only when the operator
// has selected ``row_duplication`` (or no types at all, which means
// "run them all"). Hidden in every other case to keep the form
// uncluttered.
function refreshRowDupBlock() {
  const block = document.getElementById("row-dup-block");
  if (!block) return;
  const sel = state.selectedErrorTypes;
  const relevant = sel.size === 0 || sel.has("row_duplication");
  block.hidden = !relevant;
}

// Show the multi-table fieldset only when the operator has selected
// ``inter_table_inconsistency`` (or no types at all, which means
// "run them all"). The cleaning runner auto-triggers the cross-sheet
// pipeline when that type is in scope and >= 2 sheets share a
// workbook directory.
function refreshMultiTableBlock() {
  const block = document.getElementById("mt-block");
  if (!block) return;
  const sel = state.selectedErrorTypes;
  const relevant = sel.size === 0 || sel.has("inter_table_inconsistency");
  block.hidden = !relevant;
}

function setPlaceholder(id, v) {
  if (v === null || v === undefined) return;
  const el = document.getElementById(id);
  if (el) el.placeholder = String(v);
}

function onModeChange() {
  const mode = $("#mode").value || (state.launcherConfig?.defaults?.experiment?.mode || "");
  const adaptive = mode === "adaptive";
  // The stopping-rule and perturbations fieldsets only make sense in
  // adaptive mode. Dim them (and disable interaction) under "single".
  ["gt-block", "gt-perturb-fieldset"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.opacity = adaptive ? "1" : "0.45";
    el.style.pointerEvents = adaptive ? "auto" : "none";
  });
}

// --------------------------------------------------------- submit / preview

function collectParams() {
  const val = (id) => document.getElementById(id).value.trim() || null;
  const num = (id) => {
    const v = document.getElementById(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const check = (id) => document.getElementById(id).checked;

  const csvDir = $("#csv-dir");
  const selected = csvDir.selectedOptions[0];
  let csv = null, csv_dir = null;
  if (selected && selected.value) {
    if (selected.dataset.type === "file") csv = selected.value;
    else csv_dir = selected.value;
  }

  const errorTypes = [...state.selectedErrorTypes].join(",") || null;
  // When the operator has explicitly chosen one or more error types
  // via the chip selector, bypass the YAML ``prompts.scopes`` filter
  // (which would otherwise silently drop types that don't match the
  // configured scope, e.g. row_duplication under a single-column
  // default). When no chips are selected we leave scopes alone so
  // the YAML default still applies.
  const scopes = errorTypes ? "all" : null;

  const params = {
    csv, csv_dir,
    csv_glob: val("csv-glob"),
    chunk_size: num("chunk-size"),
    max_chunks: num("max-chunks"),
    column_mode: val("column-mode"),
    columns: val("columns"),
    model: val("model"),
    reasoning_effort: val("reasoning-effort"),
    concurrency: num("concurrency"),
    synthesis_effort: val("synthesis-effort"),
    normalize_input: check("normalize-input") || null,
    normalize_fold_punctuation: check("normalize-fold-punctuation") || null,
    normalize_lowercase: check("normalize-lowercase") || null,
    error_types: errorTypes,
    scopes,
    mode: val("mode"),
  };

  // ----- row_duplication overrides (only sent when the fieldset is
  //       visible, which happens when the operator has selected
  //       ``row_duplication`` or no types at all). The detector
  //       falls back to config/experiment.yaml for blank fields.
  //       The full recall-first pipeline (normalise + canonicalise +
  //       blocking + minhash + embedding + adjudicate) is configured
  //       in YAML; only the operational dials are exposed here.
  const rowDupBlock = document.getElementById("row-dup-block");
  const rowDupVisible = rowDupBlock && !rowDupBlock.hidden;
  if (rowDupVisible) {
    params.row_dup_detection_mode = val("row-dup-detection-mode");
    params.row_dup_max_groups_per_source = num("row-dup-max-groups");
  }

  // ----- multi-table cross-sheet overrides (only sent when the
  //       fieldset is visible, which happens when the operator has
  //       selected ``inter_table_inconsistency`` or no types at all).
  //       The pipeline auto-runs whenever the type is in scope; these
  //       knobs let the user reuse groups from a prior discovery run
  //       and pick the step-3 strategy.
  const mtBlock = document.getElementById("mt-block");
  const mtVisible = mtBlock && !mtBlock.hidden;
  if (mtVisible) {
    params.multi_table_from = val("mt-from");
    params.multi_table_strategy = val("mt-strategy");
    params.multi_table_semantic_rows = num("mt-semantic-rows");
    params.multi_table_semantic_max_pages = num("mt-semantic-max-pages");
    params.no_multi_table_verify = check("mt-no-verify") || null;
    // Multi-table discovery follows the global ``mode`` (adaptive/single);
    // no separate per-discovery knobs.
  }

  if (params.mode === "adaptive") {
    Object.assign(params, {
      gt_mode: val("gt-mode"),
      gt_num_runs: num("gt-num-runs"),
      gt_min_runs: num("gt-min-runs"),
      gt_target_missing_mass: num("gt-target-missing-mass"),
      gt_stop_scope: val("gt-stop-scope"),
      gt_base_seed: num("gt-base-seed"),
      gt_shuffle_rows: check("gt-shuffle-rows") ? true : null,
      gt_shuffle_columns: check("gt-shuffle-columns") ? true : null,
      gt_prompt_ensemble: check("gt-prompt-ensemble") ? true : null,
      gt_skip_identity_run: check("gt-skip-identity-run") ? true : null,
      gt_variant_effort: val("gt-variant-effort"),
      // Scope-aware per-arm overrides. The "two consecutive" select
      // is tri-state: empty -> defer to YAML, "true"/"false" -> set.
      // Note: per-arm epsilon / k_min / k_max are not exposed here --
      // they are unified with target_missing_mass / min_runs /
      // num_runs above.
      gt_per_arm_two_consecutive: (() => {
        const v = val("gt-per-arm-two-consecutive");
        if (v === "true") return true;
        if (v === "false") return false;
        return null;
      })(),
      gt_per_arm_stop_when: val("gt-per-arm-stop-when"),
      gt_per_arm_epsilon_chao1: num("gt-per-arm-epsilon-chao1"),
    });
  }

  // Drop nulls.
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== null && v !== ""));
}

function previewCommand() {
  const params = collectParams();
  // Mimic build_detection_argv shape.
  const pairs = [];
  const map = {
    mode: "--mode", csv: "--csv", csv_dir: "--csv-dir", csv_glob: "--csv-glob",
    model: "--model", chunk_size: "--chunk-size", max_chunks: "--max-chunks",
    column_mode: "--column-mode", columns: "--columns",
    reasoning_effort: "--reasoning-effort", concurrency: "--concurrency",
    error_types: "--error-types",
    scopes: "--scopes",
    synthesis_effort: "--synthesis-effort",
    gt_mode: "--gt-mode", gt_num_runs: "--gt-num-runs",
    gt_min_runs: "--gt-min-runs", gt_target_missing_mass: "--gt-target-missing-mass",
    gt_stop_scope: "--gt-stop-scope",
    gt_base_seed: "--gt-base-seed", gt_variant_effort: "--gt-variant-effort",
    gt_per_arm_stop_when: "--gt-per-arm-stop-when",
    gt_per_arm_epsilon_chao1: "--gt-per-arm-epsilon-chao1",
    row_dup_detection_mode: "--row-dup-detection-mode",
    row_dup_max_groups_per_source: "--row-dup-max-groups-per-source",
    multi_table_from: "--multi-table-from",
    multi_table_strategy: "--multi-table-strategy",
    multi_table_semantic_rows: "--multi-table-semantic-rows",
    multi_table_semantic_max_pages: "--multi-table-semantic-max-pages",
  };
  const boolMap = {
    normalize_input: ["--normalize-input", null],
    normalize_fold_punctuation: ["--normalize-fold-punctuation", null],
    normalize_lowercase: ["--normalize-lowercase", null],
    gt_shuffle_rows: ["--gt-shuffle-rows", "--no-gt-shuffle-rows"],
    gt_shuffle_columns: ["--gt-shuffle-columns", "--no-gt-shuffle-columns"],
    gt_prompt_ensemble: ["--gt-prompt-ensemble", "--no-gt-prompt-ensemble"],
    gt_skip_identity_run: ["--gt-skip-identity-run", "--no-gt-skip-identity-run"],
    gt_per_arm_two_consecutive: [
      "--gt-per-arm-two-consecutive",
      "--no-gt-per-arm-two-consecutive",
    ],
    no_multi_table_verify: ["--no-multi-table-verify", null],
  };
  for (const [k, v] of Object.entries(params)) {
    if (k in boolMap) {
      pairs.push(v ? boolMap[k][0] : boolMap[k][1]);
    } else if (k in map) {
      pairs.push(`${map[k]} ${JSON.stringify(v)}`);
    }
  }
  const el = $("#cmd-preview");
  el.hidden = false;
  el.textContent = "python -m experiments.run_cleaning_detection \\\n    " + pairs.filter(Boolean).join(" \\\n    ");
}

async function onSubmit(e) {
  e.preventDefault();
  const params = collectParams();
  try {
    const job = await api("/api/launcher/jobs", {
      method: "POST",
      body: JSON.stringify({ kind: "detection", params }),
    });
    toast(`run launched: ${job.job_id}`);
    showTab("jobs");
    openJob(job.job_id);
    refreshJobs();
  } catch (err) {
    toast(`launch failed: ${err.message}`, 4000);
  }
}

// --------------------------------------------------------- jobs

async function refreshJobs() {
  try {
    const r = await api("/api/launcher/jobs");
    renderJobs(r.jobs || []);
  } catch (e) {
    // Server might be restarting; ignore.
  }
}

function renderJobs(jobs) {
  const running = jobs.filter((j) => j.status === "running").length;
  const badge = $("#jobs-badge");
  if (running > 0) {
    badge.hidden = false;
    badge.textContent = String(running);
  } else {
    badge.hidden = true;
  }

  const box = $("#jobs-list");
  if (!jobs.length) {
    box.innerHTML = `<div class="empty-state"><h3>No jobs yet</h3><p>Jobs appear here when you launch a run from the <b>Launch</b> tab.</p></div>`;
    return;
  }
  box.innerHTML = jobs.map((j) => {
    const rec = j.recovered ? ` <span class="tag" style="font-size:9.5px;padding:1px 6px;border-radius:4px;background:var(--panel-2);border:1px solid var(--border);color:#f59e0b;margin-left:4px;">recovered</span>` : "";
    const deleteBtn = j.status === "running"
      ? ""
      : `<button class="danger" data-action="delete">Delete</button>`;
    return `
    <div class="row" data-job="${escape(j.job_id)}">
      <div>
        <div class="kind">${escape(j.kind)}${rec}</div>
        <div class="ts">${escape(j.job_id)}</div>
      </div>
      <div><span class="status ${escape(j.status)}">${escape(j.status)}${j.exit_code != null ? ` · ${j.exit_code}` : ""}</span></div>
      <div class="argv" title="${escape(j.argv.join(' '))}">${escape(j.argv.slice(1).join(' '))}</div>
      <div class="ts">${escape(j.started_at?.slice(0, 19).replace("T", " ") || "")}</div>
      <div class="actions">
        <button data-action="open">Open</button>
        ${deleteBtn}
      </div>
    </div>
  `;
  }).join("");

  box.querySelectorAll('[data-action="open"]').forEach((b) => {
    b.addEventListener("click", () => {
      const row = b.closest(".row");
      openJob(row.dataset.job);
    });
  });

  box.querySelectorAll('[data-action="delete"]').forEach((b) => {
    b.addEventListener("click", async (e) => {
      e.stopPropagation();
      const row = b.closest(".row");
      const jobId = row.dataset.job;
      if (!confirm(`Delete job "${jobId}"?\n\nThis removes its log and metadata. The detection outputs (under outputs/) are kept.`)) {
        return;
      }
      b.disabled = true;
      b.textContent = "Deleting…";
      try {
        await api(`/api/launcher/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
        toast(`deleted job ${jobId}`);
        if (state.activeJobId === jobId) {
          $("#job-close").click?.();
        }
        refreshJobs();
      } catch (err) {
        toast(`delete failed: ${err.message}`, 4000);
        b.disabled = false;
        b.textContent = "Delete";
      }
    });
  });

  // Auto-refresh detail for active job.
  if (state.activeJobId) {
    const active = jobs.find((j) => j.job_id === state.activeJobId);
    if (active) updateJobMeta(active);
  }
}

async function openJob(jobId) {
  state.activeJobId = jobId;
  state.logOffset = 0;
  state.logCommitted = "";
  state.logPending = "";
  $("#job-detail").hidden = false;
  $("#job-detail-title").textContent = `Job ${jobId}`;
  $("#job-log").textContent = "";
  $("#job-cancel").onclick = async () => {
    try { await api(`/api/launcher/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }); toast("cancel sent"); }
    catch (e) { toast(`cancel failed: ${e.message}`, 3000); }
  };
  $("#job-close").onclick = () => {
    $("#job-detail").hidden = true;
    state.activeJobId = null;
    clearInterval(state.jobPoller);
    state.jobPoller = null;
  };
  const meta = await api(`/api/launcher/jobs/${encodeURIComponent(jobId)}`);
  updateJobMeta(meta);
  await pollLog();
  clearInterval(state.jobPoller);
  state.jobPoller = setInterval(pollLog, 2000);
}

function updateJobMeta(job) {
  const box = $("#job-detail-meta");
  // Single label entry point: ``label.html`` hosts every error type
  // (per-cell typo, multi-table findings, and inconsistency clusters
  // surfaced from the column-clusters endpoint). The page copes
  // gracefully when the run has nothing in a given category.
  const linkHTML = job.run_id
    ? ` · <a href="./label.html?run=${encodeURIComponent(job.run_id)}" title="Label this run">Label</a>` +
      ` · <a href="/api/runs/${encodeURIComponent(job.run_id)}/report" target="_blank" rel="noopener">Report</a>`
    : "";
  box.innerHTML = `<span class="status ${escape(job.status)}">${escape(job.status)}</span>` +
    ` <span>started ${escape(job.started_at || "")}</span>` +
    (job.ended_at ? ` <span>· ended ${escape(job.ended_at)}</span>` : "") +
    (job.run_id ? ` <span>· run <code>${escape(job.run_id)}</code></span>` : "") +
    linkHTML;
}

async function pollLog() {
  if (!state.activeJobId) return;
  try {
    const r = await api(`/api/launcher/jobs/${encodeURIComponent(state.activeJobId)}/log?offset=${state.logOffset}`);
    if (r.text) {
      const logEl = $("#job-log");
      const wasBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
      appendJobLog(r.text);
      if (wasBottom) logEl.scrollTop = logEl.scrollHeight;
      state.logOffset = r.next_offset;
    }
    if (r.status !== "running") {
      // One last refresh to catch metadata + stop polling.
      const job = await api(`/api/launcher/jobs/${encodeURIComponent(state.activeJobId)}`);
      updateJobMeta(job);
      clearInterval(state.jobPoller);
      state.jobPoller = null;
      refreshWorkspace();
      refreshJobs();
    }
  } catch (e) {
    // Network blip; next tick will retry.
  }
}

// Strip ANSI/CSI escape sequences (tqdm and friends emit colour + cursor codes).
const ANSI_RE = /\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*\x07)/g;

// Apply a chunk of (possibly tqdm-style) text to the log view, collapsing
// carriage-return overwrites onto a single live line so progress bars
// behave like in a real terminal.
function appendJobLog(chunk) {
  const clean = chunk.replace(ANSI_RE, "");
  let pending = state.logPending || "";
  let committed = state.logCommitted;
  for (let i = 0; i < clean.length; i++) {
    const ch = clean[i];
    if (ch === "\r") {
      // \r\n is just a hard newline; otherwise rewind the pending line.
      if (clean[i + 1] === "\n") {
        committed += pending + "\n";
        pending = "";
        i++;
      } else {
        pending = "";
      }
    } else if (ch === "\n") {
      committed += pending + "\n";
      pending = "";
    } else if (ch === "\b") {
      pending = pending.slice(0, -1);
    } else {
      pending += ch;
    }
  }
  state.logCommitted = committed;
  state.logPending = pending;
  $("#job-log").textContent = committed + pending;
}

// --------------------------------------------------------- data prep

// Field mapping for the unified curate launcher. The CSV layout (flat
// vs grouped) is auto-detected server-side by `src/data_prep/curate.py`,
// so a single field schema covers both legacy curate_tables and
// curate_pbi flows. Each row: [payloadKey, domId].
const CURATE_FIELDS = [
  ["source_dir",                "cu-source-dir"],
  ["output_dir",                "cu-output-dir"],
  ["num_targets",               "cu-num-targets"],
  ["min_columns",               "cu-min-columns"],
  ["max_columns",               "cu-max-columns"],
  ["min_rows",                  "cu-min-rows"],
  ["max_rows",                  "cu-max-rows"],
  ["min_tables",                "cu-min-tables"],
  ["max_tables",                "cu-max-tables"],
  ["oversize",                  "cu-oversize"],
  ["seed",                      "cu-seed"],
  ["max_scan",                  "cu-max-scan"],
  ["language_check",            "cu-language-check"],
  ["language_min_english_prob", "cu-language-min-english-prob"],
  ["max_non_latin_chars",       "cu-max-non-latin-chars"],
  ["max_empty_ratio",           "cu-max-empty-ratio"],
  ["max_cell_chars",            "cu-max-cell-chars"],
  ["max_cell_words",            "cu-max-cell-words"],
  ["max_oversized_row_ratio",   "cu-max-oversized-row-ratio"],
  ["dedupe_across_groups",      "cu-dedupe-across-groups"],
];

function initDataPrep() {
  // Populate source-dir dropdown from known datasets.
  const datasets = (state.launcherConfig?.datasets || []).filter((d) => d.type === "dir");
  const sel = document.getElementById("cu-source-dir");
  if (sel) {
    const preferred = sel.dataset.default || "";
    sel.innerHTML = `<option value="">(pick a dir)</option>`;
    for (const ds of datasets) {
      const o = document.createElement("option");
      o.value = ds.path;
      o.textContent = `${ds.name}/  (${ds.csv_count} CSVs)`;
      sel.appendChild(o);
    }
    const match = Array.from(sel.options).find(
      (o) => o.value && (o.value === preferred || o.value.endsWith("/" + preferred.replace(/^data\//, "")))
    );
    if (match) sel.value = match.value;
  }

  // Preview + submit.
  document.querySelector('[data-dp-preview="curate"]')?.addEventListener("click", () => {
    const params = collectDpParams();
    const pre = document.querySelector('[data-dp-cmd="curate"]');
    const parts = ["python", "-m", "experiments.data_prep.curate"];
    for (const [k, v] of Object.entries(params)) {
      if (v === null || v === undefined || v === "") continue;
      if (v === true)  { parts.push("--" + k.replaceAll("_", "-")); continue; }
      if (v === false) continue;
      parts.push("--" + k.replaceAll("_", "-"), String(v));
    }
    pre.hidden = false;
    pre.textContent = parts.join(" \\\n  ");
  });

  document.getElementById("dp-form-curate")?.addEventListener("submit", (e) => {
    e.preventDefault();
    onDpSubmit();
  });
}

function collectDpParams() {
  const out = {};
  for (const [payloadKey, domId] of CURATE_FIELDS) {
    const el = document.getElementById(domId);
    if (!el) continue;
    const raw = el.value.trim();
    if (raw === "") { out[payloadKey] = null; continue; }
    out[payloadKey] = el.type === "number" ? Number(raw) : raw;
  }
  out.overwrite = !!document.getElementById("cu-overwrite")?.checked;
  // Boolean tri-state: only emit when checkbox is unchecked (default = true).
  const dedupeEl = document.getElementById("cu-dedupe-schema-within-group");
  if (dedupeEl) {
    out.dedupe_schema_within_group = !!dedupeEl.checked;
  }
  return out;
}

async function onDpSubmit() {
  const params = collectDpParams();
  try {
    const job = await api("/api/launcher/jobs", {
      method: "POST",
      body: JSON.stringify({ kind: "curate", params }),
    });
    toast(`curate launched: ${job.job_id}`);
    showTab("jobs");
    openJob(job.job_id);
    refreshJobs();
  } catch (err) {
    toast(`launch failed: ${err.message}`, 4000);
  }
}

bootstrap().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#f87171;">bootstrap failed: ${escape(e.stack || e.message)}</pre>`;
});

// =============================================================
// Relation discovery (multi-column relations)
// =============================================================

function renderRelationsRunRow(r) {
  const reportLink = r.has_report
    ? `<a class="primary" href="/api/runs/${encodeURIComponent(r.run_id)}/report" target="_blank" rel="noopener">View relations</a>`
    : `<button data-action="build-relation-report" data-run="${escape(r.run_id)}">Build report</button>`;
  // Empirical-validation button. Always available alongside the
  // report (it's safe to re-run; the script overwrites
  // ``relation_verifications.jsonl``). The button refreshes the
  // dashboard at the end so users don't need a separate "rebuild"
  // step.
  const validateBtn = `<button data-action="validate-relations" data-run="${escape(r.run_id)}" title="Run LLM-written check() code against the data and record pass-rates">Validate</button>`;
  // Review CTA: Step-1 → Step-2 bridge. Operators open this page to
  // approve / reject the discovered relations; the detection runner
  // (``experiments/run_relation_detection``) consumes those verdicts
  // when compiling per-cell violations.
  const reviewLink = `<a class="primary" href="./relations_review.html?run=${encodeURIComponent(r.run_id)}" title="Approve or reject the discovered relations before running detection">Review</a>`;
  // Unified Label CTA: relations runs can produce per-row violation
  // clusters (from the post-validation violation-compilation step in
  // run_relation_discovery) AND multi-table findings. Both are hosted
  // by the same label.html surface as detection runs, so a single
  // link routes there. The suffixes show what's inside.
  // Counts move into the Label tooltip rather than visible suffix text,
  // so the button reads as a clean verb (consistent with detection rows).
  const labelCounts = [
    r.has_clusters ? `${Number(r.clusters ?? 0)} relation clusters` : "",
    r.has_multi_table ? `${Number(r.n_multi_table_findings ?? 0)} multi-table findings` : "",
  ].filter(Boolean).join(", ");
  const labelLink = (r.has_clusters || r.has_multi_table)
    ? `<a class="primary" href="./label.html?run=${encodeURIComponent(r.run_id)}" title="Label findings in this run (relation violations + cross-sheet)${labelCounts ? ` — ${escape(labelCounts)}` : ""}">Label</a>`
    : "";
  const tag = (cls, value, title) => value
    ? `<span class="tag ${cls}"${title ? ` title="${escape(title)}"` : ""}>${escape(String(value))}</span>`
    : "";
  const tags = [
    `<span class="tag" style="background:rgba(167,139,250,0.14);color:#c4b5fd;border-color:rgba(167,139,250,0.45);">relations</span>`,
    tag("dataset", r.dataset, `dataset: ${r.dataset || ""}`),
    tag("model", r.model),
  ].filter(Boolean).join("");
  const { checkbox, starBtn, nameBlock, selected } = renderRowLead(r, tags);
  const rowClass = [
    "row",
    r.starred ? "starred" : "",
    selected ? "row-selected" : "",
  ].filter(Boolean).join(" ");
  return `
    <div class="${rowClass}">
      ${checkbox}
      ${starBtn}
      ${nameBlock}
      <div class="num"><span class="label">relations</span>${escape(String(r.n_relations ?? 0))}</div>
      <div class="num"><span class="label">sources</span>${escape(String(r.n_sources ?? 0))}</div>
      <div class="actions">
        ${reviewLink}
        ${labelLink}
        ${reportLink}
        ${validateBtn}
        <button class="danger" data-action="delete-run" data-run="${escape(r.run_id)}" title="Delete this run">Delete</button>
      </div>
    </div>
  `;
}

function initRelationsForm() {
  const cfg = state.launcherConfig || {};

  // Dataset picker (same shape as detection launcher).
  const sel = $("#rel-csv-dir");
  if (sel) {
    sel.innerHTML = `<option value="">(use config default)</option>`;
    for (const ds of cfg.datasets || []) {
      const o = document.createElement("option");
      o.value = ds.path;
      o.dataset.type = ds.type;
      o.textContent = ds.type === "dir"
        ? `${ds.name}/  (${ds.csv_count} CSVs)`
        : `${ds.name}  (1 file)`;
      sel.appendChild(o);
    }
  }

  // Multi-column relation types only. The backend exposes the
  // whitelist on /api/launcher/config.relations_error_types — we
  // reuse the existing chip / et-group machinery from the
  // detection launcher so the styling is identical.
  state.relationsErrorTypes = state.relationsErrorTypes || new Set();
  const allowed = new Set(cfg.relations_error_types || []);
  const scopes = cfg.relations_error_type_scopes || {};
  const etBox = $("#rel-error-types");
  if (etBox) {
    etBox.innerHTML = "";
    const matched = (cfg.error_types || []).filter((et) =>
      !allowed.size || allowed.has(et.key)
    );
    if (!matched.length) {
      etBox.innerHTML = `<div class="muted" style="font-size: 12px; padding: 8px 2px;">No discovery templates configured. Add types under <code>prompts/discover_relations.yaml</code>.</div>`;
    } else {
      const groups = new Map(); // primary-category -> [et, ...]
      for (const et of matched) {
        const cat = ((et.category || "Other").split("/")[0] || "Other").trim();
        if (!groups.has(cat)) groups.set(cat, []);
        groups.get(cat).push(et);
      }
      for (const [cat, items] of groups) {
        const group = document.createElement("div");
        group.className = "et-group";
        const head = document.createElement("div");
        head.className = "et-group-head";
        head.innerHTML = `
          <span class="et-group-title">${escape(cat)}</span>
          <span class="et-group-count" data-count>0 / ${items.length}</span>
          <button type="button" class="et-group-all" data-act="toggle">Select all</button>
        `;
        const body = document.createElement("div");
        body.className = "et-group-body";
        const chips = [];
        for (const et of items) {
          const chip = makeChip(et.key, state.relationsErrorTypes, et.name || et.key);
          const scope = scopes[et.key] || "row";
          let title = et.key + (et.name && et.name !== et.key ? ` \u00b7 ${et.name}` : "");
          if (scope === "table") {
            title += "\nMulti-table relation: only fires on workbooks with 2+ sibling CSV sheets.";
            chip.dataset.scope = "table";
          } else if (scope === "group") {
            title += "\nMulti-row relation: requires the full source frame to discover row-to-row patterns.";
            chip.dataset.scope = "group";
          }
          chip.title = title;
          body.appendChild(chip);
          chips.push(chip);
        }
        const countEl = head.querySelector("[data-count]");
        const toggleBtn = head.querySelector('[data-act="toggle"]');
        const refreshGroup = () => {
          const n = chips.filter((c) => c.querySelector("input").checked).length;
          countEl.textContent = `${n} / ${items.length}`;
          countEl.classList.toggle("has-selected", n > 0);
          toggleBtn.textContent = n === items.length ? "Clear all" : "Select all";
        };
        toggleBtn.addEventListener("click", () => {
          const allOn = chips.every((c) => c.querySelector("input").checked);
          chips.forEach((c) => {
            const input = c.querySelector("input");
            const want = !allOn;
            if (input.checked !== want) {
              input.checked = want;
              input.dispatchEvent(new Event("change"));
            }
          });
          refreshGroup();
        });
        chips.forEach((c) => {
          c.querySelector("input").addEventListener("change", refreshGroup);
        });
        refreshGroup();
        group.appendChild(head);
        group.appendChild(body);
        etBox.appendChild(group);
      }
    }
  }

  // Placeholders from config.yaml defaults.
  const defaults = cfg.defaults || {};
  setPlaceholder("rel-model", defaults.llm?.model);
  setPlaceholder("rel-csv-glob", defaults.data?.csv_glob);
  setPlaceholder("rel-reasoning-effort", defaults.llm?.reasoning_effort);

  $("#relations-form")?.addEventListener("submit", onRelSubmit);
  $("#rel-show-cmd")?.addEventListener("click", previewRelCommand);

  // Adaptive checkbox: highlight the block when on, dim the
  // per-arm controls when off (the same fields are still tracked
  // in collectRelParams, which only emits them when adaptive=true).
  const adapt = $("#rel-adaptive");
  const block = $("#rel-adaptive-block");
  const adaptiveFields = [
    "rel-max-passes", "rel-min-passes", "rel-gt-threshold",
    "rel-stop-when", "rel-epsilon-chao1", "rel-base-seed",
  ];
  function syncAdaptive() {
    const on = !!adapt?.checked;
    block?.classList.toggle("is-on", on);
    for (const id of adaptiveFields) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.disabled = !on;
      const wrap = el.closest("label");
      if (wrap) wrap.classList.toggle("field-disabled", !on);
    }
  }
  adapt?.addEventListener("change", syncAdaptive);
  syncAdaptive();

  // Post-discovery pass blocks: when the master checkbox is off,
  // grey out the per-pass min-cluster input so it's clear the value
  // won't be used.
  function bindPostPass(checkboxId, blockId, fieldIds) {
    const cb = document.getElementById(checkboxId);
    const block = document.getElementById(blockId);
    function sync() {
      const on = !!cb?.checked;
      block?.classList.toggle("is-on", on);
      for (const id of fieldIds) {
        const el = document.getElementById(id);
        if (!el) continue;
        el.disabled = !on;
        const wrap = el.closest("label");
        if (wrap) wrap.classList.toggle("field-disabled", !on);
      }
    }
    cb?.addEventListener("change", sync);
    sync();
  }
  bindPostPass("rel-aggregate", "rel-aggregate-block",
               ["rel-aggregate-min-cluster"]);
  bindPostPass("rel-validate", "rel-validate-block",
               ["rel-validate-layer", "rel-validate-limit",
                "rel-validate-sample-rows", "rel-validate-timeout",
                "rel-validate-max-violations", "rel-validate-prompt"]);
}

function collectRelParams() {
  const val = (id) => document.getElementById(id)?.value.trim() || null;
  const num = (id) => {
    const v = document.getElementById(id)?.value.trim();
    return v == null || v === "" ? null : Number(v);
  };
  const check = (id) => !!document.getElementById(id)?.checked;

  const csvDir = $("#rel-csv-dir");
  const selected = csvDir?.selectedOptions[0];
  let csv = null, csv_dir = null;
  if (selected && selected.value) {
    if (selected.dataset.type === "file") csv = selected.value;
    else csv_dir = selected.value;
  }

  const errorTypes = [...(state.relationsErrorTypes || [])].join(",") || null;
  const adaptive = check("rel-adaptive");
  // Post-discovery passes. Default ON to match the CLI; user opts out
  // by un-checking. We send an explicit boolean so the backend can
  // map it to ``--no-aggregate`` cleanly.
  const aggEl = document.getElementById("rel-aggregate");
  const valEl = document.getElementById("rel-validate");
  const aggregate = aggEl ? aggEl.checked : true;
  const validate = valEl ? valEl.checked : true;
  const params = {
    csv,
    csv_dir,
    csv_glob: val("rel-csv-glob"),
    model: val("rel-model"),
    reasoning_effort: val("rel-reasoning-effort"),
    concurrency: num("rel-concurrency"),
    n_row_samples: num("rel-n-row-samples"),
    error_types: errorTypes,
    include_raw: check("rel-include-raw") || null,
    adaptive: adaptive || null,
    max_passes: adaptive ? num("rel-max-passes") : null,
    min_passes: adaptive ? num("rel-min-passes") : null,
    gt_threshold: adaptive ? num("rel-gt-threshold") : null,
    stop_when: adaptive ? val("rel-stop-when") : null,
    epsilon_chao1: adaptive ? num("rel-epsilon-chao1") : null,
    base_seed: adaptive ? num("rel-base-seed") : null,
    aggregate: aggregate,
    aggregate_min_cluster: aggregate ? num("rel-aggregate-min-cluster") : null,
    validate: validate,
    validate_layer: validate ? val("rel-validate-layer") : null,
    validate_limit: validate ? num("rel-validate-limit") : null,
    validate_sample_rows: validate ? num("rel-validate-sample-rows") : null,
    validate_timeout: validate ? num("rel-validate-timeout") : null,
    validate_max_violations: validate ? num("rel-validate-max-violations") : null,
    validate_prompt: validate ? val("rel-validate-prompt") : null,
  };
  return Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== null && v !== "")
  );
}

function previewRelCommand() {
  const p = collectRelParams();
  const map = {
    csv: "--csv", csv_dir: "--csv-dir", csv_glob: "--csv-glob",
    model: "--model", reasoning_effort: "--reasoning-effort",
    concurrency: "--concurrency", n_row_samples: "--n-row-samples",
    error_types: "--error-types",
    max_passes: "--max-passes", min_passes: "--min-passes",
    gt_threshold: "--gt-threshold", base_seed: "--base-seed",
    stop_when: "--stop-when", epsilon_chao1: "--epsilon-chao1",
    aggregate_min_cluster: "--aggregate-min-cluster",
  };
  const flags = [];
  for (const [k, v] of Object.entries(p)) {
    if (k === "adaptive" && v) flags.push("--adaptive");
    else if (k === "include_raw" && v) flags.push("--include-raw");
    else if (k === "aggregate") {
      if (v === false) flags.push("--no-aggregate");
    }
    else if (k in map) flags.push(`${map[k]} ${JSON.stringify(v)}`);
  }
  const el = $("#rel-cmd-preview");
  if (!el) return;
  el.hidden = false;
  el.textContent = "python -m experiments.run_relation_discovery \\\n    "
    + flags.join(" \\\n    ");
}

async function onRelSubmit(e) {
  e.preventDefault();
  const params = collectRelParams();
  try {
    const job = await api("/api/launcher/jobs", {
      method: "POST",
      body: JSON.stringify({ kind: "relations", params }),
    });
    toast(`relation discovery launched: ${job.job_id}`);
    showTab("jobs");
    openJob(job.job_id);
    refreshJobs();
  } catch (err) {
    toast(`launch failed: ${err.message}`, 4000);
  }
}

// ===== Step 2: Detect violations from approved relations =====
//
// Reuses the relations-discovery runs listed in the registry as the
// pool of source runs the operator can pick from. The launcher posts
// to ``kind: "relation_detection"``; the server resolves
// ``discovery_run`` (absolute project-relative path) and shells out to
// ``experiments.run_relation_detection``.

async function populateDetectDiscoveryRuns() {
  const sel = $("#rd-discovery-run");
  if (!sel) return;
  try {
    const data = await api("/api/runs");
    const runs = (data?.runs || []).filter((r) => r.kind === "relations");
    if (!runs.length) {
      sel.innerHTML = `<option value="">(no discovery runs available)</option>`;
      return;
    }
    // Pre-fill with the URL ?prefill_discovery_run if provided (from
    // the "Launch detection" link on the review page).
    const url = new URLSearchParams(location.search);
    const prefill = url.get("prefill_discovery_run") || "";
    const opts = [`<option value="">(pick a discovery run)</option>`];
    for (const r of runs) {
      const label = `${r.run_id} · ${r.n_relations ?? 0} relations`;
      const sel_ = r.run_id === prefill ? " selected" : "";
      opts.push(`<option value="${escape(r.run_id)}"${sel_}>${escape(label)}</option>`);
    }
    sel.innerHTML = opts.join("");
  } catch (err) {
    sel.innerHTML = `<option value="">(failed to load: ${escape(err.message)})</option>`;
  }
}

function collectRdParams() {
  // Resolve discovery_run -> absolute project-relative path. The runs
  // endpoint returns run_id (folder under outputs/); the backend's
  // _path_within accepts either absolute or project-relative, so we
  // hand it the relative path under outputs/.
  const runId = ($("#rd-discovery-run")?.value || "").trim();
  const params = {};
  if (runId) params.discovery_run = `outputs/${runId}`;
  const reviewFilter = $("#rd-review-filter")?.value;
  if (reviewFilter) params.review_filter = reviewFilter;
  const errorTypes = ($("#rd-error-types")?.value || "").trim();
  if (errorTypes) params.error_types = errorTypes;
  const model = ($("#rd-model")?.value || "").trim();
  if (model) params.model = model;
  const effort = ($("#rd-reasoning-effort")?.value || "").trim();
  if (effort) params.reasoning_effort = effort;
  const concurrency = $("#rd-concurrency")?.valueAsNumber;
  if (Number.isFinite(concurrency)) params.concurrency = concurrency;
  const minPr = $("#rd-min-pass-rate")?.valueAsNumber;
  if (Number.isFinite(minPr)) params.violations_min_pass_rate = minPr;
  const maxPr = $("#rd-max-pass-rate")?.valueAsNumber;
  if (Number.isFinite(maxPr)) params.violations_max_pass_rate = maxPr;
  const maxPerRule = $("#rd-max-per-rule")?.valueAsNumber;
  if (Number.isFinite(maxPerRule)) params.violations_max_per_rule = maxPerRule;
  const minConf = ($("#rd-min-confidence")?.value || "").trim();
  if (minConf) params.violations_min_confidence = minConf;
  if ($("#rd-no-llm-detect")?.checked) params.no_llm_detect = true;
  if ($("#rd-no-synthesis")?.checked) params.no_synthesis = true;
  return params;
}

function previewRdCmd() {
  const p = collectRdParams();
  const flags = [];
  const positional = p.discovery_run ? [JSON.stringify(p.discovery_run)] : [];
  delete p.discovery_run;
  const map = {
    review_filter: "--review-filter",
    error_types: "--error-types",
    model: "--model",
    reasoning_effort: "--reasoning-effort",
    concurrency: "--concurrency",
    violations_min_pass_rate: "--violations-min-pass-rate",
    violations_max_pass_rate: "--violations-max-pass-rate",
    violations_max_per_rule: "--violations-max-per-rule",
    violations_min_confidence: "--violations-min-confidence",
  };
  for (const [k, v] of Object.entries(p)) {
    if (k === "no_llm_detect" && v) flags.push("--no-llm-detect");
    else if (k === "no_synthesis" && v) flags.push("--no-synthesis");
    else if (k in map) flags.push(`${map[k]} ${JSON.stringify(v)}`);
  }
  const el = $("#rd-cmd-preview");
  if (!el) return;
  el.hidden = false;
  el.textContent = "python -m experiments.run_relation_detection \\\n    "
    + [...positional, ...flags].join(" \\\n    ");
}

async function onRdSubmit(e) {
  e.preventDefault();
  const params = collectRdParams();
  if (!params.discovery_run) {
    toast("pick a discovery run first", 3000);
    return;
  }
  try {
    const job = await api("/api/launcher/jobs", {
      method: "POST",
      body: JSON.stringify({ kind: "relation_detection", params }),
    });
    toast(`relation detection launched: ${job.job_id}`);
    showTab("jobs");
    openJob(job.job_id);
    refreshJobs();
  } catch (err) {
    toast(`launch failed: ${err.message}`, 4000);
  }
}

function initRelationDetectionForm() {
  populateDetectDiscoveryRuns();
  $("#rd-show-cmd")?.addEventListener("click", previewRdCmd);
  $("#rel-detect-form")?.addEventListener("submit", onRdSubmit);
}
