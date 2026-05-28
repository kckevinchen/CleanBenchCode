/* column_board.js — column-clustering labelling UI (Slice 6).
 *
 * Surfaces inconsistency-error labelling: per-column, the operator
 * groups distinct values into "this is the same thing" clusters and
 * picks a canonical surface form. The page is loaded with
 *   ?run=<run_id>
 * and writes through ``POST /api/runs/{run_id}/columns/{src}/{col}/clusters``.
 *
 * State model: a single mutable ``state`` object. Mutators call
 * ``rerenderCenter()`` and/or ``rerenderQueue()`` to repaint.
 * Selection is per-column-load (cleared when the active column
 * changes). Cluster-edit changes set ``state.dirty = true``; Save
 * round-trips through the API and reloads the column so we always
 * see the on-disk truth.
 */

// Override styles.css's overflow:hidden so the body grid can
// constrain panes (we manage scrolling inside each pane explicitly).
document.documentElement.style.height = "100vh";
document.documentElement.style.overflow = "hidden";

import { $, api, escapeHtml as escape } from "./shared/utils.js";

// --------------------------------------------------------- helpers

function fmt(n) { return new Intl.NumberFormat().format(n); }

const _PARAMS = new URLSearchParams(location.search);
const RUN_ID = _PARAMS.get("run") || "";
// Optional deep-link: ?source=<src>&column=<col> auto-opens that
// column after the queue is loaded. Used by the report HTML's
// "Open in column board →" link so the operator lands directly on
// the column they were inspecting in the report.
const INITIAL_SOURCE = _PARAMS.get("source") || "";
const INITIAL_COLUMN = _PARAMS.get("column") || "";

// --------------------------------------------------------- state

const state = {
  run_id: RUN_ID,
  meta: null,           // run summary {run_id, run_dir, sources: [...]}
  queue: [],            // full queue payload
  filter: { source: "", status: "unlabeled", draft: "yes" },
  active: null,         // {source, column}
  // Per-active-column working state:
  values: [],           // [{value, count}]
  // Each cluster carries an explicit ``verdict``:
  //   "accept" - keep on save (default for LLM-proposed and
  //              user-created clusters).
  //   "reject" - LLM proposed this but the operator says no;
  //              the cluster stays visible (so they can change
  //              their mind) but is omitted from the saved
  //              payload.
  // The cluster shape persisted to disk has no ``verdict`` field;
  // it's a UI-only concept (rejected -> not in payload).
  clusters: [],         // [{cluster_id, kind, canonical, target_style, values:[{value,count}], verdict, from_draft, agreement, support_runs}]
  status: "labeled",
  drop_reason: "",
  note: "",
  selection: new Set(),
  showSingletons: false,
  draft: null,          // {agreement avg/runs_completed/...} | null
  loadedRecord: null,   // server-known record at last load
  activeClusterIdx: 0,  // for j/k navigation + 1/2 verdict shortcuts
  dirty: false,
  saving: false,
};

// --------------------------------------------------------- bootstrap

async function bootstrap() {
  if (!RUN_ID) {
    $("#run-meta").textContent = "missing ?run=<run_id> in URL";
    return;
  }
  $("#label-link").href = `./label.html?run=${encodeURIComponent(RUN_ID)}`;
  $("#report-link").href = `/api/runs/${encodeURIComponent(RUN_ID)}/report`;
  bindFilterEvents();
  bindCenterEvents();
  bindKeyboardShortcuts();
  await refreshAll();
  // Honor ?source=&column= deep-link from the report. The queue is
  // gated by a cardinality filter (n_distinct / n_rows <= 0.95 by
  // default), so a report-side cluster on a near-unique column
  // (e.g. "Town Code" with 99/100 distinct) won't appear in the
  // sidebar. We still open the column directly -- the /values and
  // /clusters endpoints have no such filter, so the operator can
  // label it. We also clear the status filter so the queue is at
  // least browseable when the operator finishes with the deep-link
  // column.
  if (INITIAL_SOURCE && INITIAL_COLUMN) {
    const hit = state.queue.find(
      (it) => it.source === INITIAL_SOURCE && it.column === INITIAL_COLUMN,
    );
    state.filter.status = "";
    const statusSel = $("#filter-status");
    if (statusSel) statusSel.value = "";
    if (!hit) {
      // Inject a synthetic queue item so the sidebar shows the
      // active column and the operator can navigate back to it after
      // browsing the rest of the queue.
      state.queue.unshift({
        source: INITIAL_SOURCE,
        column: INITIAL_COLUMN,
        n_rows: 0,
        n_distinct: 0,
        top_values: [],
        status: "unlabeled",
        n_clusters: 0,
        annotator: null,
        ts: null,
        _outside_default_queue: true,
      });
    }
    rerenderQueue();
    await openColumn(INITIAL_SOURCE, INITIAL_COLUMN);
  }
}

async function refreshAll() {
  try {
    const [meta, queue] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(RUN_ID)}`),
      api(`/api/runs/${encodeURIComponent(RUN_ID)}/inconsistency-columns`),
    ]);
    state.meta = meta;
    state.queue = queue.items || [];
    // Surfaced separately by the endpoint so we can show a
    // headline like "775 candidate column(s), 51 with LLM draft".
    state.queueTotalWithDraft =
      typeof queue.total_with_draft === "number"
        ? queue.total_with_draft
        : state.queue.filter((it) => (it.n_draft_clusters || 0) > 0).length;
    renderRunMeta();
    populateSourceFilter();
    rerenderQueue();
  } catch (e) {
    $("#run-meta").textContent = `error: ${e.message}`;
  }
}

function renderRunMeta() {
  const m = state.meta;
  const total = state.queue.length;
  const withDraft = state.queueTotalWithDraft || 0;
  $("#run-meta").innerHTML =
    `<code>${escape(m.run_id)}</code> · ` +
    `${escape((m.sources || []).length)} source(s) · ` +
    `${fmt(total)} candidate column(s) · ` +
    `<strong>${fmt(withDraft)}</strong> with LLM draft`;
}

function populateSourceFilter() {
  const sel = $("#filter-source");
  const sources = new Set();
  for (const item of state.queue) sources.add(item.source);
  const cur = sel.value;
  sel.innerHTML =
    `<option value="">(all)</option>` +
    Array.from(sources).sort().map((s) =>
      `<option value="${escape(s)}"${s === cur ? " selected" : ""}>${escape(s)}</option>`
    ).join("");
}

function bindFilterEvents() {
  ["filter-source", "filter-status", "filter-draft"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      state.filter.source = $("#filter-source").value;
      state.filter.status = $("#filter-status").value;
      state.filter.draft = $("#filter-draft").value;
      rerenderQueue();
    });
  });
}

// --------------------------------------------------------- queue

function filteredQueue() {
  const f = state.filter;
  return state.queue.filter((item) => {
    if (f.source && item.source !== f.source) return false;
    if (f.status && item.status !== f.status) return false;
    if (f.draft === "yes" && !itemHasDraft(item)) return false;
    if (f.draft === "no" && itemHasDraft(item)) return false;
    return true;
  });
}

// Whether the LLM proposed at least one draft cluster for this
// column. Backed by ``n_draft_clusters`` on the queue payload
// (added by /inconsistency-columns), which is populated up-front
// from column_cluster_drafts.jsonl. Older servers that don't
// expose this field will report 0 for every item -- in that case
// the operator can switch the "Has draft" filter to "(any)" to
// see everything.
function itemHasDraft(item) {
  return (item.n_draft_clusters || 0) > 0;
}

function rerenderQueue() {
  const box = $("#queue");
  const items = filteredQueue();
  if (!items.length) {
    box.innerHTML = `<div class="muted" style="padding:12px">no columns match the current filter</div>`;
    return;
  }
  box.innerHTML = items.map((item) => {
    const isActive = state.active &&
      state.active.source === item.source &&
      state.active.column === item.column;
    const ndc = item.n_draft_clusters || 0;
    const draftHint = ndc > 0
      ? `<span class="qi-badge draft" title="LLM proposed ${ndc} cluster(s) for this column">${fmt(ndc)} LLM</span>`
      : "";
    const annot = item.annotator
      ? ` · ${escape(item.annotator)}` : "";
    return `
      <div class="queue-item${isActive ? " active" : ""}"
           data-source="${escape(item.source)}"
           data-column="${escape(item.column)}"
           role="button" tabindex="0">
        <div class="qi-title">${escape(item.column)}</div>
        <div></div>
        <div class="qi-src">${escape(item.source)}</div>
        <div class="qi-meta">
          <span class="qi-badge status-${escape(item.status)}">${escape(item.status)}</span>
          ${draftHint}
          ${fmt(item.n_distinct)} distinct${annot}
        </div>
      </div>
    `;
  }).join("");
  // Per-row click handlers are installed once by bindQueueDelegation()
  // during bootstrap. innerHTML replacement above leaves the parent
  // listener intact so no rebinding is needed.
}

// --------------------------------------------------------- open column

async function openColumn(source, column) {
  if (state.dirty) {
    if (!confirm("Unsaved changes will be lost. Continue?")) return;
  }
  state.active = { source, column };
  state.values = [];
  state.clusters = [];
  state.selection.clear();
  state.dirty = false;
  state.draft = null;
  state.loadedRecord = null;
  $("#cb-title").textContent = `${source} / ${column}`;
  $("#cb-title").classList.remove("muted");
  $("#cb-stats").textContent = "loading…";
  $("#cb-draft-meta").textContent = "";
  // The verdict block is always visible in the new layout; no
  // toggle needed.
  $("#cb-value-list").textContent = "loading…";
  $("#cb-cluster-list").textContent = "";
  rerenderQueue();

  try {
    const sEnc = encodeURIComponent(source);
    // Use the action-before-column URL shape so columns whose names
    // contain ``/`` (e.g. ``Project Name/Description``) still route.
    // The backend's ``{column:path}`` converter accepts slashes only
    // when ``column`` is the trailing path segment, hence the
    // ``…/columns/{source}/<action>/{column}`` order.
    // We deliberately do NOT call ``encodeURIComponent`` on the
    // column here because the browser's ``fetch`` will escape any
    // unsafe characters in the URL, and ``encodeURIComponent``
    // would turn ``/`` into ``%2F`` which Starlette then decodes
    // back to ``/`` before path matching -- same 404 problem.
    // The literal ``/`` is what the ``:path`` converter wants.
    const ru = encodeURIComponent(RUN_ID);
    const valuesUrl = `/api/runs/${ru}/columns/${sEnc}/values/${encodeURI(column)}?limit=5000`;
    const clustersUrl = `/api/runs/${ru}/columns/${sEnc}/clusters/${encodeURI(column)}`;
    const draftUrl = `/api/runs/${ru}/columns/${sEnc}/draft/${encodeURI(column)}`;
    const [vals, recs, draftBlob] = await Promise.all([
      api(valuesUrl),
      api(clustersUrl),
      api(draftUrl),
    ]);
    state.values = vals.values || [];
    $("#cb-stats").innerHTML =
      `${fmt(vals.n_distinct)} distinct · ${fmt(vals.n_rows)} rows` +
      (vals.n_empty ? ` · ${fmt(vals.n_empty)} empty` : "") +
      (vals.truncated ? ` · <strong>truncated</strong> at ${fmt(vals.values.length)}` : "");

    state.loadedRecord = recs.record || null;
    state.draft = draftBlob.draft || null;
    // Keep the queue item's draft count in sync with what the
    // /draft endpoint actually returned, so a refresh doesn't
    // re-sort the queue under us if the index was stale.
    const qi = state.queue.find(
      (it) => it.source === source && it.column === column,
    );
    if (qi) {
      qi.n_draft_clusters = state.draft
        ? (state.draft.draft_clusters || []).length
        : 0;
    }

    if (state.loadedRecord) {
      // Hydrate from server: cluster ids are stable strings; values
      // arrive as ``[{value, count}]`` per the schema. Any cluster
      // already on disk is implicitly accepted -- the operator can
      // still reject it after the fact (which will remove it on the
      // next save).
      state.status = state.loadedRecord.status;
      state.drop_reason = state.loadedRecord.drop_reason || "";
      state.note = state.loadedRecord.note || "";
      // ``target_style`` is not on the persisted record (it lives in
      // the LLM draft), so cross-reference each saved cluster's
      // value-set against state.draft to recover the hint when the
      // value-sets match exactly. Mirrors the server-side join in
      // ``list_column_clusters``.
      const draftByValueSet = new Map();
      for (const dc of (state.draft && state.draft.draft_clusters) || []) {
        const key = (dc.values || []).slice().sort().join("\u0001");
        if (typeof dc.target_style === "string" && dc.target_style) {
          draftByValueSet.set(key, dc.target_style);
        }
      }
      state.clusters = (state.loadedRecord.clusters || []).map((c) => {
        const key = (c.values || []).map((v) => v.value).sort().join("\u0001");
        return {
          cluster_id: c.cluster_id,
          kind: normalizeKind(c.kind),
          canonical: c.canonical_value,
          target_style: draftByValueSet.get(key) || null,
          values: (c.values || []).map((v) => ({ value: v.value, count: v.count })),
          from_draft: false,
          verdict: "accept",
          agreement: null,
          support_runs: null,
        };
      });
    } else if (state.draft && (state.draft.draft_clusters || []).length) {
      // Seed clusters from the LLM draft. We surface *every* draft
      // cluster as a card so the label page (which lists each draft
      // independently) and this edit page agree on the candidate
      // set. Overlapping drafts are seeded with ``verdict: "reject"``
      // so the default save payload is still a disjoint partition
      // (the server enforces disjointness in ``ColumnClustersIn``).
      // The operator can flip a rejected card to accept and reject
      // the conflicting one to swap which interpretation is saved.
      state.clusters = seedFromDraft(state.draft, state.values);
      state.status = "labeled";
      state.drop_reason = "";
      state.note = "";
    } else {
      state.status = "labeled";
      state.drop_reason = "";
      state.note = "";
    }

    if (state.draft) {
      const dc = (state.draft.draft_clusters || []).length;
      $("#cb-draft-meta").innerHTML =
        `· <span class="qi-badge draft">draft</span> ` +
        `${dc} candidate cluster(s) · ` +
        `${state.draft.runs_completed} runs · ` +
        `stopped=${escape(state.draft.stopped_reason)} · ` +
        `GT=${(state.draft.gt_missing_mass ?? 0).toFixed(3)}`;
    } else {
      $("#cb-draft-meta").textContent = "· no LLM draft for this column";
    }
    state.activeClusterIdx = state.clusters.length ? 0 : -1;
    rerenderCenter();
  } catch (e) {
    $("#cb-stats").textContent = `error: ${e.message}`;
  }
}

function seedFromDraft(draft, values) {
  // The draft's clusters may overlap (e.g. {Lincoln HS, LHS} and
  // {Lincoln HS, Lincoln High School}). To keep this edit page in
  // step with the label page -- which surfaces *every* draft cluster
  // as its own labelable item -- we render every draft cluster as a
  // card here too. Disjointness is preserved in the save payload by
  // seeding the verdict: the highest-agreement non-overlapping
  // subset gets ``verdict: "accept"`` (their values flow into the
  // saved record) and any later card that would re-claim a value
  // gets ``verdict: "reject"`` (rendered but omitted from the save
  // payload). The operator can swap which interpretation wins by
  // clicking Accept on a rejected card and Reject on the conflicting
  // accepted one. The server's disjointness check in
  // ``ColumnClustersIn`` is the last line of defence: if the
  // operator accepts two clusters that share a value, the save
  // round-trips a 422 with the exact conflict.
  const cand = (draft.draft_clusters || [])
    .filter((c) => (c.values || []).length >= 2)
    .map((c) => ({
      values: c.values.slice(),
      kind: normalizeKind(c.kind),
      canonical: c.canonical || c.values[0],
      target_style: (typeof c.target_style === "string" && c.target_style) ? c.target_style : null,
      agreement: c.agreement || 0,
      support_runs: (c.support_runs || []).slice(),
    }))
    .sort((a, b) =>
      (b.agreement - a.agreement) || (b.values.length - a.values.length),
    );
  // Tracks which already-accepted card each value was first claimed
  // by, so a rejected overlapping card can name the conflict.
  const claimedBy = new Map();
  const valueCounts = new Map(values.map((v) => [v.value, v.count]));
  const out = [];
  let nextId = 1;
  for (const c of cand) {
    // Drop hallucinated values (LLM may emit ones not in the live
    // column). Reject empty clusters.
    const liveValues = c.values.filter((v) => valueCounts.has(v));
    if (liveValues.length < 2) continue;
    // Does this draft cluster overlap any earlier accepted card?
    const overlapWith = new Set();
    for (const v of liveValues) {
      const prior = claimedBy.get(v);
      if (prior) overlapWith.add(prior);
    }
    const isAccept = overlapWith.size === 0;
    const cid = `c${nextId++}`;
    if (isAccept) {
      for (const v of liveValues) claimedBy.set(v, cid);
    }
    const canonical = liveValues.includes(c.canonical) ? c.canonical : liveValues[0];
    out.push({
      cluster_id: cid,
      kind: c.kind,
      canonical,
      target_style: c.target_style,
      values: liveValues.map((v) => ({ value: v, count: valueCounts.get(v) || 0 })),
      from_draft: true,
      // Disjoint-first cards default to "accept"; overlapping cards
      // default to "reject" so the save payload is valid out of the
      // box. The operator can flip either way; the card is rendered
      // either way.
      verdict: isAccept ? "accept" : "reject",
      // Names of earlier accepted cards this one would conflict
      // with. Rendered as an inline hint and used to explain why
      // the card auto-defaulted to "reject".
      overlap_with: isAccept ? [] : Array.from(overlapWith),
      agreement: c.agreement,
      support_runs: c.support_runs,
    });
  }
  return out;
}

// --------------------------------------------------------- center pane

function bindCenterEvents() {
  $("#cb-value-search").addEventListener("input", rerenderValues);
  $("#cb-make-cluster").addEventListener("click", makeClusterFromSelection);
  $("#cb-clear-selection").addEventListener("click", () => {
    state.selection.clear();
    rerenderValues();
  });
  $("#cb-status").addEventListener("change", () => {
    state.status = $("#cb-status").value;
    state.dirty = true;
    $("#cb-drop-reason-cell").hidden =
      !state.status.startsWith("dropped_");
    setSaveMsg("");
  });
  $("#cb-drop-reason").addEventListener("input", () => {
    state.drop_reason = $("#cb-drop-reason").value;
    state.dirty = true;
  });
  $("#cb-note").addEventListener("input", () => {
    state.note = $("#cb-note").value;
    state.dirty = true;
  });
  $("#cb-save").addEventListener("click", onSave);
  $("#cb-accept-all").addEventListener("click", () => {
    for (const c of state.clusters) c.verdict = "accept";
    state.dirty = true;
    setSaveMsg("");
    rerenderClusters();
  });
  $("#cb-reject-all").addEventListener("click", () => {
    for (const c of state.clusters) {
      // Only flip LLM proposals -- the operator's own clusters
      // shouldn't get bulk-rejected (they made them deliberately).
      if (c.from_draft) c.verdict = "reject";
    }
    state.dirty = true;
    setSaveMsg("");
    rerenderClusters();
  });
  // Event delegation for re-rendered lists. ``rerenderQueue``,
  // ``rerenderValues`` and ``rerenderClusters`` replace innerHTML on
  // every paint; binding per-row listeners inside each rerender did
  // O(rows) DOM work on every verdict click or filter keystroke.
  // Install ONE listener per parent here (bindCenterEvents runs once
  // during bootstrap) and route via ``closest()``.
  bindQueueDelegation();
  bindValueListDelegation();
  bindClusterListDelegation();
}

// Keyboard shortcuts: j/k navigate the cluster cards, 1/2 verdict
// the active card, Enter saves. We intentionally don't bind any
// shortcut that the operator would expect to type into an input
// (the filter / value-search / note fields trap their own keys).
function bindKeyboardShortcuts() {
  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key;
    if (k === "j" || k === "k") {
      if (!state.clusters.length) return;
      const dir = k === "j" ? 1 : -1;
      let next = (state.activeClusterIdx >= 0 ? state.activeClusterIdx : 0) + dir;
      next = Math.max(0, Math.min(state.clusters.length - 1, next));
      state.activeClusterIdx = next;
      rerenderClusters();
      // Scroll the active card into view so j/k navigation works
      // on long proposal lists.
      const card = document.querySelector(".cb-cluster-card.active");
      if (card) card.scrollIntoView({ block: "nearest" });
      ev.preventDefault();
      return;
    }
    if ((k === "1" || k === "2") && state.activeClusterIdx >= 0) {
      const c = state.clusters[state.activeClusterIdx];
      if (!c) return;
      c.verdict = (k === "1") ? "accept" : "reject";
      state.dirty = true;
      setSaveMsg("");
      rerenderClusters();
      ev.preventDefault();
      return;
    }
    if (k === "Enter") {
      onSave();
      ev.preventDefault();
    }
  });
}

function rerenderCenter() {
  rerenderValues();
  rerenderClusters();
  $("#cb-status").value = state.status;
  $("#cb-drop-reason").value = state.drop_reason;
  $("#cb-note").value = state.note;
  $("#cb-drop-reason-cell").hidden = !state.status.startsWith("dropped_");
  setSaveMsg("");
}

function valueClusterMembership() {
  // Returns Map<value, cluster_id>.
  const m = new Map();
  for (const c of state.clusters) {
    for (const v of c.values) m.set(v.value, c.cluster_id);
  }
  return m;
}

function rerenderValues() {
  const list = $("#cb-value-list");
  const q = ($("#cb-value-search").value || "").trim().toLowerCase();
  const membership = valueClusterMembership();
  const rows = [];
  const singletons = [];
  for (const v of state.values) {
    if (q && !v.value.toLowerCase().includes(q)) continue;
    const target = (v.count <= 1 && !state.selection.has(v.value))
      ? singletons : rows;
    target.push(v);
  }
  const renderRow = (v) => {
    const inCluster = membership.get(v.value);
    const sel = state.selection.has(v.value);
    const cls = "cb-value-row"
      + (sel ? " selected" : "")
      + (inCluster && !sel ? " in-cluster" : "");
    const cref = inCluster
      ? `<span class="v-cluster-ref">→ ${escape(inCluster)}</span>` : "";
    return `<div class="${cls}" data-value="${escape(v.value)}">
      <span class="v-text">${escape(v.value || "(empty)")}</span>
      ${cref}
      <span class="v-count">${fmt(v.count)}</span>
    </div>`;
  };
  let html = rows.map(renderRow).join("");
  if (singletons.length) {
    html += `<details class="cb-singletons">
      <summary class="muted">${fmt(singletons.length)} rare value(s) (count = 1) — usually safe to skip</summary>
      ${singletons.map(renderRow).join("")}
    </details>`;
  }
  if (!html) {
    html = `<div class="muted">no values match the filter</div>`;
  }
  list.innerHTML = html;
  // Per-row click handlers are installed once by bindValueListDelegation()
  // during bootstrap; no per-rerender rebinding needed.
  const actions = $("#cb-value-actions");
  actions.hidden = !state.selection.size;
  $("#cb-selected-count").textContent = `${state.selection.size} selected`;
}

function toggleSelect(value) {
  const membership = valueClusterMembership();
  if (membership.has(value)) return;  // already clustered
  if (state.selection.has(value)) state.selection.delete(value);
  else state.selection.add(value);
  rerenderValues();
}

function makeClusterFromSelection() {
  if (state.selection.size < 2) {
    setSaveMsg("Select at least 2 values to form a cluster.", "error");
    return;
  }
  const valueCounts = new Map(state.values.map((v) => [v.value, v.count]));
  const picked = Array.from(state.selection);
  // Pick the most-frequent surface form as the default canonical.
  picked.sort((a, b) => (valueCounts.get(b) || 0) - (valueCounts.get(a) || 0));
  const canonical = picked[0];
  const id = nextClusterId();
  state.clusters.push({
    cluster_id: id,
    kind: "text",
    canonical,
    values: picked.map((v) => ({ value: v, count: valueCounts.get(v) || 0 })),
    from_draft: false,
    verdict: "accept",
    agreement: null,
    support_runs: null,
  });
  // Newly-added user cluster takes focus so the keyboard shortcuts
  // operate on it.
  state.activeClusterIdx = state.clusters.length - 1;
  state.selection.clear();
  state.dirty = true;
  setSaveMsg("");
  rerenderCenter();
}

function nextClusterId() {
  const taken = new Set(state.clusters.map((c) => c.cluster_id));
  for (let i = 1; i < 10000; i++) {
    const cand = `c${i}`;
    if (!taken.has(cand)) return cand;
  }
  return `c${Date.now()}`;
}

// Cluster-kind taxonomy. Aligned with the single-column
// Inconsistency sub-patterns in data/error_type.csv:
// text (entity variants), casing, pattern, punctuation,
// numerical format, untagged. Each tuple is [enum_value, label].
// The labels include a short hint so the dropdown is self-
// documenting on small screens; the prompt + schema docs carry
// the longer per-tag definitions.
const KIND_OPTIONS = [
  ["text", "text (entity variants)"],
  ["casing", "casing"],
  ["pattern", "pattern (template shape)"],
  ["punctuation", "punctuation / whitespace"],
  ["numerical_format", "numerical format"],
  ["untagged", "untagged"],
];

// Legacy -> new taxonomy mapping. Mirrors
// ``src/labeling/schema.py::_LEGACY_KIND_MAPPING`` so persisted
// drafts written under the old vocabulary still surface in the UI
// under the right new tag (instead of all collapsing to "untagged"
// when the <select> can't find a matching <option>).
const LEGACY_KIND_MAP = {
  entity_variant: "text",
  template_shape: "pattern",
  style_convention: "casing",
  encoding_convention: "numerical_format",
};

function normalizeKind(raw) {
  if (!raw || typeof raw !== "string") return "untagged";
  if (LEGACY_KIND_MAP[raw]) return LEGACY_KIND_MAP[raw];
  // Accept anything already in the new vocabulary; fall back to
  // untagged otherwise so the <select> always has a matching option.
  if (KIND_OPTIONS.some(([k]) => k === raw)) return raw;
  return "untagged";
}

function rerenderClusters() {
  const box = $("#cb-cluster-list");
  const meta = $("#cb-proposals-meta");
  const bulk = $("#cb-bulk-actions");
  if (!state.active) {
    box.innerHTML = `<div class="placeholder muted">(no column selected)</div>`;
    meta.textContent = "";
    bulk.hidden = true;
    return;
  }
  if (!state.clusters.length) {
    box.innerHTML = `<div class="placeholder muted">No LLM clusters for this column. Use the value picker on the right to add one manually — or pick a different status (e.g. <em>no_inconsistency</em>) and save.</div>`;
    meta.textContent = "";
    bulk.hidden = true;
    return;
  }
  // Active-cluster index clamping (clusters may be removed).
  if (state.activeClusterIdx >= state.clusters.length) {
    state.activeClusterIdx = state.clusters.length - 1;
  }
  if (state.activeClusterIdx < 0 && state.clusters.length) {
    state.activeClusterIdx = 0;
  }
  const nAcc = state.clusters.filter((c) => c.verdict === "accept").length;
  const nRej = state.clusters.filter((c) => c.verdict === "reject").length;
  const nDraft = state.clusters.filter((c) => c.from_draft).length;
  const parts = [
    `${fmt(state.clusters.length)} cluster(s)`,
    `<span style="color:var(--tp)">${fmt(nAcc)} accept</span>`,
  ];
  if (nRej) parts.push(`<span style="color:var(--fp)">${fmt(nRej)} reject</span>`);
  if (nDraft) parts.push(`${fmt(nDraft)} from LLM`);
  meta.innerHTML = parts.join(" · ");
  bulk.hidden = !nDraft;

  box.innerHTML = state.clusters.map((c, ci) => {
    const isActive = ci === state.activeClusterIdx;
    // Render each cluster's elements as a vertical list of rows
    // (radio | value | count | remove) -- much easier to scan than
    // the previous horizontal pills, especially for sentence-length
    // values. Mirrors the value picker on the right for visual
    // consistency. Sort by count descending so the dominant form
    // is at the top and the rare variants are at the bottom.
    const sortedValues = c.values
      .map((v, originalIdx) => ({ v, originalIdx }))
      .sort((a, b) => (b.v.count - a.v.count) || (a.v.value < b.v.value ? -1 : 1));
    const valueRows = sortedValues.map(({ v }) => {
      const isCanonical = v.value === c.canonical;
      return `
        <div class="cb-cluster-value-row${isCanonical ? " canonical" : ""}">
          <input type="radio" class="v-canonical-radio"
            name="canonical-${escape(c.cluster_id)}"
            data-ci="${ci}" data-value="${escape(v.value)}"
            ${isCanonical ? "checked" : ""}
            title="Mark this value as the canonical form" />
          <span class="v-text" title="${escape(v.value || "(empty)")}">${escape(v.value || "(empty)")}</span>
          <span class="v-count" title="frequency in this column">${fmt(v.count)}</span>
          <button type="button" class="v-remove" data-ci="${ci}" data-value="${escape(v.value)}" title="Remove this value from the cluster">×</button>
        </div>
      `;
    }).join("");
    const kindSel = `<select data-ci="${ci}" class="cb-cluster-kind" title="Cluster kind">
      ${KIND_OPTIONS.map(([k, lbl]) =>
        `<option value="${k}"${k === c.kind ? " selected" : ""}>${lbl}</option>`
      ).join("")}
    </select>`;
    const draftBadge = c.from_draft
      ? `<span class="qi-badge draft" title="Proposed by the LLM">from LLM</span>` : "";
    const agreement = (c.agreement != null)
      ? `<span class="cc-agreement" title="Fraction of LLM passes where this exact cluster appeared">agreement ${(c.agreement * 100).toFixed(0)}%</span>`
      : "";
    // Target-style hint: free-text phrase emitted by the LLM for
    // type-2 (shared-minority-style) clusters describing the
    // column's competing majority style. ``null`` for type-1 alias
    // groups, where ``canonical`` already names the target form, so
    // the row is conditionally rendered.
    const targetStyleLine = c.target_style
      ? `<div class="cc-target-style-line" title="LLM hint: what the column's majority writes these values as"><span class="cc-target-style-prefix">target style</span><span class="cc-target-style-text">${escape(c.target_style)}</span></div>`
      : "";
    // Overlap hint: if this draft cluster shares values with an
    // earlier accepted card, name the conflict(s) so the operator
    // can see at a glance why the auto-seeded verdict is "reject"
    // and which card to flip if they prefer this interpretation.
    const overlapLine = (c.overlap_with && c.overlap_with.length)
      ? `<div class="cc-overlap-line" title="This draft shares values with an earlier accepted card; the save payload can only contain one of them."><span class="cc-overlap-prefix">overlaps with</span><span class="cc-overlap-text">${c.overlap_with.map(escape).join(", ")}</span></div>`
      : "";
    const isAcc = c.verdict === "accept";
    const isRej = c.verdict === "reject";
    return `
      <div class="cb-cluster-card${c.from_draft ? " from-draft" : ""}${isActive ? " active" : ""}${isAcc ? " verdict-accept" : ""}${isRej ? " verdict-reject" : ""}" data-ci="${ci}">
        <header class="cc-head">
          <span class="cb-cluster-id">${escape(c.cluster_id)}</span>
          ${draftBadge}
          <span class="cc-canonical-line">
            <span class="cc-canonical-prefix">canonical</span>
            <code>${escape(c.canonical || "(none)")}</code>
          </span>
          ${agreement}
          <span class="cc-kind-cell">kind ${kindSel}</span>
          <span class="cc-actions">
            <button type="button" class="cc-verdict-btn cc-accept${isAcc ? " active" : ""}" data-ci="${ci}" data-verdict="accept" title="Keep this cluster on save (shortcut: 1)">✓ Accept</button>
            <button type="button" class="cc-verdict-btn cc-reject${isRej ? " active" : ""}" data-ci="${ci}" data-verdict="reject" title="Reject — omit from saved payload (shortcut: 2)">✕ Reject</button>
            <button type="button" class="cb-cluster-remove" data-ci="${ci}" title="Delete this cluster card entirely">Delete</button>
          </span>
        </header>
        ${targetStyleLine}
        ${overlapLine}
        <div class="cb-cluster-values">${valueRows}</div>
        <div class="cb-cluster-meta">${c.values.length} values · ${c.from_draft ? "LLM proposal" : "operator-created"}${c.support_runs && c.support_runs.length ? ` · supported by runs ${c.support_runs.join(", ")}` : ""}</div>
      </div>
    `;
  }).join("");
  // Per-card/per-control handlers are installed once by
  // bindClusterListDelegation() during bootstrap; nothing to do here.
}

function bindClusterListDelegation() {
  const list = $("#cb-cluster-list");
  if (!list) return;

  // ``change`` is not in the delegated click handler -- dispatch it
  // separately because change events bubble but don't share the same
  // dispatch shape as clicks.
  list.addEventListener("change", (ev) => {
    const target = ev.target;
    if (!(target instanceof Element)) return;
    if (target.matches(".cb-cluster-kind")) {
      const ci = Number(target.dataset.ci);
      if (!Number.isFinite(ci) || !state.clusters[ci]) return;
      state.clusters[ci].kind = target.value;
      state.dirty = true;
      return;
    }
    if (target.matches(".v-canonical-radio")) {
      const ci = Number(target.dataset.ci);
      if (!Number.isFinite(ci) || !state.clusters[ci]) return;
      state.clusters[ci].canonical = target.dataset.value;
      state.dirty = true;
      rerenderClusters();
    }
  });

  list.addEventListener("click", (ev) => {
    const target = ev.target;
    if (!(target instanceof Element)) return;

    // Stop propagation for interactive sub-elements so a click on
    // them doesn't also fire the card-level "make this card active"
    // handler below.
    const verdictBtn = target.closest(".cc-verdict-btn");
    if (verdictBtn && list.contains(verdictBtn)) {
      ev.stopPropagation();
      const ci = Number(verdictBtn.dataset.ci);
      const c = state.clusters[ci];
      if (!c) return;
      const v = verdictBtn.dataset.verdict;
      // Toggle off if user clicks the same verdict (rare; mostly
      // they'll flip between accept/reject).
      c.verdict = c.verdict === v ? null : v;
      state.activeClusterIdx = ci;
      state.dirty = true;
      setSaveMsg("");
      rerenderClusters();
      return;
    }

    const removeBtn = target.closest(".cb-cluster-remove");
    if (removeBtn && list.contains(removeBtn)) {
      ev.stopPropagation();
      const ci = Number(removeBtn.dataset.ci);
      state.clusters.splice(ci, 1);
      state.dirty = true;
      rerenderCenter();
      return;
    }

    const valueRemoveBtn = target.closest(".v-remove");
    if (valueRemoveBtn && list.contains(valueRemoveBtn)) {
      ev.stopPropagation();
      const ci = Number(valueRemoveBtn.dataset.ci);
      const value = valueRemoveBtn.dataset.value;
      const c = state.clusters[ci];
      if (!c) return;
      c.values = c.values.filter((v) => v.value !== value);
      if (c.canonical === value) {
        c.canonical = c.values.length
          ? c.values.slice().sort((a, b) => b.count - a.count)[0].value
          : null;
      }
      if (c.values.length < 2) state.clusters.splice(ci, 1);
      state.dirty = true;
      rerenderCenter();
      return;
    }

    // The kind <select> and canonical radio shouldn't activate
    // the card just because the user clicked them.
    if (target.closest(".cb-cluster-kind") ||
        target.closest(".v-canonical-radio")) {
      ev.stopPropagation();
      return;
    }

    // Fall-through: card-level click activates the card.
    const card = target.closest(".cb-cluster-card");
    if (card && list.contains(card)) {
      const ci = Number(card.dataset.ci);
      if (Number.isFinite(ci)) {
        state.activeClusterIdx = ci;
        rerenderClusters();
      }
    }
  });
}

function bindQueueDelegation() {
  const box = $("#queue");
  if (!box) return;
  const open = (item) => {
    if (!item) return;
    openColumn(item.dataset.source, item.dataset.column);
  };
  box.addEventListener("click", (ev) => {
    const item = ev.target instanceof Element
      ? ev.target.closest(".queue-item") : null;
    if (item && box.contains(item)) open(item);
  });
  // Keyboard activation: the .queue-item has role="button"
  // tabindex="0" so Enter / Space should open it just like a click.
  box.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    const item = ev.target instanceof Element
      ? ev.target.closest(".queue-item") : null;
    if (item && box.contains(item)) {
      ev.preventDefault();
      open(item);
    }
  });
}

function bindValueListDelegation() {
  const list = $("#cb-value-list");
  if (!list) return;
  list.addEventListener("click", (ev) => {
    const row = ev.target instanceof Element
      ? ev.target.closest(".cb-value-row") : null;
    if (row && list.contains(row)) toggleSelect(row.dataset.value);
  });
}

// --------------------------------------------------------- save

async function onSave() {
  if (!state.active || state.saving) return;
  const status = state.status;
  // Only accepted (non-rejected) clusters are persisted. Rejected
  // ones stay in the UI so the operator can change their mind, but
  // they're not part of the on-disk record.
  const acceptedClusters = state.clusters.filter((c) => c.verdict !== "reject");
  // Client-side mirror of the server invariants so we surface
  // mistakes immediately. The server is still authoritative.
  if (status === "labeled" && !acceptedClusters.length) {
    setSaveMsg(
      state.clusters.length
        ? "All clusters are rejected. Accept at least one, or switch status to no_inconsistency."
        : "Status=labeled needs at least one cluster.",
      "error",
    );
    return;
  }
  if (status !== "labeled" && acceptedClusters.length) {
    setSaveMsg(`Status=${status} must have an empty cluster list. Reject the remaining clusters or switch back to labeled.`, "error");
    return;
  }
  if (status.startsWith("dropped_") && !state.drop_reason.trim()) {
    setSaveMsg(`drop_reason is required for status=${status}.`, "error");
    return;
  }
  const payload = {
    source: state.active.source,
    column: state.active.column,
    status,
    drop_reason: status.startsWith("dropped_") ? state.drop_reason.trim() : null,
    clusters: acceptedClusters.map((c) => ({
      cluster_id: c.cluster_id,
      kind: c.kind,
      canonical_value: c.canonical,
      values: c.values.map((v) => ({ value: v.value, count: v.count })),
    })),
    note: state.note || null,
  };
  state.saving = true;
  $("#cb-save").disabled = true;
  setSaveMsg("saving…");
  try {
    const sEnc = encodeURIComponent(state.active.source);
    const ru = encodeURIComponent(RUN_ID);
    // Same path-converter trick as openColumn: action before column,
    // ``encodeURI`` (not ``encodeURIComponent``) on the column so a
    // literal ``/`` in the name survives to the router.
    await api(
      `/api/runs/${ru}/columns/${sEnc}/clusters/${encodeURI(state.active.column)}`,
      { method: "POST", body: JSON.stringify(payload) },
    );
    state.dirty = false;
    setSaveMsg("saved.", "ok");
    // Refresh both the queue (status badge may have flipped) and
    // the current column (canonical record is now the server's).
    await refreshAll();
    await openColumn(state.active.source, state.active.column);
  } catch (e) {
    setSaveMsg(`save failed: ${e.message}`, "error");
  } finally {
    state.saving = false;
    $("#cb-save").disabled = false;
  }
}

function setSaveMsg(text, cls = "") {
  const el = $("#cb-save-msg");
  if (!el) return;
  el.textContent = text;
  el.className = `muted ${cls}`.trim();
}

// --------------------------------------------------------- go

bootstrap().catch((e) => {
  $("#run-meta").textContent = `bootstrap error: ${e.message}`;
});
