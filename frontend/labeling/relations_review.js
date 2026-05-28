// Relation-review UI. Lean front-end that lists discovered relations
// from a discovery run, lets the operator approve/reject/clear each,
// and POSTs verdicts to ``/api/runs/{run_id}/relation-reviews``.
//
// The detection runner (``experiments/run_relation_detection``) reads
// the same ``relation_reviews.jsonl`` to decide which relations to
// compile into per-cell violations, so the review verdicts here are
// the bridge between Step 1 (discovery) and Step 2 (detection).

const params = new URLSearchParams(location.search);
const RUN_ID = params.get("run");
if (!RUN_ID) {
  document.getElementById("run-meta").textContent = "no run id in URL";
  throw new Error("missing run id");
}
const API = `/api/runs/${encodeURIComponent(RUN_ID)}`;

const state = {
  layer: "auto",
  error_type: "",
  verdict: "",
  search: "",
  items: [],     // [{relation, review, verdict, verification}, ...]
  counts: {},
  selected: null,
  errorTypes: new Set(),
};

const $ = (sel) => document.querySelector(sel);
const escapeHTML = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let body = "";
    try { body = await r.text(); } catch (_) {}
    throw new Error(`${r.status} ${r.statusText}: ${body.slice(0, 200)}`);
  }
  return r.json();
}

async function loadRun() {
  try {
    const info = await fetchJSON(`/api/runs?include=${encodeURIComponent(RUN_ID)}`)
      .catch(() => null);
    let meta = `run: ${RUN_ID}`;
    if (info && Array.isArray(info)) {
      const found = info.find((r) => r.run_id === RUN_ID);
      if (found) {
        const bits = [found.model, found.mode, `${found.n_relations ?? 0} relations`].filter(Boolean);
        meta = `${RUN_ID} · ${bits.join(" · ")}`;
      }
    }
    $("#run-meta").textContent = meta;
  } catch (exc) {
    $("#run-meta").textContent = `run: ${RUN_ID}`;
  }
}

async function loadRelations() {
  const qs = new URLSearchParams();
  if (state.layer && state.layer !== "auto") qs.set("layer", state.layer);
  if (state.error_type) qs.set("error_type", state.error_type);
  if (state.verdict) qs.set("verdict", state.verdict);
  $("#queue").innerHTML = `<div class="empty-note">loading…</div>`;
  try {
    const data = await fetchJSON(`${API}/relations?${qs.toString()}`);
    state.items = data.relations || [];
    state.counts = data.counts || {};
    // Populate error-type filter from the union of all items, the first time.
    if (state.errorTypes.size === 0) {
      for (const it of state.items) {
        const et = it.relation?.error_type;
        if (et) state.errorTypes.add(et);
      }
      const sel = $("#filter-error-type");
      [...state.errorTypes].sort().forEach((et) => {
        const o = document.createElement("option");
        o.value = et; o.textContent = et;
        sel.appendChild(o);
      });
    }
    renderQueue();
    renderCounts();
    if (state.items.length && !state.selected) {
      selectItem(0);
    } else if (state.selected) {
      // Try to reselect by species_key.
      const idx = state.items.findIndex(
        (it) => (it.relation?.species_key ?? "") === state.selected,
      );
      if (idx >= 0) selectItem(idx);
      else $("#detail").innerHTML = `<div class="empty-note">Selected relation no longer matches filters.</div>`;
    }
  } catch (exc) {
    $("#queue").innerHTML = `<div class="empty-note">load failed: ${escapeHTML(exc.message)}</div>`;
  }
}

function renderCounts() {
  const total = state.items.length;
  const parts = [`${total} shown`];
  for (const k of ["approved", "rejected", "added", "cleared", "unreviewed"]) {
    if (state.counts[k]) parts.push(`${k}: ${state.counts[k]}`);
  }
  $("#counts-bar").textContent = parts.join(" · ");
}

function passRateClass(pr) {
  if (pr == null) return "";
  if (pr >= 0.95) return "pass-rate-good";
  if (pr >= 0.5) return "pass-rate-mid";
  return "pass-rate-bad";
}

function renderQueue() {
  const search = state.search.trim().toLowerCase();
  const list = state.items.filter((it) => {
    if (!search) return true;
    const r = it.relation || {};
    const hay = [
      r.predicate, r.kind, ...(r.columns || []), r.error_type, r.rationale,
    ].filter(Boolean).join(" ").toLowerCase();
    return hay.includes(search);
  });
  if (!list.length) {
    $("#queue").innerHTML = `<div class="empty-note">No relations match the filters.</div>`;
    return;
  }
  const html = list.map((it, i) => {
    const r = it.relation || {};
    const v = it.verdict || "unreviewed";
    const cols = (r.columns || []).join(", ");
    const isSel = state.selected && state.selected === (r.species_key ?? "");
    return `
      <div class="qitem ${isSel ? "active" : ""}" data-idx="${i}">
        <div class="kind">${escapeHTML(r.kind || "?")} · ${escapeHTML(r.error_type || "?")}</div>
        <div class="columns">${escapeHTML(cols)}</div>
        <div class="pred">${escapeHTML((r.predicate || "").slice(0, 140))}</div>
        ${r.executable
          ? `<div class="pred" style="font-family:monospace; opacity:0.85;">${escapeHTML(String(r.executable).slice(0, 140))}</div>`
          : ""}
        <div class="meta"><span class="verdict-pill ${v}">${escapeHTML(v)}</span>${
          it.verification?.pass_rate != null
            ? `<span class="${passRateClass(it.verification.pass_rate)}">pr=${it.verification.pass_rate.toFixed(2)}</span>`
            : ""
        }</div>
      </div>`;
  }).join("");
  $("#queue").innerHTML = html;
  $("#queue").querySelectorAll(".qitem").forEach((el) => {
    el.addEventListener("click", () => {
      // The data-idx is the filtered list's index; resolve back to the
      // unfiltered ``items`` index via species_key.
      const filteredIdx = Number(el.dataset.idx);
      const item = list[filteredIdx];
      const sk = item.relation?.species_key ?? "";
      const realIdx = state.items.findIndex(
        (it) => (it.relation?.species_key ?? "") === sk,
      );
      if (realIdx >= 0) selectItem(realIdx);
    });
  });
}

function selectItem(idx) {
  const it = state.items[idx];
  if (!it) return;
  state.selected = it.relation?.species_key ?? "";
  renderQueue();
  renderDetail(it);
}

function renderDetail(it) {
  const r = it.relation || {};
  const v = it.verification || null;
  const verdict = it.verdict || "unreviewed";
  const sampleViolators = v?.sample_violators;
  const verLine = v
    ? `verifier_kind=${v.verifier_kind ?? "?"} · pass_rate=${
        v.pass_rate != null ? `<span class="${passRateClass(v.pass_rate)}">${v.pass_rate.toFixed(3)}</span>` : "(n/a)"
      } · verifiable=${v.verifiable === true ? "yes" : v.verifiable === false ? "no" : "(unknown)"}`
    : `<span class="muted">no verification record (run validation first)</span>`;
  $("#detail").innerHTML = `
    <div class="detail-card">
      <h2>
        ${escapeHTML(r.kind || "?")} · ${escapeHTML(r.error_type || "?")}
        <span class="verdict-pill ${verdict}" style="margin-left:8px;">${escapeHTML(verdict)}</span>
      </h2>
      <div class="row"><span class="label">Source</span> ${escapeHTML(r.source || "")}</div>
      <div class="row"><span class="label">Columns</span> ${(r.columns || []).map(escapeHTML).join(", ")}</div>
      ${r.lhs && r.lhs.length ? `<div class="row"><span class="label">LHS → RHS</span> ${
        (r.lhs || []).map(escapeHTML).join(", ")} &nbsp;<b>→</b>&nbsp; ${
        (r.rhs || []).map(escapeHTML).join(", ")}</div>` : ""}
      <div class="row"><span class="label">Predicate</span></div>
      <pre class="predicate">${escapeHTML(r.predicate || "")}</pre>
      ${r.rationale ? `<div class="row"><span class="label">Rationale</span></div>
        <pre class="predicate" style="opacity:0.85;">${escapeHTML(r.rationale)}</pre>` : ""}
      ${r.executable ? `<div class="row"><span class="label">Formula</span></div>
        <pre class="predicate" style="font-family:monospace;">${escapeHTML(r.executable)}</pre>` : ""}
      <div class="row verification">${verLine}</div>
      ${v?.refusal_reason
        ? `<div class="row verification"><span class="label">Refusal</span> <span class="muted">${escapeHTML(v.refusal_reason)}</span></div>`
        : ""}
      ${Array.isArray(sampleViolators) && sampleViolators.length
        ? `<div class="row sample-rows"><span class="label">Sample violators</span>
            ${sampleViolators.slice(0,5).map((row) => `<pre>${escapeHTML(JSON.stringify(row))}</pre>`).join("")}
          </div>`
        : ""}
      <div class="row"><span class="label">species_key</span> <code style="font-size:0.78em;">${escapeHTML(r.species_key || "")}</code></div>
      <div class="actions-row">
        <button class="approve ${verdict === "approved" ? "active" : ""}" data-decision="approved">Approve</button>
        <button class="reject ${verdict === "rejected" ? "active" : ""}" data-decision="rejected">Reject</button>
        <button class="clear" data-decision="cleared">Clear</button>
      </div>
      <div id="decision-note" style="margin-top:8px; color:#9aa3b8; font-size:0.85em;"></div>
    </div>`;
  $("#detail").querySelectorAll(".actions-row button").forEach((b) => {
    b.addEventListener("click", () => submitDecision(it, b.dataset.decision));
  });
}

async function submitDecision(item, decision) {
  const r = item.relation || {};
  const sk = r.species_key;
  if (!sk) {
    $("#decision-note").textContent = "missing species_key; cannot submit.";
    return;
  }
  const payload = {
    relation_id: sk,
    decision,
    relation: {
      kind: r.kind || "",
      columns: r.columns || [],
      lhs: r.lhs || [],
      rhs: r.rhs || [],
      predicate: r.predicate || "",
      executable: r.executable || null,
      rationale: r.rationale || "",
      source: r.source || null,
      error_type: r.error_type || null,
    },
  };
  $("#decision-note").textContent = `submitting ${decision}…`;
  try {
    await fetchJSON(`${API}/relation-reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    item.verdict = decision;
    item.review = { ...(item.review || {}), decision };
    $("#decision-note").textContent = `recorded ${decision}.`;
    renderDetail(item);
    renderQueue();
    // Re-fetch counts (cheap; we'd otherwise need to recompute locally).
    refreshCounts();
  } catch (exc) {
    $("#decision-note").textContent = `failed: ${exc.message}`;
  }
}

async function refreshCounts() {
  // Cheap recompute from local state — avoids a network round-trip.
  const counts = {};
  for (const it of state.items) {
    const v = it.verdict || "unreviewed";
    counts[v] = (counts[v] || 0) + 1;
  }
  state.counts = counts;
  renderCounts();
}

function wireFilters() {
  $("#filter-layer").addEventListener("change", (e) => {
    state.layer = e.target.value;
    state.errorTypes.clear();  // rebuild from new payload
    $("#filter-error-type").innerHTML = `<option value="">(all)</option>`;
    loadRelations();
  });
  $("#filter-error-type").addEventListener("change", (e) => {
    state.error_type = e.target.value;
    loadRelations();
  });
  $("#filter-verdict").addEventListener("change", (e) => {
    state.verdict = e.target.value;
    loadRelations();
  });
  let searchTimer = null;
  $("#filter-search").addEventListener("input", (e) => {
    state.search = e.target.value;
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(renderQueue, 120);
  });
}

(async function bootstrap() {
  wireFilters();
  await loadRun();
  await loadRelations();
})();
