// Runs dashboard + launcher + jobs panel.
// One SPA, three tabs. State is held here; server is the source of truth.

// Override styles.css's `html, body { overflow: hidden; height: 100vh; }`
// which is correct for the labelling view but kills scrolling here.
document.documentElement.style.height = "auto";
document.documentElement.style.overflow = "auto";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

function toast(msg, ms = 1800) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

function escape(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

// --------------------------------------------------------- tabs

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== name));
  localStorage.setItem("dq-ws-tab", name);
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

  initDataPrep();
  initRelationsForm();

  // Poll jobs list every 3s (cheap - one small JSON).
  setInterval(refreshJobs, 3000);
}

// --------------------------------------------------------- runs

async function refreshWorkspace() {
  const ws = await api("/api/runs");
  state.workspace = ws;
  $("#ws-meta").textContent = `${ws.project_root} · annotator ${ws.annotator} · ${ws.runs.length} runs`;
  renderRuns(ws.runs);
}

function renderRuns(runs) {
  $("#runs-count").textContent = `(${runs.length})`;
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
  box.innerHTML = runs.map((r) => {
    if (r.kind === "relations") {
      return renderRelationsRunRow(r);
    }
    const labelLink = r.has_clusters
      ? `<a class="primary" href="./label.html?run=${encodeURIComponent(r.run_id)}">Label</a>`
      : `<span class="muted">(no clusters)</span>`;
    const reportLink = r.has_report
      ? `<a href="/api/runs/${encodeURIComponent(r.run_id)}/report" target="_blank" rel="noopener">Report</a>`
      : `<button data-action="build-report" data-run="${escape(r.run_id)}">Build report</button>`;
    const exportLink = r.has_clusters
      ? `<a href="/api/runs/${encodeURIComponent(r.run_id)}/export/benchmark" download>Benchmark</a>`
      : "";
    const created = r.created ? escape(r.created.slice(0, 19).replace("T", " ")) + " UTC" : "";
    // Build tag spans through one helper so future contributors adding
    // a new tag can't accidentally forget to escape the value.
    const tagSpan = (cls, value) => value
      ? `<span class="tag ${cls}">${escape(value)}</span>`
      : "";
    const tags = [
      tagSpan("model", r.model),
      tagSpan("mode", r.mode),
    ].join("");
    return `
      <div class="row">
        <div>
          <div class="id">${escape(r.run_id)}</div>
          <div class="sub">${tags}${created ? `<span>${created}</span>` : ""}</div>
        </div>
        <div class="num"><span class="label">clusters</span>${escape(String(r.clusters ?? 0))}</div>
        <div class="num"><span class="label">issues</span>${escape(String(r.issues ?? 0))}</div>
        <div class="num"><span class="label">labelled</span>${r.has_labels ? "yes" : "–"}</div>
        <div class="actions">
          ${labelLink}
          ${reportLink}
          ${exportLink}
          <button class="danger" data-action="delete-run" data-run="${escape(r.run_id)}" title="Delete this run">Delete</button>
        </div>
      </div>
    `;
  }).join("");

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

  box.querySelectorAll('[data-action="delete-run"]').forEach((b) => {
    b.addEventListener("click", async () => {
      const runId = b.dataset.run;
      if (!confirm(`Delete run "${runId}"?\n\nThis will permanently remove its outputs/${runId}/ directory (clusters, summary, report, labels). This cannot be undone.`)) {
        return;
      }
      b.disabled = true;
      b.textContent = "Deleting…";
      try {
        await api(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
        toast(`deleted run ${runId}`);
        refreshWorkspace();
      } catch (e) {
        toast(`delete failed: ${e.message}`, 4000);
        b.disabled = false;
        b.textContent = "Delete";
      }
    });
  });
}

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
  // Checkbox meaning is inverted: `no-synthesis` is ON when
  // enable_synthesis is FALSE.
  setChecked("no-synthesis", llm.enable_synthesis === false);

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
  });
  return label;
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
    no_synthesis: check("no-synthesis") || null,
    synthesis_effort: val("synthesis-effort"),
    normalize_input: check("normalize-input") || null,
    normalize_fold_punctuation: check("normalize-fold-punctuation") || null,
    normalize_lowercase: check("normalize-lowercase") || null,
    error_types: errorTypes,
    mode: val("mode"),
  };

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
    synthesis_effort: "--synthesis-effort",
    gt_mode: "--gt-mode", gt_num_runs: "--gt-num-runs",
    gt_min_runs: "--gt-min-runs", gt_target_missing_mass: "--gt-target-missing-mass",
    gt_stop_scope: "--gt-stop-scope",
    gt_base_seed: "--gt-base-seed", gt_variant_effort: "--gt-variant-effort",
    gt_per_arm_stop_when: "--gt-per-arm-stop-when",
    gt_per_arm_epsilon_chao1: "--gt-per-arm-epsilon-chao1",
  };
  const boolMap = {
    no_synthesis: ["--no-synthesis", null],
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
  const linkHTML = job.run_id
    ? ` · <a href="./label.html?run=${encodeURIComponent(job.run_id)}">Label</a> · <a href="/api/runs/${encodeURIComponent(job.run_id)}/report" target="_blank" rel="noopener">Report</a>`
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
// Schema discovery (multi-column relations)
// =============================================================

function renderRelationsRunRow(r) {
  const created = r.created
    ? escape(r.created.slice(0, 19).replace("T", " ")) + " UTC"
    : "";
  const reportLink = r.has_report
    ? `<a class="primary" href="/api/runs/${encodeURIComponent(r.run_id)}/report" target="_blank" rel="noopener">View relations</a>`
    : `<button data-action="build-relation-report" data-run="${escape(r.run_id)}">Build report</button>`;
  const tag = (cls, value, title) => value
    ? `<span class="tag ${cls}"${title ? ` title="${escape(title)}"` : ""}>${escape(String(value))}</span>`
    : "";
  const tags = [
    `<span class="tag" style="background:rgba(167,139,250,0.14);color:#c4b5fd;border-color:rgba(167,139,250,0.45);">relations</span>`,
    tag("model", r.model),
    r.mode ? tag("mode", r.mode) : "",
    r.n_passes != null
      ? tag("", `passes ${r.n_passes}`, "adaptive passes")
      : "",
    r.n_arms != null && r.n_saturated_arms != null
      ? tag("", `${r.n_saturated_arms}/${r.n_arms} sat`, "saturated arms / total arms")
      : "",
  ].filter(Boolean).join("");
  // Third stat slot: passes count when adaptive, otherwise show the
  // adaptive flag is off so the row stays the same width as detection rows.
  const thirdStat = r.n_passes != null
    ? `<div class="num"><span class="label">passes</span>${escape(String(r.n_passes))}</div>`
    : `<div class="num"><span class="label">mode</span>single</div>`;
  return `
    <div class="row">
      <div>
        <div class="id">${escape(r.run_id)}</div>
        <div class="sub">${tags}${created ? `<span>${created}</span>` : ""}</div>
      </div>
      <div class="num"><span class="label">relations</span>${escape(String(r.n_relations ?? 0))}</div>
      <div class="num"><span class="label">sources</span>${escape(String(r.n_sources ?? 0))}</div>
      ${thirdStat}
      <div class="actions">
        ${reportLink}
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
          chip.title = et.key + (et.name && et.name !== et.key ? ` \u00b7 ${et.name}` : "");
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
  const adaptiveFields = ["rel-max-passes", "rel-min-passes", "rel-gt-threshold", "rel-base-seed"];
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
    base_seed: adaptive ? num("rel-base-seed") : null,
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
  };
  const flags = [];
  for (const [k, v] of Object.entries(p)) {
    if (k === "adaptive" && v) flags.push("--adaptive");
    else if (k === "include_raw" && v) flags.push("--include-raw");
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
    toast(`schema discovery launched: ${job.job_id}`);
    showTab("jobs");
    openJob(job.job_id);
    refreshJobs();
  } catch (err) {
    toast(`launch failed: ${err.message}`, 4000);
  }
}
