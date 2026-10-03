// Churn Risk Checker – talks to the FastAPI backend.
// The backend normally serves this page itself (same origin). If the page is opened another way
// (double-clicked file, VS Code Live Server, …) it falls back to the local backend; ?api=<url> overrides both.
const DEFAULT_BACKEND = "http://127.0.0.1:8000";
const BACKEND_CANDIDATES = [
  new URLSearchParams(location.search).get("api"),
  location.protocol.startsWith("http") ? "" : null,
  DEFAULT_BACKEND,
].filter((b) => b !== null);
let API = "";
let FIELDS = [];
let THRESHOLD = 0.51;
let EXAMPLES = {};
let batchResults = [];

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, text) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const riskClass = (level) => level.toLowerCase();

// ------------------------------------------------------------------ start-up
async function init() {
  setupTabs();
  try {
    const health = await findBackend();
    const schema = await getJSON("/api/schema");
    FIELDS = schema.fields;
    THRESHOLD = schema.decision_threshold;
    buildForm();
    showModelInfo(health.model);
    $("#service-status").textContent = "Service online";
    $("#service-status").className = "status status-ok";
  } catch (e) {
    $("#service-status").textContent = "Service offline";
    $("#service-status").className = "status status-down";
    $("#service-error").hidden = false;
    return;
  }
  EXAMPLES = await getJSON("/examples.json").catch(() => ({}));   // served by the backend with the page
  document.querySelectorAll("[data-example]").forEach((b) =>
    b.addEventListener("click", () => loadExample(b.dataset.example)));
  $("#customer-form").addEventListener("submit", onPredict);
  $("#clear-btn").addEventListener("click", clearForm);
  setupBatch();
}

// Use the first backend address that answers the health check
async function findBackend() {
  for (const base of BACKEND_CANDIDATES) {
    try {
      const r = await fetch(base + "/api/health");
      if (r.ok) {
        API = base;
        return await r.json();
      }
    } catch (e) { /* not reachable – try the next address */ }
  }
  throw new Error("backend not reachable");
}

async function getJSON(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    $("#tab-single").hidden = tab.dataset.tab !== "single";
    $("#tab-batch").hidden = tab.dataset.tab !== "batch";
  }));
}

function showModelInfo(m) {
  const t = m.test_metrics;
  $("#model-info").textContent =
    `Model: tuned ${m.model} (decision threshold ${m.decision_threshold}). ` +
    `Held-out test performance – F1 ${t.F1}, ROC-AUC ${t["ROC-AUC"]}, recall ${t.Recall}, precision ${t.Precision}. ` +
    `Predictions support, not replace, the retention team's judgement.`;
}

// ------------------------------------------------------------------ form
function buildForm() {
  const container = $("#form-groups");
  const groups = [...new Set(FIELDS.map((f) => f.group))];
  for (const g of groups) {
    const fs = el("fieldset");
    fs.appendChild(el("legend", {}, g));
    const grid = el("div", { class: "fields" });
    for (const f of FIELDS.filter((x) => x.group === g)) grid.appendChild(buildField(f));
    fs.appendChild(grid);
    container.appendChild(fs);
  }
  // fields that depend on another answer (no phone → no multiple lines, etc.)
  for (const ctrl of new Set(FIELDS.filter((f) => f.depends_on).map((f) => f.depends_on.field))) {
    $(`#f-${ctrl}`).addEventListener("change", applyDependencies);
  }
  applyDependencies();
}

function buildField(f) {
  const wrap = el("div", { class: "field", id: `w-${f.name}` });
  const label = el("label", { for: `f-${f.name}` }, f.label);
  if (f.required) label.appendChild(el("span", { class: "req", "aria-hidden": "true" }, " *"));
  wrap.appendChild(label);
  let input;
  if (f.type === "select") {
    input = el("select", { id: `f-${f.name}`, name: f.name });
    input.appendChild(el("option", { value: "" }, "Select…"));
    for (const o of f.options) input.appendChild(el("option", { value: o }, o));
  } else {
    input = el("input", { id: `f-${f.name}`, name: f.name, type: "number", min: f.min, max: f.max, step: f.step,
                          inputmode: "decimal", placeholder: f.required ? "" : "optional" });
  }
  input.addEventListener("input", () => clearFieldError(f.name));
  wrap.appendChild(input);
  if (f.help) wrap.appendChild(el("span", { class: "help" }, f.help));
  wrap.appendChild(el("span", { class: "error", id: `e-${f.name}` }));
  return wrap;
}

function applyDependencies() {
  for (const f of FIELDS.filter((x) => x.depends_on)) {
    const d = f.depends_on;
    const input = $(`#f-${f.name}`);
    const disabled = $(`#f-${d.field}`).value === d.disabled_when;
    if (disabled) {
      input.value = d.value_when_disabled;
      clearFieldError(f.name);
    } else if (input.disabled) {
      input.value = "";            // re-enabled: ask the user for a real answer
    }
    input.disabled = disabled;
  }
}

function clearForm() {
  $("#customer-form").reset();
  FIELDS.forEach((f) => clearFieldError(f.name));
  $("#form-error").hidden = true;
  applyDependencies();
}

function loadExample(name) {
  const ex = EXAMPLES[name];
  if (!ex) return;
  clearForm();
  // set the controlling fields first so dependent fields are enabled correctly
  for (const key of ["phone_service", "internet_type"]) $(`#f-${key}`).value = ex[key];
  applyDependencies();
  for (const f of FIELDS) {
    const input = $(`#f-${f.name}`);
    if (!input.disabled && ex[f.name] !== undefined && ex[f.name] !== null) input.value = ex[f.name];
  }
}

function clearFieldError(name) {
  const w = $(`#w-${name}`);
  if (!w) return;
  w.classList.remove("invalid");
  $(`#e-${name}`).textContent = "";
}

function setFieldError(name, message) {
  const w = $(`#w-${name}`);
  if (!w) return false;
  w.classList.add("invalid");
  $(`#e-${name}`).textContent = message;
  return true;
}

// Client-side validation mirrors the API rules, so most mistakes are caught before sending
function readForm() {
  const data = {};
  const errors = [];
  for (const f of FIELDS) {
    const input = $(`#f-${f.name}`);
    const raw = input.value.trim();
    if (input.disabled) { data[f.name] = f.type === "number" ? Number(raw) : raw; continue; }
    if (raw === "") {
      if (f.required) errors.push([f.name, "This field is required."]);
      else data[f.name] = null;
      continue;
    }
    if (f.type === "number") {
      const v = Number(raw);
      if (!Number.isFinite(v)) { errors.push([f.name, "Enter a number."]); continue; }
      if (v < f.min || v > f.max) { errors.push([f.name, `Enter a value between ${f.min} and ${f.max}.`]); continue; }
      if (f.step === 1 && !Number.isInteger(v)) { errors.push([f.name, "Enter a whole number."]); continue; }
      data[f.name] = v;
    } else {
      data[f.name] = raw;
    }
  }
  return { data, errors };
}

async function onPredict(event) {
  event.preventDefault();
  FIELDS.forEach((f) => clearFieldError(f.name));
  $("#form-error").hidden = true;
  const { data, errors } = readForm();
  if (errors.length) {
    errors.forEach(([n, m]) => setFieldError(n, m));
    showFormError(`Please fix ${errors.length} field${errors.length > 1 ? "s" : ""} highlighted below.`);
    $(`#f-${errors[0][0]}`).focus();
    return;
  }
  const btn = $("#predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const r = await fetch(API + "/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const body = await r.json();
    if (r.status === 422) {
      const unmatched = body.errors.filter((e) => !setFieldError(e.field, e.message));
      showFormError("The service rejected some inputs: " +
        (unmatched.length ? unmatched.map((e) => `${e.field}: ${e.message}`).join("; ") : "see the highlighted fields."));
      return;
    }
    if (!r.ok) throw new Error(body.detail || r.statusText);
    renderResult(body);
    if (window.innerWidth < 900) $("#result").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    showFormError(`Prediction failed: ${e.message}. Is the backend running?`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict churn risk";
  }
}

function showFormError(msg) {
  const b = $("#form-error");
  b.textContent = msg;
  b.hidden = false;
}

// ------------------------------------------------------------------ result
function renderResult(r) {
  const box = $("#result");
  box.replaceChildren();
  box.appendChild(el("h2", {}, "Prediction"));
  const cls = riskClass(r.risk_level);
  box.appendChild(el("div", { class: `verdict ${cls}` },
    r.at_risk ? "Likely to churn" : (r.risk_level === "Moderate" ? "Likely to stay – keep an eye on" : "Likely to stay")));
  box.appendChild(el("div", { class: "prob" }, r.churn_probability_pct));
  box.appendChild(el("div", { class: "prob-label" }, `chance of churning · ${r.risk_level} risk`));

  const gauge = el("div", { class: "gauge", role: "img",
    "aria-label": `Churn probability ${r.churn_probability_pct}; customers at or above ${pct(r.threshold)} are flagged` });
  const fill = el("div", { class: "gauge-fill" });
  fill.style.width = `${r.churn_probability * 100}%`;
  fill.style.background = `var(--${r.at_risk ? "risk" : cls === "moderate" ? "moderate" : "safe"})`;
  const th = el("div", { class: "gauge-threshold", title: `Decision threshold ${pct(r.threshold)}` });
  th.style.left = `${r.threshold * 100}%`;
  gauge.append(fill, th);
  box.appendChild(gauge);
  const scale = el("div", { class: "gauge-scale" });
  scale.append(el("span", {}, "0%"), el("span", {}, `flagged from ${pct(r.threshold)}`), el("span", {}, "100%"));
  box.appendChild(scale);

  const maxImpact = Math.max(0.01, ...r.factors_increasing_risk.map((f) => f.impact),
                             ...r.factors_decreasing_risk.map((f) => -f.impact));
  const factorList = (title, factors, dir) => {
    if (!factors.length) return;
    box.appendChild(el("h3", {}, title));
    for (const f of factors) {
      const row = el("div", { class: "factor" });
      const name = el("div", { class: "name" }, f.feature);
      name.appendChild(el("small", {}, f.value));
      const bar = el("div", { class: `bar ${dir}` });
      bar.style.width = `${Math.max(6, (Math.abs(f.impact) / maxImpact) * 100)}%`;
      row.append(name, bar);
      box.appendChild(row);
    }
  };
  factorList("What raises the risk", r.factors_increasing_risk, "up");
  factorList("What lowers the risk", r.factors_decreasing_risk, "down");

  box.appendChild(el("h3", {}, "Suggested actions"));
  const ul = el("ul", { class: "actions-list" });
  r.recommendations.forEach((a) => ul.appendChild(el("li", {}, a)));
  box.appendChild(ul);

  if (r.warnings.length) {
    const w = el("ul", { class: "warn-list" });
    r.warnings.forEach((t) => w.appendChild(el("li", {}, t)));
    box.appendChild(w);
  }
}

// ------------------------------------------------------------------ batch (CSV)
function setupBatch() {
  const file = $("#csv-file");
  file.addEventListener("change", () => { $("#csv-btn").disabled = !file.files.length; });
  $("#csv-btn").addEventListener("click", onScoreCsv);
  $("#only-risk").addEventListener("change", renderBatchTable);
  $("#download-btn").addEventListener("click", downloadResults);
}

async function onScoreCsv() {
  const f = $("#csv-file").files[0];
  const err = $("#batch-error");
  err.hidden = true;
  if (!f) return;
  if (!f.name.toLowerCase().endsWith(".csv")) { err.textContent = "Please choose a .csv file."; err.hidden = false; return; }
  const btn = $("#csv-btn");
  btn.disabled = true;
  btn.textContent = "Scoring…";
  try {
    const form = new FormData();
    form.append("file", f);
    const r = await fetch(API + "/api/predict/csv", { method: "POST", body: form });
    const body = await r.json();
    if (!r.ok) {
      const msg = body.errors ? body.errors.map((e) => e.message).join(" ") : (body.detail || r.statusText);
      throw new Error(msg);
    }
    batchResults = body.results.sort((a, b) => b.churn_probability - a.churn_probability);
    $("#sum-count").textContent = body.count;
    $("#sum-risk").textContent = body.at_risk_count;
    $("#sum-rate").textContent = body.count ? pct(body.at_risk_count / body.count) : "–";
    $("#sum-errors").textContent = body.errors.length;
    renderBatchTable();
    const list = $("#row-error-list");
    list.replaceChildren();
    for (const e of body.errors) {
      const who = e.customer_id ? `Row ${e.row} (${e.customer_id})` : `Row ${e.row}`;
      list.appendChild(el("li", {}, `${who}: ` + e.errors.map((x) => `${x.field} – ${x.message}`).join("; ")));
    }
    $("#row-errors").hidden = body.errors.length === 0;
    $("#batch-result").hidden = false;
  } catch (e) {
    err.textContent = `Could not score the file: ${e.message}`;
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Score file";
  }
}

function renderBatchTable() {
  const tbody = $("#batch-table tbody");
  tbody.replaceChildren();
  const rows = $("#only-risk").checked ? batchResults.filter((r) => r.at_risk) : batchResults;
  rows.forEach((r, i) => {
    const tr = el("tr");
    tr.appendChild(el("td", {}, String(i + 1)));
    tr.appendChild(el("td", {}, r.customer_id || `Row ${r.index}`));
    const prob = el("td", { class: "num" }, r.churn_probability_pct);
    const mini = el("span", { class: "mini" });
    const bar = el("i");
    bar.style.width = `${r.churn_probability * 100}%`;
    bar.style.background = `var(--${r.at_risk ? "risk" : r.risk_level === "Moderate" ? "moderate" : "safe"})`;
    mini.appendChild(bar);
    prob.appendChild(mini);
    tr.appendChild(prob);
    const lvl = el("td");
    lvl.appendChild(el("span", { class: `pill ${riskClass(r.risk_level)}` }, r.risk_level));
    tr.appendChild(lvl);
    const top = r.factors_increasing_risk[0];
    tr.appendChild(el("td", {}, top ? `${top.feature}: ${top.value}` : "–"));
    tr.appendChild(el("td", {}, r.at_risk ? r.recommendations[0] : "No urgent action"));
    tbody.appendChild(tr);
  });
}

function downloadResults() {
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const header = ["rank", "customer_id", "churn_probability", "prediction", "risk_level", "top_risk_factor", "suggested_action"];
  const lines = [header.join(",")];
  batchResults.forEach((r, i) => {
    const top = r.factors_increasing_risk[0];
    lines.push([i + 1, r.customer_id || `row ${r.index}`, r.churn_probability, r.prediction, r.risk_level,
      top ? `${top.feature}: ${top.value}` : "", r.recommendations[0] || ""].map(esc).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "churn_predictions.csv" });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

init();
