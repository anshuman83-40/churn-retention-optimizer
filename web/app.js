/* ChurnOpt dashboard: plain JS + Chart.js, talks to the FastAPI backend. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const usd = (v) => {
  const a = Math.abs(v), sign = v < 0 ? "-" : "";
  return sign + (a >= 1e6 ? `$${(a / 1e6).toFixed(2)}M` : a >= 1e4 ? `$${(a / 1e3).toFixed(1)}k` : `$${Math.round(a).toLocaleString()}`);
};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};

const SEG = {
  "Persuadable": { color: "#4f8cff", key: "persuadable" },
  "Lost Cause": { color: "#f25f5c", key: "lost_cause" },
  "Sleeping Dog": { color: "#8b5cf6", key: "sleeping_dog" },
  "Loyal": { color: "#22c55e", key: "loyal" },
  "Monitor": { color: "#f5a524", key: "monitor" },
};
const FILTERS = [
  ["all", "All"], ["high", "High Risk"], ["persuadable", "Persuadable"],
  ["lost_cause", "Lost Cause"], ["sleeping_dog", "Sleeping Dog"], ["loyal", "Loyal"],
];
const riskColor = (p) => (p >= 0.6 ? "#f25f5c" : p >= 0.3 ? "#f5a524" : p >= 0.15 ? "#4f8cff" : "#22c55e");
const tag = (seg) => {
  const c = SEG[seg]?.color || "#8b97b3";
  return `<span class="tag" style="color:${c};border-color:${c}55;background:${c}1a">${esc(seg)}</span>`;
};

Chart.defaults.color = "#8b97b3";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.borderColor = "#1f2b47";

const state = { view: "dashboard", segment: "all", q: "", sort: "churn_probability", desc: true, page: 1, size: 8, selected: null, summary: null, detail: null, tab: "factors" };
const charts = {};

/* ------------------------------------------------------------------ campaign (per-browser) */
const store = {
  get() { try { return JSON.parse(localStorage.getItem("churnopt.campaign") || "{}"); } catch { return {}; } },
  set(v) { try { localStorage.setItem("churnopt.campaign", JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
function updateCampaignBadge() {
  const n = Object.keys(store.get()).length;
  const b = $("#campaignCount");
  b.hidden = n === 0;
  b.textContent = n;
}
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ------------------------------------------------------------------ routing */
function route() {
  const v = (location.hash || "#dashboard").slice(1);
  state.view = ["dashboard", "customers", "whatif", "actions", "model", "about"].includes(v) ? v : "dashboard";
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === state.view));
  $$(".view").forEach((s) => s.classList.remove("active", "customers"));
  const section = state.view === "customers" ? "dashboard" : state.view;
  $(`#view-${section}`).classList.add("active");
  if (state.view === "customers") {
    $("#view-dashboard").classList.add("customers");
    $("#heroTitle").textContent = "Customers";
    $("#heroSub").textContent = "Every customer, scored and segmented. Click a row for the full insight.";
    state.size = 15;
  } else if (state.view === "dashboard") {
    $("#heroTitle").textContent = "Churn & Retention Optimizer";
    $("#heroSub").textContent = "Predict, understand and take action to keep your customers.";
    state.size = 8;
  }
  if (section === "dashboard") loadTable();
  if (state.view === "whatif") renderWhatIf();
  if (state.view === "actions") renderActions();
  if (state.view === "model") renderModel();
  $("#sidebar").classList.remove("open");
  openSheet(false);
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ dashboard */
function renderSummary(s) {
  const k = s.kpis;
  $("#kTotal").textContent = k.total_customers.toLocaleString();
  $("#kTotalSub").textContent = `${k.high_risk.toLocaleString()} at high risk`;
  $("#kChurn").textContent = pct(k.predicted_churn_rate);
  $("#kChurnSub").textContent = `historical actual: ${pct(k.actual_churn_rate)}`;
  $("#kRisk").textContent = usd(k.revenue_at_risk);
  $("#kRiskSub").textContent = `${pct(k.revenue_at_risk / (k.monthly_revenue * 12))} of annual revenue`;
  $("#kSave").textContent = k.customers_to_save.toLocaleString();
  $("#kSaveSub").textContent = `offer worth ${usd(k.campaign_value)} expected`;
  $("#sideValue").textContent = usd(k.campaign_value);
  $("#sideNote").textContent = `${k.customers_to_save.toLocaleString()} persuadable customers`;
  $("#modelChip").textContent = `XGBoost + ${s.model.uplift_learner} · live`;
  $("#donutTotal").textContent = k.total_customers.toLocaleString();

  // churn by tenure
  const t = s.churn_by_tenure;
  const ctx = $("#tenureChart").getContext("2d");
  const grad = (c) => { const g = ctx.createLinearGradient(0, 0, 0, 210); g.addColorStop(0, c + "55"); g.addColorStop(1, c + "00"); return g; };
  charts.tenure = new Chart(ctx, {
    type: "line",
    data: {
      labels: t.labels.map((l) => `${l} mo`),
      datasets: [
        { label: "Actual churn", data: t.actual, borderColor: "#f25f5c", backgroundColor: grad("#f25f5c"), fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: "#f25f5c" },
        { label: "Predicted churn", data: t.predicted, borderColor: "#4f8cff", backgroundColor: grad("#4f8cff"), fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: "#4f8cff" },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${pct(c.parsed.y)}` } } },
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => `${Math.round(v * 100)}%` }, grid: { color: "#1a2440" } },
        x: { grid: { display: false }, title: { display: true, text: "Tenure (months)" } } },
    },
  });

  // segments donut
  charts.seg = new Chart($("#segChart"), {
    type: "doughnut",
    data: { labels: s.segments.map((x) => x.name), datasets: [{ data: s.segments.map((x) => x.count), backgroundColor: s.segments.map((x) => SEG[x.name].color), borderColor: "#111a30", borderWidth: 3, hoverOffset: 6 }] },
    options: { cutout: "70%", maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.parsed.toLocaleString()}` } } },
      onClick: (_, els) => { if (els.length) setFilter(SEG[s.segments[els[0].index].name].key); } },
  });
  $("#segList").innerHTML = s.segments.map((x) => `
    <div class="seg-row" data-key="${SEG[x.name].key}" title="${esc(x.description)}">
      <span class="sw" style="background:${SEG[x.name].color}"></span>${esc(x.name)}<span class="pct">${pct(x.share, 0)}</span>
    </div>`).join("");
  $$("#segList .seg-row").forEach((r) => r.addEventListener("click", () => setFilter(r.dataset.key)));

  // factors
  const fc = ["#f25f5c", "#f97b4f", "#f5c542", "#4f8cff", "#8b5cf6"];
  const max = Math.max(...s.top_factors.map((f) => f.share));
  $("#factorBars").innerHTML = s.top_factors.map((f, i) => `
    <div class="bar-row"><span>${esc(f.name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(f.share / max) * 100}%;background:${fc[i]}"></div></div>
      <span class="num">${pct(f.share, 0)}</span></div>`).join("");

  $("#segExplain").innerHTML = s.segments.map((x) => `<li><strong style="color:${SEG[x.name].color}">${esc(x.name)}</strong> (${pct(x.share, 0)}): ${esc(x.description)}</li>`).join("");
}

function renderFilters() {
  $("#filters").innerHTML = FILTERS.map(([k, l]) => `<button class="pill ${state.segment === k ? "active" : ""}" data-k="${k}">${l}</button>`).join("");
  $$("#filters .pill").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.k)));
}
function setFilter(k) {
  state.segment = k;
  state.page = 1;
  renderFilters();
  loadTable();
  $(".bottom").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadTable() {
  const p = new URLSearchParams({ segment: state.segment, q: state.q, sort: state.sort, desc: state.desc, page: state.page, size: state.size });
  const data = await api(`/api/customers?${p}`);
  $("#exportBtn").href = `/api/customers/export.csv?${new URLSearchParams({ segment: state.segment, q: state.q })}`;
  $("#tableHint").textContent = state.segment === "all" ? "" : FILTERS.find((f) => f[0] === state.segment)?.[1] || "";
  $$("th[data-sort]").forEach((th) => {
    th.querySelector(".arrow")?.remove();
    if (th.dataset.sort === state.sort) th.insertAdjacentHTML("beforeend", `<span class="arrow">${state.desc ? "↓" : "↑"}</span>`);
  });
  $("#rows").innerHTML = data.rows.length ? data.rows.map((r) => `
    <tr data-id="${esc(r.customerID)}" class="${r.customerID === state.selected ? "selected" : ""}">
      <td class="c-id"><div class="mono">${esc(r.customerID)}</div><div class="hint">${esc(r.Contract)} · ${r.tenure} mo · $${Math.round(r.MonthlyCharges)}/mo</div></td>
      <td class="c-prob"><div class="prob"><b style="color:${riskColor(r.churn_probability)}">${Math.round(r.churn_probability * 100)}%</b>
        <div class="bar-track"><div class="bar-fill" style="width:${r.churn_probability * 100}%;background:${riskColor(r.churn_probability)}"></div></div></div></td>
      <td class="c-seg">${tag(r.segment)}</td>
      <td class="factors-cell c-fac" title="${esc(r.main_factors)}">${esc(r.main_factors)}</td>
      <td class="c-act">${esc(r.primary_action)}</td>
      <td class="c-btn"><button class="btn primary sm" data-view-id="${esc(r.customerID)}">View</button></td>
    </tr>`).join("") : `<tr><td colspan="6" class="empty">No customers match.</td></tr>`;
  $$("#rows tr[data-id]").forEach((tr) => tr.addEventListener("click", () => selectCustomer(tr.dataset.id, true)));

  const pages = Math.max(1, Math.ceil(data.total / state.size));
  const from = data.total ? (state.page - 1) * state.size + 1 : 0;
  $("#pageInfo").textContent = `Showing ${from}–${Math.min(state.page * state.size, data.total)} of ${data.total.toLocaleString()} customers`;
  const nums = new Set([1, pages, state.page - 1, state.page, state.page + 1].filter((n) => n >= 1 && n <= pages));
  let html = `<button ${state.page === 1 ? "disabled" : ""} data-p="${state.page - 1}" aria-label="Previous page">‹</button>`;
  let prev = 0;
  [...nums].sort((a, b) => a - b).forEach((n) => {
    if (n - prev > 1) html += `<span style="color:var(--dim)">…</span>`;
    html += `<button class="${n === state.page ? "active" : ""}" data-p="${n}">${n}</button>`;
    prev = n;
  });
  html += `<button ${state.page === pages ? "disabled" : ""} data-p="${state.page + 1}" aria-label="Next page">›</button>`;
  $("#pages").innerHTML = html;
  $$("#pages button[data-p]").forEach((b) => b.addEventListener("click", () => { state.page = +b.dataset.p; loadTable(); }));

  if (!state.selected && data.rows.length) selectCustomer(data.rows[0].customerID);
}

/* ------------------------------------------------------------------ insight panel */
const isPhone = () => window.matchMedia("(max-width: 900px)").matches;
function openSheet(open) {
  $("#insight").classList.toggle("open", open);
  $("#sheetBackdrop").classList.toggle("show", open);
  document.body.classList.toggle("no-scroll", open);
}
async function selectCustomer(id, fromTap = false) {
  state.selected = id;
  $$("#rows tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));
  state.detail = await api(`/api/customers/${encodeURIComponent(id)}`);
  renderInsight();
  if (!fromTap) return;
  if (isPhone()) openSheet(true);
  else if (window.innerWidth < 1180) $("#insight").scrollIntoView({ behavior: "smooth" });
}

function spark(values, color) {
  return `<span class="spark">${values.map((v) => `<i style="height:${Math.max(4, v * 22)}px;background:${color}"></i>`).join("")}</span>`;
}

function renderInsight() {
  const c = state.detail;
  if (!c) return;
  const color = riskColor(c.churn_probability);
  const inCampaign = !!store.get()[c.customerID];
  const maxImpact = Math.max(0.01, ...c.factors.map((f) => f.impact));
  const fc = ["#f25f5c", "#f97b4f", "#f5c542", "#4f8cff"];
  const tabs = {
    factors: c.factors.length ? c.factors.map((f, i) => `
      <div class="bar-row" style="grid-template-columns:150px 1fr 40px;margin-bottom:10px">
        <span style="font-size:12.5px">${esc(f.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(f.impact / maxImpact) * 100}%;background:${fc[i % 4]}"></div></div>
        <span class="num" style="font-size:12.5px">${f.impact.toFixed(2)}</span></div>`).join("") + `<div class="hint">SHAP impact on churn log-odds</div>`
      : `<div class="hint">No factors pushing this customer toward churn.</div>`,
    profile: `<div class="kv">
      <span>Contract</span><b>${esc(c.Contract)}</b><span>Tenure</span><b>${c.tenure} months</b>
      <span>Monthly bill</span><b>$${c.MonthlyCharges.toFixed(2)}</b><span>Internet</span><b>${esc(c.InternetService)}</b>
      <span>Tech support</span><b>${esc(c.TechSupport)}</b><span>Payment</span><b>${esc(c.PaymentMethod)}</b>
      <span>Senior / Partner</span><b>${esc(c.SeniorCitizen)} / ${esc(c.Partner)}</b><span>Paperless</span><b>${esc(c.PaperlessBilling)}</b></div>`,
    offer: `<div class="kv">
      <span>Churn cut by offer</span><b>${(c.offer_uplift * 100).toFixed(1)} pp</b>
      <span>Expected value</span><b style="color:${c.offer_expected_value_usd > 20 ? "var(--green)" : "var(--amber)"}">${usd(c.offer_expected_value_usd)}</b>
      <span>Decision</span><b>${c.recommendation === "send_offer" ? "Send offer" : "Don't send"}</b></div>
      <div class="reason">${esc(c.segment_description)}</div><div class="reason">${esc(c.recommendation_reason)}</div>`,
  };
  $("#insight").innerHTML = `
    <div class="sheet-handle"></div>
    <div class="card-head"><h3>Customer Insight</h3><div style="display:flex;gap:8px"><a class="btn sm" href="#whatif" id="toWhatIf">Simulate</a>
      <button class="btn sm sheet-close" id="closeSheet" aria-label="Close">✕</button></div></div>
    <div class="cust-head">
      <div class="avatar" style="color:${SEG[c.segment].color}">${esc(c.customerID.slice(0, 2))}</div>
      <div><div class="id mono">${esc(c.customerID)}</div><div class="meta">${esc(c.gender)} · ${c.tenure} mo · ${esc(c.Contract)}</div></div>
      ${tag(c.segment)}
    </div>
    <div class="mini">
      <div><div class="label">Churn Probability</div><div class="v"><span style="color:${color}">${Math.round(c.churn_probability * 100)}%</span>${spark([0.25, 0.45, 0.65, c.churn_probability].map((v) => Math.min(v, c.churn_probability)), color)}</div></div>
      <div><div class="label">Revenue at Risk (12 mo)</div><div class="v"><span>${usd(c.revenue_at_risk)}</span>${spark([0.3, 0.5, 0.7, 1], "#22c55e")}</div></div>
    </div>
    <div class="tabs">${[["factors", "Key Factors"], ["profile", "Profile"], ["offer", "Offer"]].map(([k, l]) => `<button class="${state.tab === k ? "active" : ""}" data-tab="${k}">${l}</button>`).join("")}</div>
    <div class="tab-body">${tabs[state.tab]}</div>
    <div class="actions">
      <div class="actions-head"><h4><svg class="icon" viewBox="0 0 24 24"><path d="M12 3l1.9 5.8H20l-4.9 3.6 1.9 5.8L12 14.6 7 18.2l1.9-5.8L4 8.8h6.1z"/></svg>Recommended Actions</h4>
        <button class="btn primary sm" id="applyBtn">${inCampaign ? "Update plan" : "Add to plan"}</button></div>
      ${c.actions.map((a, i) => `<label class="check"><input type="checkbox" value="${esc(a)}" ${i < 2 || inCampaign ? "checked" : ""}>${esc(a)}</label>`).join("")}
    </div>`;
  $("#closeSheet").addEventListener("click", () => openSheet(false));
  $$("#insight .tabs button").forEach((b) => b.addEventListener("click", () => { state.tab = b.dataset.tab; renderInsight(); }));
  $("#applyBtn").addEventListener("click", () => {
    const actions = $$("#insight .check input:checked").map((i) => i.value);
    const camp = store.get();
    if (!actions.length) { delete camp[c.customerID]; toast("Removed from plan"); } else {
      camp[c.customerID] = { actions, segment: c.segment, prob: c.churn_probability, value: c.offer_expected_value_usd, monthly: c.MonthlyCharges };
      toast(`${c.customerID} added to retention plan`);
    }
    store.set(camp);
    updateCampaignBadge();
    renderInsight();
  });
}

/* ------------------------------------------------------------------ what-if */
const FIELDS = [
  ["Account"],
  ["tenure", "Months as customer", "range", [0, 72]],
  ["Contract", "Contract", ["Month-to-month", "One year", "Two year"]],
  ["MonthlyCharges", "Monthly bill ($)", "number", [18, 120]],
  ["PaymentMethod", "Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]],
  ["PaperlessBilling", "Paperless billing", ["Yes", "No"]],
  ["PhoneService", "Phone service", ["Yes", "No"]],
  ["Services"],
  ["InternetService", "Internet service", ["Fiber optic", "DSL", "No"]],
  ["OnlineSecurity", "Online security", "net"], ["OnlineBackup", "Online backup", "net"],
  ["DeviceProtection", "Device protection", "net"], ["TechSupport", "Tech support", "net"],
  ["StreamingTV", "Streaming TV", "net"], ["StreamingMovies", "Streaming movies", "net"],
  ["MultipleLines", "Multiple lines", "phone"],
  ["Customer"],
  ["gender", "Gender", ["Female", "Male"]], ["SeniorCitizen", "Senior citizen", ["No", "Yes"]],
  ["Partner", "Partner", ["No", "Yes"]], ["Dependents", "Dependents", ["No", "Yes"]],
];
const DEFAULT_CUSTOMER = { tenure: 3, Contract: "Month-to-month", MonthlyCharges: 85, PaymentMethod: "Electronic check", PaperlessBilling: "Yes", PhoneService: "Yes", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "Yes", StreamingTV: "Yes", StreamingMovies: "No", MultipleLines: "No", gender: "Female", SeniorCitizen: "No", Partner: "No", Dependents: "No" };
let whatIf = { ...DEFAULT_CUSTOMER };

function optionsFor(type) {
  if (type === "net") return whatIf.InternetService === "No" ? ["No internet service"] : ["No", "Yes"];
  if (type === "phone") return whatIf.PhoneService === "No" ? ["No phone service"] : ["No", "Yes"];
  return type;
}
function renderWhatIf() {
  if (state.detail && renderWhatIf.loadedFrom !== state.detail.customerID && location.hash === "#whatif" && renderWhatIf.fromInsight) {
    FIELDS.forEach(([k]) => { if (k in state.detail) whatIf[k] = state.detail[k]; });
    renderWhatIf.loadedFrom = state.detail.customerID;
  }
  renderWhatIf.fromInsight = false;
  $("#whatifForm").innerHTML = FIELDS.map(([k, label, type, range]) => {
    if (!label) return `<div class="form-section">${k}</div>`;
    if (type === "range") return `<div class="field"><label>${label}: <b id="v_${k}">${whatIf[k]}</b></label><input type="range" name="${k}" min="${range[0]}" max="${range[1]}" value="${whatIf[k]}"></div>`;
    if (type === "number") return `<div class="field"><label>${label}</label><input type="number" name="${k}" min="${range[0]}" max="${range[1]}" step="1" value="${whatIf[k]}"></div>`;
    const opts = optionsFor(type);
    if (!opts.includes(whatIf[k])) whatIf[k] = opts[0];
    return `<div class="field"><label>${label}</label><select name="${k}">${opts.map((o) => `<option ${o === whatIf[k] ? "selected" : ""}>${o}</option>`).join("")}</select></div>`;
  }).join("") + `<div class="form-section" style="display:flex;gap:10px;align-items:center;text-transform:none;letter-spacing:0">
      <button type="button" class="btn sm" id="resetWhatIf">Reset</button>
      <span class="hint">${renderWhatIf.loadedFrom ? `Started from customer ${esc(renderWhatIf.loadedFrom)}` : "Example customer"}</span></div>`;
  $("#whatifForm").oninput = (e) => {
    const el = e.target;
    whatIf[el.name] = el.type === "range" || el.type === "number" ? +el.value : el.value;
    if (el.type === "range") $(`#v_${el.name}`).textContent = el.value;
    if (["InternetService", "PhoneService"].includes(el.name)) { renderWhatIf(); return; }
    predictWhatIf();
  };
  $("#resetWhatIf").onclick = () => { whatIf = { ...DEFAULT_CUSTOMER }; renderWhatIf.loadedFrom = null; renderWhatIf(); };
  predictWhatIf();
}
async function predictWhatIf() {
  clearTimeout(predictWhatIf._t);
  predictWhatIf._t = setTimeout(async () => {
    const body = { ...whatIf, TotalCharges: whatIf.tenure * whatIf.MonthlyCharges };
    const r = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const color = riskColor(r.churn_probability);
    $("#whatifSticky").innerHTML = `<b style="color:${color}">${Math.round(r.churn_probability * 100)}%</b>
      <span>${r.risk_level} risk</span><span class="sep"></span>
      <span class="${r.recommendation === "send_offer" ? "ok" : "no"}">${r.recommendation === "send_offer" ? "✓ Send offer" : "✕ No offer"}</span>
      <span class="more">Details ↓</span>`;
    $("#whatifResult").innerHTML = `
      <h3>Prediction</h3>
      <div class="gauge"><canvas id="gauge"></canvas><div class="gauge-center"><b style="color:${color}">${Math.round(r.churn_probability * 100)}%</b><span class="hint">${r.risk_level.toUpperCase()} churn risk</span></div></div>
      <div class="verdict ${r.recommendation === "send_offer" ? "good" : "bad"}"><b>${r.recommendation === "send_offer" ? "✓ Send the retention offer." : "✕ Don't send the offer."}</b> ${esc(r.recommendation_reason)}</div>
      <div class="mini">
        <div><div class="label">Offer cuts churn by</div><div class="v">${(r.offer_uplift * 100).toFixed(1)} pp</div></div>
        <div><div class="label">Expected offer value</div><div class="v">${usd(r.offer_expected_value_usd)}</div></div>
      </div>
      <h3 style="font-size:15px">Why? (SHAP)</h3>
      ${r.top_risk_factors.length ? r.top_risk_factors.map((f) => `<div class="bar-row" style="grid-template-columns:150px 1fr 40px;margin-bottom:9px"><span style="font-size:12.5px">${esc(f.feature)} = ${esc(f.value)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, f.impact * 120)}%;background:#f25f5c"></div></div><span class="num" style="font-size:12.5px">+${f.impact.toFixed(2)}</span></div>`).join("") : `<div class="hint">Nothing is pushing this customer toward churn.</div>`}`;
    charts.gauge?.destroy();
    charts.gauge = new Chart($("#gauge"), {
      type: "doughnut",
      data: { datasets: [{ data: [r.churn_probability, 1 - r.churn_probability], backgroundColor: [color, "#1b2643"], borderWidth: 0 }] },
      options: { rotation: -90, circumference: 180, cutout: "78%", maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: false } }, animation: { duration: 300 } },
    });
  }, 150);
}

/* ------------------------------------------------------------------ actions view */
function renderActions() {
  const camp = store.get();
  const ids = Object.keys(camp);
  if (!ids.length) {
    $("#actionsBody").innerHTML = `<div class="placeholder">No customers in your plan yet.<br><br><a class="btn primary" href="#customers">Browse customers</a></div>`;
    return;
  }
  const total = ids.reduce((s, id) => s + (camp[id].actions.some((a) => a.startsWith("Send")) ? camp[id].value : 0), 0);
  $("#actionsBody").innerHTML = `
    <div class="card-head"><h3>${ids.length} customer${ids.length > 1 ? "s" : ""} in plan · expected offer value ${usd(total)}</h3>
      <div style="display:flex;gap:8px"><button class="btn" id="clearPlan">Clear</button><button class="btn primary" id="dlPlan">Download CSV</button></div></div>
    <div class="table-wrap"><table><thead><tr><th class="nosort">Customer ID</th><th class="nosort">Segment</th><th class="nosort">Churn</th><th class="nosort">Offer value</th><th class="nosort">Planned actions</th><th class="nosort"></th></tr></thead>
    <tbody>${ids.map((id) => { const c = camp[id]; return `<tr data-id="${esc(id)}"><td class="mono a-id">${esc(id)}</td><td class="a-seg">${tag(c.segment)}</td>
      <td class="a-prob" style="color:${riskColor(c.prob)}">${Math.round(c.prob * 100)}% <span class="ph-label">churn</span></td><td class="a-val">${usd(c.value)} <span class="ph-label">offer value</span></td>
      <td class="a-acts" style="white-space:normal">${c.actions.map(esc).join("<br>")}</td><td class="a-rm"><button class="btn sm" data-remove="${esc(id)}">Remove</button></td></tr>`; }).join("")}</tbody></table></div>`;
  $$("[data-remove]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const c = store.get(); delete c[b.dataset.remove]; store.set(c); updateCampaignBadge(); renderActions();
  }));
  $("#clearPlan").onclick = () => { store.set({}); updateCampaignBadge(); renderActions(); };
  $("#dlPlan").onclick = () => {
    const lines = [["customerID", "segment", "churn_probability", "offer_expected_value_usd", "actions"]]
      .concat(ids.map((id) => [id, camp[id].segment, camp[id].prob, camp[id].value, `"${camp[id].actions.join("; ")}"`]));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.map((l) => l.join(",")).join("\n")], { type: "text/csv" }));
    a.download = "retention_plan.csv";
    a.click();
  };
}

/* ------------------------------------------------------------------ model view */
const SHORT_POLICY = {
  "Offer to everyone": "Everyone",
  "Top 20% by churn risk (traditional)": "Risk, top 20%",
  "Top 20% by predicted uplift": "Uplift, top 20%",
  "Churn risk, same # of offers as uplift policy": "Risk, same budget",
  "Uplift model: offer if expected value > 0": "Uplift, value > 0",
  "Oracle (true effect, upper bound)": "Oracle (max)",
};
function renderModel() {
  const m = state.summary?.model;
  if (!m || renderModel.done) return;
  renderModel.done = true;
  const u = m.uplift_vs_risk;
  $("#metrics").innerHTML = [
    ["ROC-AUC", m.roc_auc.toFixed(3), `Logistic baseline ${m.baseline_roc_auc.toFixed(3)}`],
    ["PR-AUC", m.pr_auc.toFixed(3), `F1 ${m.f1.toFixed(3)} · recall ${pct(m.recall, 0)}`],
    ["Top-decile lift", `${m.lift.toFixed(1)}x`, "10% riskiest hold 28% of churners"],
    ["Uplift vs risk targeting", `${(u.uplift / u.risk).toFixed(1)}x`, `${usd(u.uplift)} vs ${usd(u.risk)} at same budget`],
  ].map(([l, v, s]) => `<div class="card metric"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");

  const pol = m.policies.filter((p) => p.policy !== "No offers");
  new Chart($("#policyChart"), {
    type: "bar",
    data: { labels: pol.map((p) => SHORT_POLICY[p.policy] || p.policy),
      datasets: [{ data: pol.map((p) => p.net_value_usd), borderRadius: 6,
        backgroundColor: pol.map((p) => (p.net_value_usd < 0 ? "#f25f5c" : p.policy.includes("uplift") || p.policy.includes("Uplift") ? "#4f8cff" : p.policy.startsWith("Oracle") ? "#2dd4bf" : "#5d6a88")) }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` Net value ${usd(c.parsed.x)}` } } },
      scales: { x: { ticks: { callback: (v) => usd(v) }, grid: { color: "#1a2440" } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } },
  });
}

/* ------------------------------------------------------------------ init */
async function init() {
  updateCampaignBadge();
  renderFilters();
  $$("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
    if (state.sort === th.dataset.sort) state.desc = !state.desc; else { state.sort = th.dataset.sort; state.desc = true; }
    state.page = 1;
    loadTable();
  }));
  let t;
  $("#tableSearch").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { state.q = e.target.value.trim(); state.page = 1; loadTable(); }, 250); });
  $("#globalSearch").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    state.q = e.target.value.trim(); state.page = 1; $("#tableSearch").value = state.q;
    if (location.hash !== "#customers") location.hash = "#customers"; else loadTable();
  });
  $("#whatifSticky").addEventListener("click", () => $("#whatifResult").scrollIntoView({ behavior: "smooth" }));
  $("#menuBtn").addEventListener("click", () => { $("#sidebar").classList.toggle("open"); $("#sheetBackdrop").classList.toggle("show", $("#sidebar").classList.contains("open")); });
  $("#sheetBackdrop").addEventListener("click", () => { openSheet(false); $("#sidebar").classList.remove("open"); });
  document.addEventListener("click", (e) => { if (e.target.id === "toWhatIf") renderWhatIf.fromInsight = true; });

  try {
    state.summary = await api("/api/summary");
  } catch (err) {
    $("#loading").innerHTML = `<div>Could not load data (${esc(err.message)}). Is the API running?</div>`;
    return;
  }
  // Show the dashboard section before drawing, so Chart.js measures real sizes.
  $("#view-dashboard").classList.add("active");
  renderSummary(state.summary);
  window.addEventListener("hashchange", route);
  route();
  $("#loading").classList.add("hide");
}
init();
