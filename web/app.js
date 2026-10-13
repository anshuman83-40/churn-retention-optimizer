/* ChurnOpt dashboard: plain JS + Chart.js, talks to the FastAPI backend. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const usd = (v) => {
  const a = Math.abs(v), sign = v < 0 ? "-" : "";
  return sign + (a >= 1e6 ? `$${(a / 1e6).toFixed(2)}M` : a >= 1e4 ? `$${(a / 1e3).toFixed(1)}k` : `$${Math.round(a).toLocaleString()}`);
};
// Every data call names the model set to use: "own" (ChurnUpliftNet) or "library" (XGBoost + S-learner).
const withModel = (path) => `${path}${path.includes("?") ? "&" : "?"}model=${state.model}`;
const api = async (path, opts) => {
  const r = await fetch(withModel(path), opts);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const icon = (id) => `<svg class="icon"><use href="#i-${id}"/></svg>`;

// Colours are CSS variables so every component follows the light/dark theme.
const SEG = {
  "Persuadable": { c: "--green", key: "persuadable", ini: "PE" },
  "Lost Cause": { c: "--red", key: "lost_cause", ini: "LC" },
  "Sleeping Dog": { c: "--purple", key: "sleeping_dog", ini: "SD" },
  "Loyal": { c: "--blue", key: "loyal", ini: "LO" },
  "Monitor": { c: "--amber", key: "monitor", ini: "MO" },
};
const FILTERS = [
  ["all", "All"], ["high", "High Risk"], ["persuadable", "Persuadable"],
  ["lost_cause", "Lost Cause"], ["sleeping_dog", "Sleeping Dog"], ["loyal", "Loyal"],
];
const VIEWS = ["overview", "customers", "whatif", "actions", "model", "about"];
const riskVar = (p) => (p >= 0.6 ? "--red" : p >= 0.3 ? "--amber" : "--green");
const tag = (seg) => `<span class="tag" style="--c:var(${SEG[seg]?.c || "--muted"})">${esc(seg)}</span>`;
const meter = (v, cvar) => `<div class="meter"><i style="width:${Math.max(0, Math.min(1, v)) * 100}%;background:var(${cvar})"></i></div>`;

const initialModel = () => {
  const q = /[?&]model=(own|library)/.exec(location.search);
  if (q) return q[1];
  try { const m = localStorage.getItem("churnopt.model"); if (m === "own" || m === "library") return m; } catch { /* ignore */ }
  return "own";
};
const state = { model: initialModel(), view: "overview", segment: "all", q: "", sort: "churn_probability", desc: true, page: 1, size: 12, selected: null, summary: null, detail: null, tab: "factors" };
const charts = {};

/* ------------------------------------------------------------------ theme */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("churnopt.theme", t); } catch { /* storage unavailable */ }
  if (state.view === "model") renderModel(); // Chart.js canvases need redrawing with the new colours
  else renderModel.stale = true;
}

/* ------------------------------------------------------------------ model switch */
function paintModelSwitch() {
  $$("#modelSwitch button").forEach((b) => b.classList.toggle("active", b.dataset.model === state.model));
}
async function setModel(m) {
  if (m === state.model) return;
  state.model = m;
  try { localStorage.setItem("churnopt.model", m); } catch { /* storage unavailable */ }
  paintModelSwitch();
  toast("Scoring customers with the selected model…");
  state.summary = await api("/api/summary");
  state.selected = null;
  state.detail = null;
  renderOverview(state.summary);
  renderCandidates();
  renderModel.stale = true;
  addMsg("bot", `Now using ${esc(state.summary.served.name)}. Numbers on every page come from this model.`);
  toast(`Model: ${state.summary.served.name}`);
  route();
}
const impactLabel = () => (state.model === "own" ? "Occlusion impact on churn log-odds (own model)" : "SHAP impact on churn log-odds");

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
  let v = (location.hash || "#overview").slice(1);
  if (v === "dashboard") v = "overview";
  state.view = VIEWS.includes(v) ? v : "overview";
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === state.view));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${state.view}`));
  openSheet(false);
  if (state.view === "customers") loadTable();
  if (state.view === "whatif") renderWhatIf();
  if (state.view === "actions") renderActions();
  if (state.view === "model") renderModel();
  window.scrollTo(0, 0);
}
function go(view) {
  if (location.hash === `#${view}`) route(); else location.hash = `#${view}`;
}

/* ------------------------------------------------------------------ SVG tick gauge */
function tickGauge(el, parts, { ticks = 46, sweep = 264 } = {}) {
  const cx = 100, cy = 100, r0 = 70, len = 24, w = 5.2;
  let html = "", acc = 0, idx = 0;
  const bounds = parts.map((p) => (acc += p.share));
  for (let i = 0; i < ticks; i++) {
    const f = (i + 0.5) / ticks;
    while (idx < bounds.length - 1 && f > bounds[idx] / acc) idx++;
    const ang = -sweep / 2 + sweep * (i / (ticks - 1));
    html += `<rect data-p="${idx}" x="${cx - w / 2}" y="${cy - r0 - len}" width="${w}" height="${len}" rx="${w / 2}" transform="rotate(${ang.toFixed(2)} ${cx} ${cy})" style="fill:var(${parts[idx].c})"/>`;
  }
  el.innerHTML = `<svg viewBox="0 0 200 172" role="img">${html}</svg>`;
}
function highlightGauge(el, i) {
  $$("rect", el).forEach((r) => { r.style.opacity = i == null || +r.dataset.p === i ? 1 : 0.22; });
}

/* ------------------------------------------------------------------ overview */
function renderOverview(s) {
  const k = s.kpis;
  $("#stamp").textContent = `${s.served.name} · ${k.total_customers.toLocaleString()} customers`;

  // --- churn by tenure: CSS bars (actual) with a marker line for the model's prediction
  const t = s.churn_by_tenure;
  const top = Math.ceil(Math.max(...t.actual, ...t.predicted) * 10) / 10;
  const peak = t.actual.indexOf(Math.max(...t.actual));
  const steps = Math.round(top * 10);
  const axis = Array.from({ length: steps + 1 }, (_, i) => i / 10);
  $("#tenureBars").innerHTML = `
    <div class="bars-axis" style="margin-bottom:24px">${axis.map((v) => `<span style="bottom:${(v / top) * 100}%">${Math.round(v * 100)}%</span>`).join("")}</div>
    <div class="bars-plot">
      ${axis.map((v) => `<div class="gridline" style="bottom:calc(24px + (100% - 24px) * ${v / top})"></div>`).join("")}
      ${t.labels.map((l, i) => `
        <div class="bar-col ${i === peak ? "peak" : ""}">
          <div class="bar-track"></div>
          <div class="bar-fill" data-h="${(t.actual[i] / top) * 100}" style="height:0%">${Math.round(t.actual[i] * 100)}%</div>
          <div class="bar-pred" data-b="${(t.predicted[i] / top) * 100}" style="bottom:0%"></div>
          <div class="bar-tip" data-b="${(Math.max(t.actual[i], t.predicted[i]) / top) * 100}" style="bottom:0%">Actual ${pct(t.actual[i], 0)} · Model ${pct(t.predicted[i], 0)}</div>
          <div class="bar-x">${esc(l)}<span> mo</span></div>
        </div>`).join("")}
    </div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$("#tenureBars .bar-fill").forEach((b) => { b.style.height = `${b.dataset.h}%`; });
    $$("#tenureBars [data-b]").forEach((b) => { b.style.bottom = `${b.dataset.b}%`; });
  }));

  // --- KPI cards with block meters (every share is a real proportion)
  const annual = k.monthly_revenue * 12;
  const kpis = [
    ["users", "Total Customers", k.total_customers.toLocaleString(), `${k.high_risk.toLocaleString()} high risk`, "warn", "High-risk share", k.high_risk / k.total_customers],
    ["pie", "Predicted Churn", pct(k.predicted_churn_rate), `actual ${pct(k.actual_churn_rate)}`, "", "Of all customers", k.predicted_churn_rate],
    ["dollar", "Revenue at Risk", usd(k.revenue_at_risk), "next 12 mo", "warn", "Of annual revenue", k.revenue_at_risk / annual],
    ["shield", "Customers to Save", k.customers_to_save.toLocaleString(), `+${usd(k.campaign_value)}`, "", "Worth an offer", k.customers_to_save / k.total_customers],
  ];
  const N = 14;
  $("#kpis").innerHTML = kpis.map(([ic, label, value, chip, tone, mlabel, share], i) => `
    <div class="card kpi reveal" style="--i:${i + 2}">
      <div class="kpi-label">${icon(ic)}${label}</div>
      <div class="kpi-value"><b>${value}</b><span class="chip ${tone}">${chip}</span></div>
      <div class="kpi-meter"><span>${mlabel}</span><span>${pct(share, 0)}</span></div>
      <div class="blocks" role="img" aria-label="${mlabel}: ${pct(share, 0)}">${Array.from({ length: N }, (_, j) => {
        const on = j < Math.round(share * N);
        return `<i class="${on ? "on" : ""}" style="${on ? `opacity:${1 - (j / N) * 0.6}` : ""}"></i>`;
      }).join("")}</div>
    </div>`).join("");

  // --- segments: tick gauge + legend
  const segs = s.segments.map((x) => ({ ...x, c: SEG[x.name].c }));
  const gauge = $("#segGauge");
  tickGauge(gauge, segs);
  const show = (i) => {
    const x = segs[i ?? 0];
    $("#segPct").textContent = pct(x.share, 0);
    $("#segName").textContent = x.name;
    highlightGauge(gauge, i);
    $$("#segList .seg-row").forEach((r, j) => r.classList.toggle("on", j === i));
  };
  $("#segList").innerHTML = segs.map((x, i) => `
    <button class="seg-row" data-i="${i}" title="${esc(x.description)}"><i style="background:var(${x.c})"></i>${esc(x.name)}
      <em>${x.count.toLocaleString()} · ${pct(x.share, 0)}</em></button>`).join("");
  $$("#segList .seg-row").forEach((r) => {
    r.addEventListener("mouseenter", () => show(+r.dataset.i));
    r.addEventListener("focus", () => show(+r.dataset.i));
    r.addEventListener("click", () => setFilter(SEG[segs[+r.dataset.i].name].key));
  });
  $("#segList").addEventListener("mouseleave", () => show(null));
  show(null);

  $("#segExplain").innerHTML = s.segments.map((x) => `<li><strong style="color:var(${SEG[x.name].c})">${esc(x.name)}</strong> (${pct(x.share, 0)}): ${esc(x.description)}</li>`).join("");
}

async function renderCandidates() {
  const data = await api("/api/customers?segment=persuadable&sort=offer_expected_value_usd&desc=true&size=3");
  $("#custCards").innerHTML = data.rows.map((r) => `
    <button class="cc" data-id="${esc(r.customerID)}">
      <div class="cc-head"><span class="initials">${SEG[r.segment].ini}</span>
        <div><span class="mono">${esc(r.customerID)}</span><small>${esc(r.Contract)} · ${r.tenure} mo</small></div></div>
      <div class="cc-row"><span>Churn risk</span><b style="color:var(${riskVar(r.churn_probability)})">${Math.round(r.churn_probability * 100)}%</b></div>
      <div class="cc-row"><span>Segment</span>${tag(r.segment)}</div>
      <div class="cc-row"><span>Offer value</span><b>+${usd(r.offer_expected_value_usd)}</b></div>
    </button>`).join("");
  $$("#custCards .cc").forEach((b) => b.addEventListener("click", () => openCustomer(b.dataset.id)));
}

/* ------------------------------------------------------------------ insights assistant (rule-based lookups) */
function addMsg(who, html) {
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", `<div class="msg ${who}"><div class="msg-who">${who === "me" ? "You" : "Insights"}</div>${html}</div>`);
  chat.scrollTop = chat.scrollHeight;
  $$(".msg-item[data-id]", chat.lastElementChild).forEach((b) => b.addEventListener("click", () => openCustomer(b.dataset.id)));
  $$("[data-filter]", chat.lastElementChild).forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); setFilter(b.dataset.filter); }));
}
const custList = (rows, right) => `<div class="msg-list">${rows.map((r) => `
  <button class="msg-item" data-id="${esc(r.customerID)}"><span><span class="mono">${esc(r.customerID)}</span>
    <small>${Math.round(r.churn_probability * 100)}% churn · ${esc(r.Contract)}</small></span><b>${right(r)}</b></button>`).join("")}</div>`;

const ANSWERS = {
  async offer() {
    const k = state.summary.kpis;
    const d = await api("/api/customers?segment=persuadable&sort=offer_expected_value_usd&size=3");
    return `${k.customers_to_save.toLocaleString()} customers are persuadable. Sending them the 20% offer is expected to return ${usd(k.campaign_value)}. Best candidates:
      ${custList(d.rows, (r) => `+${usd(r.offer_expected_value_usd)}`)}<a href="#" data-filter="persuadable">See all persuadable →</a>`;
  },
  async lost() {
    const n = state.summary.segments.find((x) => x.name === "Lost Cause").count;
    const d = await api("/api/customers?segment=lost_cause&size=3");
    return `${n.toLocaleString()} customers are high risk, but a discount is not expected to keep them. Fix the service issue instead of spending on offers.
      ${custList(d.rows, (r) => `${Math.round(r.churn_probability * 100)}%`)}<a href="#" data-filter="lost_cause">See all lost causes →</a>`;
  },
  async sleeping() {
    const n = state.summary.segments.find((x) => x.name === "Sleeping Dog").count;
    const d = await api("/api/customers?segment=sleeping_dog&size=3");
    return `${n.toLocaleString()} loyal customers are likely to react badly to an offer. Leave them alone.
      ${custList(d.rows, (r) => usd(r.offer_expected_value_usd))}<a href="#" data-filter="sleeping_dog">See all sleeping dogs →</a>`;
  },
  async drivers() {
    return `The biggest churn drivers, by share of the model's total feature impact:
      <div class="msg-list">${state.summary.top_factors.map((f) => `<div class="msg-item"><span>${esc(f.name)}</span><b>${pct(f.share, 0)}</b></div>`).join("")}</div>`;
  },
  async revenue() {
    const k = state.summary.kpis, u = state.summary.model.uplift_vs_risk;
    return `${usd(k.revenue_at_risk)} of revenue is at risk over the next 12 months, ${pct(k.revenue_at_risk / (k.monthly_revenue * 12), 0)} of annual revenue, with ${k.high_risk.toLocaleString()} customers at high risk.
      <br><br>On the held-out test, uplift targeting earned ${(u.uplift / u.risk).toFixed(1)}x what risk targeting did with the same budget (${usd(u.uplift)} vs ${usd(u.risk)}).`;
  },
};
const LABELS = { offer: "Who should get the offer?", lost: "Identify lost causes", drivers: "Top churn drivers", revenue: "Revenue at risk" };
const KEYWORDS = [
  [/offer|persuad|discount|save|target/i, "offer"], [/lost|cause/i, "lost"], [/sleep|dog|backfire/i, "sleeping"],
  [/driver|factor|why|reason|shap/i, "drivers"], [/revenue|money|risk|\$/i, "revenue"],
];
async function ask(key, text) {
  addMsg("me", esc(text || LABELS[key]));
  try {
    if (!key) {
      const hit = KEYWORDS.find(([re]) => re.test(text));
      if (hit) key = hit[1];
    }
    if (key) return addMsg("bot", await ANSWERS[key]());
    const d = await api(`/api/customers?q=${encodeURIComponent(text)}&size=3`);
    addMsg("bot", d.total
      ? `${d.total.toLocaleString()} matching customer${d.total > 1 ? "s" : ""}. Highest risk first:${custList(d.rows, (r) => `${Math.round(r.churn_probability * 100)}%`)}`
      : `No customers match "${esc(text)}". Try a customer ID such as 7590-VHVEG, or a word like "fiber" or "two year".`);
  } catch (err) {
    addMsg("bot", `Could not load that (${esc(err.message)}).`);
  }
}

/* ------------------------------------------------------------------ customers table */
function renderFilters() {
  $("#filters").innerHTML = FILTERS.map(([k, l]) => `<button class="pill ${state.segment === k ? "active" : ""}" data-k="${k}">${l}</button>`).join("");
  $$("#filters .pill").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.k)));
}
function setFilter(k) {
  state.segment = k;
  state.page = 1;
  state.selected = null; // show the top customer of the new list in the insight panel
  renderFilters();
  go("customers");
}
function openCustomer(id) {
  state.pendingOpen = id;
  go("customers");
}

async function loadTable() {
  const p = new URLSearchParams({ segment: state.segment, q: state.q, sort: state.sort, desc: state.desc, page: state.page, size: state.size });
  const data = await api(`/api/customers?${p}`);
  $("#exportBtn").href = `/api/customers/export.csv?${new URLSearchParams({ segment: state.segment, q: state.q, model: state.model })}`;
  $("#tableHint").textContent = `${FILTERS.find((f) => f[0] === state.segment)?.[1] || "All"} · ${data.total.toLocaleString()} customers`;
  $$("th[data-sort]").forEach((th) => {
    th.querySelector(".arrow")?.remove();
    if (th.dataset.sort === state.sort) th.insertAdjacentHTML("beforeend", `<span class="arrow">${state.desc ? "↓" : "↑"}</span>`);
  });
  $("#rows").innerHTML = data.rows.length ? data.rows.map((r) => `
    <tr data-id="${esc(r.customerID)}" class="${r.customerID === state.selected ? "selected" : ""}">
      <td class="c-id"><div class="mono">${esc(r.customerID)}</div><div class="hint">${esc(r.Contract)} · ${r.tenure} mo · $${Math.round(r.MonthlyCharges)}/mo</div></td>
      <td class="c-prob"><div class="prob"><b style="color:var(${riskVar(r.churn_probability)})">${Math.round(r.churn_probability * 100)}%</b>${meter(r.churn_probability, riskVar(r.churn_probability))}</div></td>
      <td class="c-seg">${tag(r.segment)}</td>
      <td class="c-fac" title="${esc(r.main_factors)}">${esc(r.main_factors)}</td>
      <td class="c-act">${esc(r.primary_action)}</td>
      <td class="c-btn"><button class="btn primary sm">View</button></td>
    </tr>`).join("") : `<tr><td colspan="6" class="empty">No customers match.</td></tr>`;
  $$("#rows tr[data-id]").forEach((tr) => tr.addEventListener("click", () => selectCustomer(tr.dataset.id, true)));

  const pages = Math.max(1, Math.ceil(data.total / state.size));
  const from = data.total ? (state.page - 1) * state.size + 1 : 0;
  $("#pageInfo").textContent = `Showing ${from}–${Math.min(state.page * state.size, data.total)} of ${data.total.toLocaleString()}`;
  const nums = new Set([1, pages, state.page - 1, state.page, state.page + 1].filter((n) => n >= 1 && n <= pages));
  let html = `<button ${state.page === 1 ? "disabled" : ""} data-p="${state.page - 1}" aria-label="Previous page">‹</button>`;
  let prev = 0;
  [...nums].sort((a, b) => a - b).forEach((n) => {
    if (n - prev > 1) html += `<span>…</span>`;
    html += `<button class="${n === state.page ? "active" : ""}" data-p="${n}">${n}</button>`;
    prev = n;
  });
  html += `<button ${state.page === pages ? "disabled" : ""} data-p="${state.page + 1}" aria-label="Next page">›</button>`;
  $("#pages").innerHTML = html;
  $$("#pages button[data-p]").forEach((b) => b.addEventListener("click", () => { state.page = +b.dataset.p; loadTable(); }));

  if (state.pendingOpen) {
    const id = state.pendingOpen;
    state.pendingOpen = null;
    selectCustomer(id, true);
  } else if (!state.selected && data.rows.length) selectCustomer(data.rows[0].customerID);
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
  else if (window.innerWidth < 1100) $("#insight").scrollIntoView({ behavior: "smooth" });
}

function renderInsight() {
  const c = state.detail;
  if (!c) return;
  const rv = riskVar(c.churn_probability);
  const inCampaign = !!store.get()[c.customerID];
  const maxImpact = Math.max(0.01, ...c.factors.map((f) => f.impact));
  const tabs = {
    factors: c.factors.length ? c.factors.map((f) => `
      <div class="frow"><span>${esc(f.label)}</span>${meter(f.impact / maxImpact, rv)}<span class="num">${f.impact.toFixed(2)}</span></div>`).join("") + `<div class="hint">${impactLabel()}</div>`
      : `<div class="hint">No factors pushing this customer toward churn.</div>`,
    profile: `<div class="kv">
      <span>Contract</span><b>${esc(c.Contract)}</b><span>Tenure</span><b>${c.tenure} month${c.tenure === 1 ? "" : "s"}</b>
      <span>Monthly bill</span><b>$${c.MonthlyCharges.toFixed(2)}</b><span>Internet</span><b>${esc(c.InternetService)}</b>
      <span>Tech support</span><b>${esc(c.TechSupport)}</b><span>Payment</span><b>${esc(c.PaymentMethod)}</b>
      <span>Senior / Partner</span><b>${esc(c.SeniorCitizen)} / ${esc(c.Partner)}</b><span>Paperless</span><b>${esc(c.PaperlessBilling)}</b></div>`,
    offer: `<div class="kv">
      <span>Churn cut by offer</span><b>${(c.offer_uplift * 100).toFixed(1)} pp</b>
      <span>Expected value</span><b style="color:var(${c.offer_expected_value_usd > 0 ? "--green-ink" : "--amber"})">${usd(c.offer_expected_value_usd)}</b>
      <span>Decision</span><b>${c.recommendation === "send_offer" ? "Send offer" : "Don't send"}</b></div>
      <div class="reason">${esc(c.segment_description)}</div><div class="reason">${esc(c.recommendation_reason)}</div>`,
  };
  $("#insight").innerHTML = `
    <div class="sheet-handle"></div>
    <div class="card-head"><h3>Customer Insight</h3><div style="display:flex;gap:8px"><a class="btn sm" href="#whatif" id="toWhatIf">Simulate</a>
      <button class="btn sm sheet-close" id="closeSheet" aria-label="Close">${icon("x")}</button></div></div>
    <div class="cust-head" style="--c:var(${SEG[c.segment].c})">
      <span class="initials">${SEG[c.segment].ini}</span>
      <div><div class="id mono">${esc(c.customerID)}</div><div class="meta">${esc(c.gender)} · ${c.tenure} mo · ${esc(c.Contract)}</div></div>
      ${tag(c.segment)}
    </div>
    <div class="mini">
      <div><div class="label">Churn probability</div><div class="v" style="color:var(${rv})">${Math.round(c.churn_probability * 100)}%</div></div>
      <div><div class="label">Revenue at risk, 12 mo</div><div class="v">${usd(c.revenue_at_risk)}</div></div>
    </div>
    <div class="tabs">${[["factors", "Key factors"], ["profile", "Profile"], ["offer", "Offer"]].map(([k, l]) => `<button class="${state.tab === k ? "active" : ""}" data-tab="${k}">${l}</button>`).join("")}</div>
    <div class="tab-body">${tabs[state.tab]}</div>
    <div class="actions">
      <div class="actions-head"><h4>${icon("spark")}Recommended actions</h4>
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
  if (state.detail && renderWhatIf.loadedFrom !== state.detail.customerID && renderWhatIf.fromInsight) {
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
function predictWhatIf() {
  clearTimeout(predictWhatIf._t);
  predictWhatIf._t = setTimeout(async () => {
    const body = { ...whatIf, TotalCharges: whatIf.tenure * whatIf.MonthlyCharges };
    const r = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const rv = riskVar(r.churn_probability), send = r.recommendation === "send_offer";
    const maxImpact = Math.max(0.01, ...r.top_risk_factors.map((f) => f.impact));
    $("#whatifSticky").innerHTML = `<b style="color:var(${rv})">${Math.round(r.churn_probability * 100)}%</b>
      <span>${r.risk_level} risk</span><span class="sep"></span>
      <span class="${send ? "ok" : "no"}">${send ? "✓ Send offer" : "✕ No offer"}</span><span class="more">Details ↓</span>`;
    $("#whatifResult").innerHTML = `
      <h3>Prediction</h3>
      <div class="result-gauge"><div class="tick-gauge" id="riskGauge"></div>
        <div class="gauge-mid"><b style="color:var(${rv})">${Math.round(r.churn_probability * 100)}%</b><span>${r.risk_level} churn risk</span></div></div>
      <div class="verdict ${send ? "good" : "bad"}"><b>${send ? "✓ Send the retention offer." : "✕ Don't send the offer."}</b> ${esc(r.recommendation_reason)}</div>
      <div class="mini">
        <div><div class="label">Offer cuts churn by</div><div class="v">${(r.offer_uplift * 100).toFixed(1)} pp</div></div>
        <div><div class="label">Expected offer value</div><div class="v">${usd(r.offer_expected_value_usd)}</div></div>
      </div>
      <h4 style="margin:4px 0 12px">Why? <span class="hint">${state.model === "own" ? "occlusion impact" : "SHAP impact"}</span></h4>
      ${r.top_risk_factors.length ? r.top_risk_factors.map((f) => `<div class="frow"><span>${esc(f.feature)} = ${esc(f.value)}</span>${meter(f.impact / maxImpact, rv)}<span class="num">+${f.impact.toFixed(2)}</span></div>`).join("") : `<div class="hint">Nothing is pushing this customer toward churn.</div>`}`;
    tickGauge($("#riskGauge"), [{ share: Math.max(r.churn_probability, 0.001), c: rv }, { share: 1 - r.churn_probability, c: "--track" }]);
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
      <div style="display:flex;gap:8px"><button class="btn" id="clearPlan">Clear</button><button class="btn primary" id="dlPlan">${icon("download")}Download CSV</button></div></div>
    <div class="table-wrap"><table><thead><tr><th class="nosort">Customer ID</th><th class="nosort">Segment</th><th class="nosort">Churn</th><th class="nosort">Offer value</th><th class="nosort">Planned actions</th><th class="nosort"></th></tr></thead>
    <tbody>${ids.map((id) => { const c = camp[id]; return `<tr data-id="${esc(id)}"><td class="mono a-id">${esc(id)}</td><td class="a-seg">${tag(c.segment)}</td>
      <td class="a-prob" style="color:var(${riskVar(c.prob)})">${Math.round(c.prob * 100)}% <span class="ph-label">churn</span></td><td class="a-val">${usd(c.value)} <span class="ph-label">offer value</span></td>
      <td class="a-acts" style="white-space:normal">${c.actions.map(esc).join("<br>")}</td><td class="a-rm"><button class="btn sm" data-remove="${esc(id)}">Remove</button></td></tr>`; }).join("")}</tbody></table></div>`;
  $$("[data-remove]").forEach((b) => b.addEventListener("click", (e) => {
    e.stopPropagation();
    const c = store.get(); delete c[b.dataset.remove]; store.set(c); updateCampaignBadge(); renderActions();
  }));
  $$("#actionsBody tbody tr").forEach((tr) => tr.addEventListener("click", () => openCustomer(tr.dataset.id)));
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

/* ------------------------------------------------------------------ model view (Chart.js, redrawn on theme change) */
const SHORT_POLICY = {
  "Offer to everyone": "Everyone",
  "Top 20% by churn risk (traditional)": "Risk, top 20%",
  "Top 20% by predicted uplift": "Uplift, top 20%",
  "Churn risk, same # of offers as uplift policy": "Risk, same budget",
  "Uplift model: offer if expected value > 0": "Uplift, value > 0",
  "Own model: top 20% by predicted uplift": "Own model, top 20%",
  "Own model: offer if expected value > 0": "Own model, value > 0",
  "Oracle (true effect, upper bound)": "Oracle (max)",
};
function renderModel() {
  const m = state.summary?.model;
  if (!m) return;
  if (renderModel.done && !renderModel.stale) return;
  renderModel.done = true;
  renderModel.stale = false;
  const u = m.uplift_vs_risk;
  const own = state.model === "own";
  const cm = m.churn_models.find((x) => x.own === own);        // the model set currently applied
  const other = m.churn_models.find((x) => x.own !== own && x.lift);
  const top20 = own ? m.own_vs_risk.own_top20 : u.uplift;
  $("#metrics").innerHTML = [
    ["ROC-AUC", cm.roc_auc.toFixed(3), `${other.name.split(" (")[0]} ${other.roc_auc.toFixed(3)}`],
    ["PR-AUC", cm.pr_auc.toFixed(3), `F1 ${cm.f1.toFixed(3)} · recall ${pct(cm.recall, 0)}`],
    ["Top-decile lift", `${cm.lift.toFixed(1)}x`, `10% riskiest hold ${Math.round(cm.lift * 10)}% of churners`],
    ["Uplift vs risk targeting", `${(top20 / u.risk).toFixed(1)}x`, `${usd(top20)} vs ${usd(u.risk)}, same budget`],
  ].map(([l, v, s]) => `<div class="card metric"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");
  $("#appliedNote").innerHTML = `Applied in the app right now: <b>${esc(state.summary.served.name)}</b>. Switch models with the control in the header.`;

  const best = (rows, key, lower) => rows.reduce((a, r) => (r[key] == null ? a : a == null || (lower ? r[key] < a : r[key] > a) ? r[key] : a), null);
  const cell = (rows, r, key, d, lower) => `<td class="${r[key] === best(rows, key, lower) ? "best" : ""}">${r[key] == null ? "–" : r[key].toFixed(d)}</td>`;
  $("#churnTable").innerHTML = `<thead><tr><th class="nosort">Model</th><th class="nosort">ROC-AUC</th><th class="nosort">PR-AUC</th><th class="nosort">F1</th><th class="nosort">Recall</th><th class="nosort">Brier ↓</th></tr></thead><tbody>${
    m.churn_models.map((r) => `<tr class="${r.own ? "own-row" : ""}"><td>${esc(r.name)}${(own ? r.own : r === other) ? ' <span class="chip">applied</span>' : ""}</td>${cell(m.churn_models, r, "roc_auc", 3)}${cell(m.churn_models, r, "pr_auc", 3)}${cell(m.churn_models, r, "f1", 3)}${cell(m.churn_models, r, "recall", 3)}${cell(m.churn_models, r, "brier", 3, true)}</tr>`).join("")}</tbody>`;
  const sc0 = Object.keys(m.uplift_learners);
  const learners = Object.keys(m.uplift_learners[sc0[0]]);
  const urows = learners.map((name) => ({ name, ...Object.fromEntries(sc0.flatMap((scn) => [[`q_${scn}`, m.uplift_learners[scn][name].qini_coefficient], [`c_${scn}`, m.uplift_learners[scn][name].corr_with_true_uplift]])) }));
  $("#upliftTable").innerHTML = `<thead><tr><th class="nosort">Uplift model</th>${sc0.map((scn) => `<th class="nosort">Qini, ${scn.replace("_", "-")}</th><th class="nosort">Corr. with true effect</th>`).join("")}</tr></thead><tbody>${
    urows.map((r) => `<tr class="${r.name === "ChurnUpliftNet" ? "own-row" : ""}"><td>${esc(r.name)}${r.name === "ChurnUpliftNet" ? " (own, NumPy)" : ""}</td>${sc0.map((scn) => cell(urows, r, `q_${scn}`, 4) + cell(urows, r, `c_${scn}`, 2)).join("")}</tr>`).join("")}</tbody>`;

  const C = { text: css("--text"), muted: css("--muted"), line: css("--line"), green: css("--green"), bright: css("--green-bright"), red: css("--red"), dim: css("--dim"), card: css("--card") };
  Chart.defaults.color = C.muted;
  Chart.defaults.borderColor = C.line;
  Chart.defaults.font.family = "Outfit, system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  ["policy", "scenario", "tenure"].forEach((k) => charts[k]?.destroy());
  const tip = { backgroundColor: C.text, titleColor: C.card, bodyColor: C.card, padding: 10, cornerRadius: 8, displayColors: false };
  const grid = { color: C.line, drawTicks: false }, border = { display: false };

  const pol = m.policies.filter((p) => p.policy !== "No offers");
  charts.policy = new Chart($("#policyChart"), {
    type: "bar",
    data: { labels: pol.map((p) => SHORT_POLICY[p.policy] || p.policy),
      datasets: [{ data: pol.map((p) => p.net_value_usd), borderRadius: 7, borderSkipped: false, maxBarThickness: 26,
        backgroundColor: pol.map((p) => (p.net_value_usd < 0 ? C.red : p.policy.startsWith("Own") ? C.green : /uplift/i.test(p.policy) ? C.bright + "99" : p.policy.startsWith("Oracle") ? C.text : C.dim)) }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { ...tip, callbacks: { label: (c) => `Net value ${usd(c.parsed.x)}` } } },
      scales: { x: { ticks: { callback: (v) => usd(v) }, grid, border }, y: { grid: { display: false }, border } } },
  });

  const sc = m.scenarios, keys = Object.keys(sc);
  const series = [["Risk, top 20%", "churn_risk_top20_net_value", C.dim], ["Risk, same budget", "churn_risk_same_budget_net_value", C.muted],
    ["Library uplift", "uplift_policy_net_value", C.bright + "99"], ["Own model", "own_policy_net_value", C.green], ["Oracle (max)", "oracle_net_value", C.text]];
  charts.scenario = new Chart($("#scenarioChart"), {
    type: "bar",
    data: { labels: keys.map((k) => k.replace("_", "-")),
      datasets: series.map(([label, key, color]) => ({ label, data: keys.map((k) => sc[k][key] / 1000), backgroundColor: color, borderRadius: 6, borderSkipped: false, maxBarThickness: 34 })) },
    options: { maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 10, useBorderRadius: true, borderRadius: 3, padding: 14 } }, tooltip: { ...tip, callbacks: { label: (c) => `${c.dataset.label}: $${c.parsed.y.toFixed(1)}k` } } },
      scales: { y: { ticks: { callback: (v) => `$${v}k` }, grid, border }, x: { grid: { display: false }, border } } },
  });

  const t = state.summary.churn_by_tenure;
  const ctx = $("#tenureLine").getContext("2d");
  const fill = ctx.createLinearGradient(0, 0, 0, 270);
  fill.addColorStop(0, C.bright + "55");
  fill.addColorStop(1, C.bright + "00");
  charts.tenure = new Chart(ctx, {
    type: "line",
    data: { labels: t.labels.map((l) => `${l} mo`), datasets: [
      { label: "Actual churn", data: t.actual, borderColor: C.green, backgroundColor: fill, fill: true, tension: 0.35, pointRadius: 3, pointBackgroundColor: C.green, borderWidth: 2 },
      { label: "Predicted churn", data: t.predicted, borderColor: C.text, borderDash: [5, 4], tension: 0.35, pointRadius: 0, borderWidth: 1.5 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 14, boxHeight: 2, padding: 14 } }, tooltip: { ...tip, callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.parsed.y)}` } } },
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => `${Math.round(v * 100)}%` }, grid, border }, x: { grid: { display: false }, border } } },
  });
}

/* ------------------------------------------------------------------ init */
async function init() {
  updateCampaignBadge();
  renderFilters();
  paintModelSwitch();
  $$("#modelSwitch button").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
  $("#themeDark").addEventListener("click", () => setTheme("dark"));
  $("#themeLight").addEventListener("click", () => setTheme("light"));
  $$("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
    if (state.sort === th.dataset.sort) state.desc = !state.desc; else { state.sort = th.dataset.sort; state.desc = true; }
    state.page = 1;
    loadTable();
  }));
  let t;
  $("#tableSearch").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { state.q = e.target.value.trim(); state.page = 1; state.selected = null; loadTable(); }, 250); });
  $("#globalSearch").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    state.q = e.target.value.trim(); state.page = 1; state.selected = null; $("#tableSearch").value = state.q;
    go("customers");
  });
  $("#whatifSticky").addEventListener("click", () => $("#whatifResult").scrollIntoView({ behavior: "smooth" }));
  $("#sheetBackdrop").addEventListener("click", () => openSheet(false));
  document.addEventListener("click", (e) => { if (e.target.closest("#toWhatIf")) renderWhatIf.fromInsight = true; });
  $$("#quick button").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
  $("#askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("#askInput").value.trim();
    if (!text) return;
    $("#askInput").value = "";
    ask(null, text);
  });

  try {
    state.summary = await api("/api/summary");
  } catch (err) {
    $("#loading").innerHTML = `<div>Could not load data (${esc(err.message)}). Is the API running?</div>`;
    return;
  }
  // Show the overview before drawing so sizes are measured correctly.
  $("#view-overview").classList.add("active");
  renderOverview(state.summary);
  renderCandidates();
  addMsg("bot", "Pick a quick action, or search a customer ID such as 7590-VHVEG.");
  window.addEventListener("hashchange", route);
  route();
  $("#loading").classList.add("hide");
  setTimeout(() => $("#loading").remove(), 400); // drop the overlay once it has faded out
}
init();
