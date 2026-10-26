// InferSim dashboard: precomputed results load instantly; the Python package then
// boots in a Pyodide worker so every control recomputes live.

const PYODIDE = new URLSearchParams(location.search).get("pyodide") ||
  "https://cdn.jsdelivr.net/npm/pyodide@314.0.7/";
const COLORS = {};
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let R = null;          // results.json
let worker = null;
let ready = false;
let seq = 0;
const pending = new Map();

// ---------- formatting ----------
function si(x, digits = 3) {
  if (x === 0) return "0";
  if (!isFinite(x)) return "inf";
  const units = [[1e15, "P"], [1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]];
  for (const [v, u] of units) if (Math.abs(x) >= v) return +(x / v).toPrecision(digits) + u;
  if (Math.abs(x) < 0.01) return x.toExponential(1);
  return String(+x.toPrecision(digits));
}
const ms = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)) + " ms";
const pct = (x) => (100 * x).toFixed(0) + "%";
const int = (x) => Math.round(x).toLocaleString();

// ---------- tooltip ----------
const tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
function showTip(e, text) {
  tip.textContent = text;
  tip.style.display = "block";
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => (tip.style.display = "none");

// ---------- tiny SVG plotting ----------
function scale(log, d0, d1, r0, r1) {
  if (log) {
    const a = Math.log10(d0), b = Math.log10(d1);
    return (v) => r0 + ((Math.log10(Math.max(v, d0 * 1e-3)) - a) / (b - a)) * (r1 - r0);
  }
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}
function ticks(log, d0, d1) {
  if (log) {
    const out = [];
    const lo = Math.floor(Math.log10(d0)), hi = Math.ceil(Math.log10(d1));
    const many = hi - lo <= 2;
    for (let e = lo; e <= hi; e++) for (const m of many ? [1, 2, 5] : [1]) {
      const v = m * 10 ** e;
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
    return out;
  }
  const step = 10 ** Math.floor(Math.log10((d1 - d0) / 4 || 1));
  const n = Math.ceil((d1 - d0) / step);
  const k = n > 10 ? (n > 25 ? 5 : 2) : 1;
  const out = [];
  for (let v = Math.ceil(d0 / (step * k)) * step * k; v <= d1 + 1e-9; v += step * k) out.push(+v.toPrecision(6));
  return out;
}
function domain(vals, log, pad = 0.08) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (log) {
    lo = Math.max(lo, 1e-12);
    const a = Math.log10(lo), b = Math.log10(hi), p = Math.max((b - a) * pad, 0.15);
    return [10 ** (a - p), 10 ** (b + p)];
  }
  const p = (hi - lo || Math.abs(hi) || 1) * pad;
  return [Math.min(0, lo - p) === 0 && lo >= 0 ? 0 : lo - p, hi + p];
}
function plot(el, o) {
  const W = el.clientWidth || 600, H = el.clientHeight || 360;
  const m = { l: 62, r: 16, t: 12, b: 42 };
  const xs = [], ys = [];
  for (const d of o.dots || []) { xs.push(d.x); ys.push(d.y); }
  for (const l of o.lines || []) for (const [x, y] of l.pts) { xs.push(x); ys.push(y); }
  const [x0, x1] = o.xDomain || domain(xs, o.xLog);
  const [y0, y1] = o.yDomain || domain(ys, o.yLog);
  const X = scale(o.xLog, x0, x1, m.l, W - m.r), Y = scale(o.yLog, y0, y1, H - m.b, m.t);
  const g = [];
  g.push(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "chart")}">`);
  for (const t of ticks(o.yLog, y0, y1)) {
    g.push(`<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`);
    g.push(`<text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end">${si(t)}</text>`);
  }
  for (const t of ticks(o.xLog, x0, x1)) {
    g.push(`<line class="gridline" y1="${m.t}" y2="${H - m.b}" x1="${X(t)}" x2="${X(t)}"/>`);
    g.push(`<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle">${si(t)}</text>`);
  }
  g.push(`<g class="axis"><line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/></g>`);
  g.push(`<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel || "")}</text>`);
  g.push(`<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel || "")}</text>`);
  g.push(`<clipPath id="c-${el.id}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></clipPath><g clip-path="url(#c-${el.id})">`);
  for (const l of o.lines || []) {
    const d = l.pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
    g.push(`<path d="${d}" fill="none" stroke="${l.color}" stroke-width="${l.width || 2}" ${l.dash ? `stroke-dasharray="${l.dash}"` : ""} opacity="${l.opacity ?? 1}"/>`);
  }
  (o.dots || []).forEach((d, i) => {
    g.push(`<circle data-i="${i}" cx="${X(d.x).toFixed(1)}" cy="${Y(d.y).toFixed(1)}" r="${d.r || 4}" fill="${d.fill ?? d.color}" stroke="${d.stroke || d.color}" stroke-width="${d.sw || 1}" opacity="${d.opacity ?? 0.9}"/>`);
  });
  g.push(`</g>`);
  for (const t of o.labels || []) {
    g.push(`<text class="lab" x="${X(t.x) + (t.dx || 0)}" y="${Y(t.y) + (t.dy || 0)}" text-anchor="${t.anchor || "start"}">${esc(t.text)}</text>`);
  }
  g.push(`</svg>`);
  el.innerHTML = g.join("");
  el.querySelectorAll("circle").forEach((c) => {
    const d = o.dots[+c.dataset.i];
    if (!d.tip) return;
    c.addEventListener("mousemove", (e) => showTip(e, d.tip));
    c.addEventListener("mouseleave", hideTip);
  });
}
const legend = (el, items) => (el.innerHTML = items.map(([n, c]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join(""));

// ---------- worker ----------
function run(fn, kwargs) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, fn, kwargs });
  });
}
function bootEngine() {
  const pill = $("#engine");
  try {
    worker = new Worker("worker.js", { type: "module" });
  } catch (e) {
    pill.textContent = "Live engine unavailable";
    pill.className = "pill pill-err";
    return;
  }
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    e.data.ok ? p.resolve(e.data) : p.reject(new Error(e.data.error));
  };
  const id = ++seq;
  pending.set(id, {
    resolve: (d) => {
      ready = true;
      pill.textContent = `Live: Python ${d.result.python} in your browser`;
      pill.className = "pill pill-ok";
      document.querySelectorAll("#step-run, #serve-run").forEach((b) => (b.disabled = false));
      $("#step-status").textContent = "";
    },
    reject: (err) => {
      pill.textContent = "Live engine failed to load";
      pill.className = "pill pill-err";
      $("#step-status").textContent = "Showing precomputed results. " + err.message;
    },
  });
  worker.postMessage({ id, type: "boot", indexURL: PYODIDE });
}

// ---------- findings ----------
function renderFindings() {
  $("#finding-cards").innerHTML = R.findings.map((f) => `<div class="card"><strong>${esc(f.title)}</strong><p>${esc(f.body)}</p></div>`).join("");
}

// ---------- step explorer ----------
function fillSelects() {
  const models = Object.keys(R.catalog.models);
  const hws = Object.keys(R.catalog.hardware);
  document.querySelectorAll("select[name=model]").forEach((s) => {
    s.innerHTML = models.map((m) => `<option value="${m}">${m} (${R.catalog.models[m].params_b.toFixed(m.includes("0.5") ? 2 : 0)}B)</option>`).join("");
    s.value = "llama-3.1-8b";
  });
  document.querySelectorAll("select[name=hw]").forEach((s) => {
    s.innerHTML = hws.map((h) => `<option value="${h}">${esc(R.catalog.hardware[h].name)}</option>`).join("");
    s.value = "h100";
  });
}
function formArgs(form) {
  const f = new FormData(form);
  const o = {};
  for (const [k, v] of f.entries()) o[k] = v;
  for (const k of ["tp", "pp", "batch", "tokens", "requests", "prompt", "output"]) if (k in o) o[k] = parseInt(o[k], 10);
  if ("rate" in o) o.rate = parseFloat(o.rate);
  if (form.id === "step-form") o.ep = !!f.get("ep");
  return o;
}
function kpi(v, k) { return `<div class="kpi"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`; }
function stackBar(title, shares, colors) {
  const entries = Object.entries(shares).sort((a, b) => b[1] - a[1]);
  const parts = entries.map(([k, v]) => `<span title="${esc(k)} ${pct(v)}" style="width:${(100 * v).toFixed(2)}%;background:${colors[k] || css("--other")}"></span>`).join("");
  const lbl = entries.map(([k, v]) => `<span><i style="background:${colors[k] || css("--other")}"></i>${esc(k)} ${pct(v)}</span>`).join("");
  return `<div class="bar"><div class="lbl"><span>${esc(title)}</span></div><div class="stack">${parts}</div><div class="legend" style="margin-top:6px">${lbl}</div></div>`;
}
function renderStep(v, args) {
  lastStep = [v, args];
  const k = $("#step-kpis");
  if (!v.fits) {
    k.innerHTML = kpi("does not fit", v.label) + kpi(si(v.weight_gb_per_device) + " GB", "weights per device");
    $("#roof-chart").innerHTML = `<p class="note">The weights do not fit in this layout's memory. Raise TP/PP or lower precision.</p>`;
    $("#step-bars").innerHTML = "";
    $("#step-top").innerHTML = "";
    return;
  }
  const decode = (args?.phase || "decode") === "decode";
  k.innerHTML = [
    kpi(ms(v.latency_ms), "step latency"),
    decode ? kpi(int(v.tok_s_user), "tokens/s per user") : kpi(si(v.prefill_tok_s), "prefill tokens/s"),
    decode ? kpi(si(v.tok_s_total), "tokens/s total") : kpi(v.devices, "devices"),
    kpi(v.dominant_bound, "dominant bound"),
    kpi(si(v.weight_gb_per_device) + " GB", "weights / device"),
    kpi(si(v.kv_capacity_tokens), "KV capacity (tokens)"),
  ].join("");
  const rf = v.roofline;
  const cats = { attention: css("--attention"), ffn: css("--ffn"), other: css("--other"), comm: css("--comm") };
  const total = v.latency_ms;
  const xsAll = rf.points.map((p) => p.intensity);
  const x0 = Math.min(0.5, ...xsAll) / 2, x1 = Math.max(rf.ridge * 8, ...xsAll) * 2;
  const roofX = [x0, rf.ridge, x1];
  const lines = [{ pts: roofX.map((x) => [x, Math.min(rf.peak, x * rf.mem_bw)]), color: css("--ink"), width: 2 }];
  if (rf.peak_attention !== rf.peak) lines.push({ pts: [[rf.peak_attention / rf.mem_bw, rf.peak_attention], [x1, rf.peak_attention]], color: css("--muted"), dash: "4 4", width: 1.5 });
  const dots = rf.points.map((p) => ({
    x: p.intensity, y: Math.max(p.achieved_flops, 1), r: 3 + 18 * Math.sqrt(p.time_ms / total),
    color: cats[p.category], fill: cats[p.category], opacity: 0.75,
    tip: `${p.name} (${p.category})\n${p.bound}-bound\nintensity ${si(p.intensity)} FLOP/B\nachieved ${si(p.achieved_flops)}FLOP/s\nroof ${si(p.attainable_flops)}FLOP/s\n${p.time_ms.toFixed(3)} ms (${pct(p.time_ms / total)} of step)`,
  }));
  const ys = dots.map((d) => d.y);
  const top = [...rf.points].sort((a, b) => b.time_ms - a.time_ms).slice(0, 3);
  const opLabels = top.map((p, i) => ({ x: p.intensity, y: Math.max(p.achieved_flops, 1), text: p.name, dx: 10, dy: 14 + 13 * i }));
  plot($("#roof-chart"), {
    title: "roofline", xLog: true, yLog: true, xLabel: "arithmetic intensity (FLOP/byte)", yLabel: "achieved FLOP/s",
    xDomain: [x0, x1], yDomain: [Math.min(...ys, rf.peak / 1e4) / 2, rf.peak * 2.5], lines, dots,
    labels: [
      { x: rf.ridge, y: rf.peak, text: `peak ${si(rf.peak)}FLOP/s ${rf.dtype}`, dy: -8, dx: 6 },
      { x: x0 * 1.6, y: x0 * 1.6 * rf.mem_bw, text: `${si(rf.mem_bw)}B/s`, dy: -8 },
      ...opLabels,
    ],
  });
  legend($("#roof-legend"), [["attention", cats.attention], ["FFN", cats.ffn], ["other", cats.other]]);
  const bounds = { memory: css("--attention"), compute: css("--ffn"), comm: css("--comm"), overhead: css("--other") };
  $("#step-bars").innerHTML = stackBar("by operator category", v.category_share, cats) + stackBar("by roofline bound", v.bound_share, bounds) +
    (rf.comm.length ? `<p class="note">Collectives: ${rf.comm.map((c) => `${esc(c.name)} ${c.time_ms.toFixed(2)} ms`).join(", ")}</p>` : "");
  $("#step-top").innerHTML = `<tr><th>op</th><th>bound</th><th class="num">ms</th><th class="num">share</th></tr>` +
    v.top_ops.map((o) => `<tr><td class="mono">${esc(o.name)}</td><td>${esc(o.bound)}</td><td class="num">${o.ms.toFixed(3)}</td><td class="num">${pct(o.ms / total)}</td></tr>`).join("");
}
async function runStep(e) {
  e && e.preventDefault();
  const args = formArgs($("#step-form"));
  $("#tok-label").textContent = args.phase === "prefill" ? "Prompt" : "Context";
  if (!ready) return;
  $("#step-status").textContent = "Computing...";
  try {
    const r = await run("step_view", args);
    renderStep(r.result, args);
    $("#step-status").textContent = `Computed in ${r.ms.toFixed(0)} ms, in your browser.`;
  } catch (err) {
    $("#step-status").textContent = err.message;
  }
}

// ---------- design space ----------
const FAM = () => ({ gpu: css("--gpu"), lpu: css("--lpu"), disaggregated: css("--disaggregated"), afd: css("--afd") });
const FAM_NAME = { gpu: "GPU", lpu: "LPU", disaggregated: "GPU prefill + LPU decode", afd: "attention GPU / FFN LPU" };
const SLAS = [20, 50, 100, 200, 500];
let dseModel = "llama-3.1-70b";
let lastStep = null;
function renderDSE() {
  const d = R.dse[dseModel];
  const fam = FAM();
  const sla = SLAS[+$("#sla").value];
  $("#sla-val").textContent = sla;
  const best = d.best_under_sla[String(sla)];
  const tipOf = (p) => `${p.label}\nbatch ${p.batch}, ${p.users} users\nTPOT ${p.tpot_ms.toFixed(2)} ms -> ${p.tok_s_user.toFixed(0)} tok/s/user\n${p.tok_s_device.toFixed(0)} tok/s per device\nTTFT ${p.ttft_ms.toFixed(0)} ms\n${p.dominant_bound}-bound (${p.dominant_category})`;
  const dots = d.points.map((p) => ({ x: p.tok_s_user, y: p.tok_s_device, r: 3, color: fam[p.family], fill: fam[p.family], opacity: 0.25, tip: tipOf(p) }));
  const lines = Object.entries(d.pareto_by_family).map(([f, pts]) => ({ pts: pts.map((p) => [p.tok_s_user, p.tok_s_device]), color: fam[f], width: 2 }));
  for (const [f, pts] of Object.entries(d.pareto_by_family)) for (const p of pts) dots.push({ x: p.tok_s_user, y: p.tok_s_device, r: 4.5, color: fam[f], fill: fam[f], opacity: 1, tip: tipOf(p) });
  if (best) dots.push({ x: best.tok_s_user, y: best.tok_s_device, r: 10, color: css("--ink"), fill: "none", sw: 2, opacity: 1, tip: "Best at this SLA\n" + tipOf(best) });
  const xs = d.points.map((p) => p.tok_s_user), ys = d.points.map((p) => p.tok_s_device);
  const xDom = domain(xs.concat([sla]), true), yDom = domain(ys, true);
  lines.push({ pts: [[sla, yDom[0]], [sla, yDom[1]]], color: css("--muted"), dash: "3 4", width: 1 });
  plot($("#dse-chart"), { title: "pareto", xLog: true, yLog: true, xLabel: "tokens/s per user (interactivity)", yLabel: "tokens/s per device", xDomain: xDom, yDomain: yDom, dots, lines });
  legend($("#dse-legend"), Object.keys(d.pareto_by_family).map((f) => [FAM_NAME[f], fam[f]]));
  $("#sla-best").innerHTML = best
    ? `Best at &ge; ${sla} tok/s/user: <strong>${esc(best.label)}</strong>, batch ${best.batch}, ${int(best.tok_s_device)} tok/s per device (${best.tok_s_user.toFixed(0)} tok/s/user, ${best.dominant_bound}-bound).`
    : `No swept layout reaches ${sla} tok/s per user for this model.`;
  const rows = [...d.pareto].sort((a, b) => b.tok_s_device - a.tok_s_device);
  $("#dse-table").innerHTML = `<tr><th>layout</th><th>type</th><th class="num">batch</th><th class="num">users</th><th class="num">TPOT</th><th class="num">tok/s/user</th><th class="num">tok/s/device</th><th>bound</th></tr>` +
    rows.map((p) => `<tr class="${best && p.label === best.label && p.batch === best.batch ? "hl" : ""}"><td>${esc(p.label)}</td><td><span class="tag" style="color:${fam[p.family]}">${p.family}</span></td><td class="num">${p.batch}</td><td class="num">${p.users}</td><td class="num">${p.tpot_ms.toFixed(2)}</td><td class="num">${p.tok_s_user.toFixed(0)}</td><td class="num">${int(p.tok_s_device)}</td><td>${p.dominant_bound}</td></tr>`).join("");
}
function initDSE() {
  const seg = $("#dse-models");
  seg.innerHTML = Object.keys(R.dse).map((m) => `<button type="button" data-m="${m}" class="${m === dseModel ? "on" : ""}">${m}</button>`).join("");
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    dseModel = b.dataset.m;
    seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    renderDSE();
  });
  $("#sla").addEventListener("input", renderDSE);
  renderDSE();
}

// ---------- serving ----------
function renderServeCurve() {
  const c = R.serving["llama-3.1-8b_h100"];
  const col = css("--attention"), col2 = css("--comm");
  plot($("#serve-curve"), {
    title: "load", xLog: false, yLog: false, xLabel: "throughput (output tokens/s)", yLabel: "latency (ms)",
    lines: [
      { pts: c.map((p) => [p.throughput_tok_s, p.tpot_p50_ms]), color: col },
      { pts: c.map((p) => [p.throughput_tok_s, p.ttft_p50_ms]), color: col2, dash: "5 4" },
    ],
    dots: c.flatMap((p) => [
      { x: p.throughput_tok_s, y: p.tpot_p50_ms, r: 4, color: col, tip: `${p.rate} req/s\nTPOT p50 ${p.tpot_p50_ms.toFixed(1)} ms, p99 ${p.tpot_p99_ms.toFixed(1)} ms\n${int(p.throughput_tok_s)} tok/s` },
      { x: p.throughput_tok_s, y: p.ttft_p50_ms, r: 4, color: col2, tip: `${p.rate} req/s\nTTFT p50 ${p.ttft_p50_ms.toFixed(0)} ms, p99 ${p.ttft_p99_ms.toFixed(0)} ms` },
    ]),
    labels: [{ x: c[c.length - 1].throughput_tok_s, y: c[c.length - 1].tpot_p50_ms, text: "TPOT p50", anchor: "end", dy: -10 },
      { x: c[c.length - 1].throughput_tok_s, y: c[c.length - 1].ttft_p50_ms, text: "TTFT p50", anchor: "end", dy: -10 }],
  });
}
function histogram(el, vals, color, label) {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const n = 30, w = (hi - lo) / n || 1;
  const bins = new Array(n).fill(0);
  for (const v of vals) bins[Math.min(n - 1, Math.floor((v - lo) / w))]++;
  const pts = [];
  bins.forEach((b, i) => { pts.push([lo + i * w, b], [lo + (i + 1) * w, b]); });
  plot(el, { title: "hist", xLabel: label, yLabel: "requests", lines: [{ pts, color, width: 2 }], dots: [], yDomain: [0, Math.max(...bins) * 1.1] });
}
async function runServe(e) {
  e.preventDefault();
  if (!ready) return;
  const args = formArgs($("#serve-form"));
  $("#serve-status").textContent = "Simulating...";
  try {
    const r = await run("serve_view", args);
    const s = r.result;
    $("#serve-kpis").innerHTML = [
      kpi(int(s.throughput_tok_s), "output tokens/s"),
      kpi(ms(s.ttft_p50_ms), "TTFT p50"), kpi(ms(s.ttft_p99_ms), "TTFT p99"),
      kpi(ms(s.tpot_p50_ms), "TPOT p50"), kpi(ms(s.tpot_p99_ms), "TPOT p99"),
      kpi(s.mean_batch_tokens.toFixed(0), "mean tokens per step"),
    ].join("");
    $("#serve-hist-panel").hidden = false;
    histogram($("#serve-hist"), s.ttft_ms, css("--comm"), "TTFT (ms)");
    $("#serve-status").textContent = `${esc(s.label)}: ${s.requests} requests, ${s.steps} steps, simulated in ${r.ms.toFixed(0)} ms.`;
  } catch (err) {
    $("#serve-status").textContent = err.message;
  }
}

// ---------- method ----------
function renderHW() {
  const rows = Object.entries(R.catalog.hardware).map(([k, h]) => {
    const pk = Object.entries(h.peak_flops).map(([d, v]) => `${d} ${si(v)}`).join(", ");
    return `<tr><td><strong>${esc(h.name)}</strong><br><span class="mono">${k}</span></td><td>${h.kind}</td><td class="num">${pk}</td><td class="num">${si(h.mem_bw)}B/s</td><td class="num">${si(h.mem_capacity)}B</td><td class="num">${si(h.link_bw)}B/s</td><td>${h.sources.map(esc).join("<br>")}${h.assumptions.length ? `<br><em>Assumed:</em> ${h.assumptions.map(esc).join("; ")}` : ""}</td></tr>`;
  }).join("");
  $("#hw-table").innerHTML = `<tr><th>device</th><th>kind</th><th class="num">peak FLOP/s (dense)</th><th class="num">memory BW</th><th class="num">capacity</th><th class="num">link / dir</th><th>sources and assumptions</th></tr>` + rows;
}

// ---------- nav ----------
function initNav() {
  const links = [...document.querySelectorAll("#nav a")];
  const obs = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) links.forEach((l) => l.classList.toggle("on", l.getAttribute("href") === "#" + e.target.id));
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((s) => obs.observe(s));
}

async function main() {
  R = await (await fetch("data/results.json")).json();
  $("#ver").textContent = "v" + R.version;
  fillSelects();
  renderFindings();
  renderStep(R.steps["8b_h100_decode_b1"], { phase: "decode" });
  $("#step-status").textContent = "Precomputed view. The live engine is loading.";
  initDSE();
  renderServeCurve();
  renderHW();
  initNav();
  $("#step-form").addEventListener("submit", runStep);
  $("#step-form").addEventListener("change", runStep);
  $("#serve-form").addEventListener("submit", runServe);
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { renderDSE(); renderServeCurve(); if (lastStep) renderStep(...lastStep); }, 200); });
  bootEngine();
}
main();
