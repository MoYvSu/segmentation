"use strict";

const $ = (id) => document.getElementById(id);
const palette = ["#56cee9", "#e2b86b", "#af9cf3", "#46c49a", "#ee8ca6"];
const storagePrefix = "aicomp-watch-v1:";
const kindNames = {baseline: "起点", score: "分数", rank: "排名", new: "上榜", missing: "列表变化", update: "更新"};
let state = null, history = null, activeStage = "", focusId = "", selected = [], watches = new Set();
let showWatched = false, metric = "score", sortKey = "rank", sortDirection = 1;
let requestSeq = 0, inFlight = false, historyKey = "", renderedKey = "", chartHover = null;

function readPref(key, fallback) {
  try { const value = localStorage.getItem(storagePrefix + key); return value === null ? fallback : JSON.parse(value); }
  catch { return fallback; }
}
function savePref(key, value) {
  try { localStorage.setItem(storagePrefix + key, JSON.stringify(value)); } catch { /* 本地存储被禁用时仍可使用。 */ }
}
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = String(text);
  if (className) el.className = className;
  return el;
}
function setText(id, text) { $(id).textContent = text ?? "—"; }
function num(value, digits = 4) {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}
function signed(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  return (Number(value) > 0 ? "+" : "") + num(value, digits);
}
function changeClass(value) { return value === null || value === undefined || Number(value) === 0 ? "unchanged" : Number(value) > 0 ? "up" : "down"; }
function timeText(value, full = false) {
  if (!value) return "—";
  // 官网无时区的时间按中国标准时间解释，浏览器所在时区不改变其含义。
  const date = new Date(value.includes("T") ? value : value.replace(" ", "T") + "+08:00");
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", ...(full ? {second: "2-digit"} : {}), hour12: false}).format(date);
}
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { $("toast").hidden = true; }, 4200);
}
async function api(path, post = false) {
  const response = await fetch(path, post ? {method: "POST", headers: {"X-Leaderboard-Control": "1"}} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}


const FX = globalThis.LeaderboardChart;
let basis = "change", viewport = null, timeBounds = null, rangePreset = 0, followLatest = true;
let autoY = true, manualY = null, plot = null, navPlot = null, chartDrag = null, navDrag = null;
let framePending = false, seriesCache = new Map();
const maxGap = () => Math.max(120000, (state?.collector.interval_seconds || 300) * 2500);
function getSeries(kind, ids = [...new Set([...selected, focusId].filter(Boolean))]) {
  const key = kind + ":" + ids.join("|");
  if (!seriesCache.has(key)) seriesCache.set(key, FX.series(history, ids, kind));
  return seriesCache.get(key);
}
function resetY() { autoY = true; manualY = null; }
function loadStagePreferences() {
  const ids = new Set(state.rows.map((r) => r.team_id));
  const savedWatches = readPref(activeStage + ":watches", []);
  watches = new Set(Array.isArray(savedWatches) ? savedWatches.filter((id) => typeof id === "string") : []);
  const savedSelected = readPref(activeStage + ":selected", null);
  selected = Array.isArray(savedSelected) ? [...new Set(savedSelected.filter((id) => ids.has(id)))].slice(0, 5) : state.rows.slice(0, 3).map((r) => r.team_id);
  focusId = readPref(activeStage + ":focus", "");
  if (!ids.has(focusId)) focusId = state.rows[0]?.team_id || "";
  const savedBasis = readPref(activeStage + ":fx-basis", "change");
  basis = ["change", "absolute", "percent"].includes(savedBasis) ? savedBasis : "change";
  metric = "score"; $("basis-select").value = basis;
  viewport = null; timeBounds = null; rangePreset = 0; followLatest = true; resetY(); seriesCache.clear();
}
function focusTeam(teamId) {
  if (!state.rows.some((r) => r.team_id === teamId)) return;
  focusId = teamId; savePref(activeStage + ":focus", focusId);
  if (!selected.includes(teamId)) {
    if (selected.length >= 5) selected.pop();
    selected.push(teamId); savePref(activeStage + ":selected", selected);
  }
  resetY(); seriesCache.clear();
  renderTable(); renderMarket(); renderQuotes(); renderFocus(); renderLegend(); renderChart();
}
function toggleWatch(teamId) {
  if (!teamId) return;
  watches.has(teamId) ? watches.delete(teamId) : watches.add(teamId);
  savePref(activeStage + ":watches", [...watches]);
  renderTable(); renderMarket(); renderFocus();
}
function renderStages() {
  $("stages").replaceChildren(...(state.source.stages || []).map((stage) => {
    const button = node("button", stage, stage === activeStage ? "active" : "");
    button.setAttribute("aria-current", stage === activeStage ? "page" : "false");
    button.addEventListener("click", () => {
      if (activeStage === stage) return;
      activeStage = stage; history = null; historyKey = ""; renderedKey = ""; plot = null; viewport = null;
      seriesCache.clear(); $("trend-chart").replaceChildren(); $("rankings").replaceChildren();
      $("chart-empty").hidden = false; $("chart-empty").textContent = "正在读取该阶段…";
      load(true);
    });
    return button;
  }));
}
function renderStatus() {
  if (!state) return;
  const collector = state.collector;
  const successMs = state.last_success_at ? Date.parse(state.last_success_at) : 0;
  const stale = successMs && Date.now() - successMs > Math.max(collector.interval_seconds * 2500, 180000);
  const failed = state.last_check && !state.last_check.success;
  let text = "监控中", color = "good";
  if (collector.collecting) text = "正在检查公开榜单";
  else if (failed || collector.error) { text = "采集异常，等待重试"; color = "warn"; }
  else if (stale) { text = "数据已过期，等待检查"; color = "warn"; }
  else if (!state.last_success_at) { text = "等待首次有效数据"; color = "warn"; }
  setText("status-label", text); $("status-dot").className = "dot " + color;
  const next = collector.next_check_at ? Math.max(0, Math.ceil((Date.parse(collector.next_check_at) - Date.now()) / 1000)) : null;
  setText("next-check", collector.collecting ? "采集器在后台运行" : next === null ? "等待采集计划" : next > 0 ? `${Math.floor(next / 60)} 分 ${String(next % 60).padStart(2, "0")} 秒后检查` : "等待采集器响应");
  $("refresh-button").disabled = collector.collecting;
  setText("freshness", `上次有效检查 ${timeText(state.last_success_at, true)} · 每 ${Math.round(collector.interval_seconds / 60)} 分钟检查`);
  const error = failed ? state.last_check.error : collector.error;
  $("error-banner").hidden = !error && !stale;
  setText("error-banner", error ? `${error}。保留上次有效榜单，失败记录不计作新成绩；后续自动重试。` : stale ? "距离上次有效检查已超过正常间隔。这里显示的是上次记录，缺失时段不补写成绩。" : "");
}

function renderState(changedStage) {
  setText("competition-title", state.source.title || "公开竞赛排行榜");
  $("source-link").href = state.source_url;
  renderStages(); renderStatus();
  setText("team-count", state.rows.length);
  const scored = state.rows.filter((r) => r.score !== null);
  setText("scored-count", activeStage + " · " + scored.length + " 队已公布分数");
  setText("leader-score", num(scored[0]?.score)); setText("leader-name", scored[0]?.name || "—");
  setText("leader-gap", num(state.leading_gap)); setText("change-count", state.changes_today);
  setText("observation-count", "已检查 " + state.checks_count + " 次 · 首次记录为基线");
  setText("footer-history", "记录起点 " + timeText(state.started_at, true) + " · 最近变化 " + timeText(state.last_changed_at, true));
  $("export-link").href = "/api/export.json?stage=" + encodeURIComponent(activeStage) + "&hours=0";
  $("insights").replaceChildren(...state.insights.map((text) => node("li", text)));
  const key = activeStage + ":" + state.snapshot_id;
  if (key !== renderedKey || changedStage) {
    renderedKey = key;
    const ids = new Set(state.rows.map((r) => r.team_id));
    selected = selected.filter((id) => ids.has(id));
    if (!ids.has(focusId)) focusId = state.rows[0]?.team_id || "";
    seriesCache.clear(); renderQuotes(); renderMarket(); renderTable(); renderFocus(); renderLegend(); renderEvents();
  }
}
function renderQuotes() {
  $("quote-strip").replaceChildren(...state.rows.slice(0, 6).map((r) => {
    const tile = node("button", null, "quote-tile" + (r.team_id === focusId ? " focused" : ""));
    const name = node("span", null, "quote-name");
    name.append(node("span", r.name), node("small", r.team_id.replace("AIC-2026-", "AIC / ")));
    const number = node("span", null, "quote-number");
    number.append(node("strong", num(r.score)), node("small", signed(r.score_delta), changeClass(r.score_delta)));
    tile.append(node("span", "#" + (r.rank ?? "—"), "tile-rank"), name, number);
    tile.addEventListener("click", () => focusTeam(r.team_id));
    return tile;
  }));
}
function renderMarket() {
  const term = $("watch-search").value.trim().toLocaleLowerCase();
  const rows = state.rows.filter((r) => (!showWatched || watches.has(r.team_id)) && (!term || (r.name + " " + r.team_id).toLocaleLowerCase().includes(term)));
  $("market-list").replaceChildren(...rows.map((r) => {
    const row = node("div", null, "market-row" + (r.team_id === focusId ? " focused" : ""));
    const star = node("button", watches.has(r.team_id) ? "★" : "☆", "market-star" + (watches.has(r.team_id) ? " selected" : ""));
    star.setAttribute("aria-label", (watches.has(r.team_id) ? "取消关注" : "关注") + r.name);
    star.addEventListener("click", () => toggleWatch(r.team_id));
    const team = node("button", null, "market-team");
    const price = node("span", null, "price");
    price.append(node("b", num(r.score)), node("small", signed(r.score_delta), changeClass(r.score_delta)));
    team.append(node("span", r.name, "name"), price);
    team.addEventListener("click", () => focusTeam(r.team_id));
    row.append(star, team, node("span", "#" + (r.rank ?? "—"), "market-rank"));
    return row;
  }));
  if (!rows.length) $("market-list").append(node("p", "没有匹配的队伍", "table-empty"));
  $("watch-filter").setAttribute("aria-pressed", String(showWatched));
  $("watch-filter").textContent = showWatched ? "★" : "☆";
}
function renderTable() {
  if (!state) return;
  const term = $("search").value.trim().toLocaleLowerCase();
  const rows = state.rows.filter((r) => (!showWatched || watches.has(r.team_id)) && (!term || `${r.name} ${r.team_id}`.toLocaleLowerCase().includes(term)));
  rows.sort((a, b) => {
    if (a[sortKey] === null || a[sortKey] === undefined) return b[sortKey] === null || b[sortKey] === undefined ? 0 : 1;
    if (b[sortKey] === null || b[sortKey] === undefined) return -1;
    return (Number(a[sortKey]) - Number(b[sortKey])) * sortDirection || (a.rank ?? 999999) - (b.rank ?? 999999);
  });
  $("rankings").replaceChildren(...rows.map((r) => {
    const tr = node("tr", null, r.team_id === focusId ? "focused" : "");
    const starCell = node("td");
    const star = node("button", watches.has(r.team_id) ? "★" : "☆", "star" + (watches.has(r.team_id) ? " selected" : ""));
    star.setAttribute("aria-label", `${watches.has(r.team_id) ? "取消关注" : "关注"}${r.name}`);
    star.setAttribute("aria-pressed", String(watches.has(r.team_id)));
    star.addEventListener("click", (event) => { event.stopPropagation(); toggleWatch(r.team_id); });
    starCell.append(star); tr.append(starCell, node("td", r.rank ?? "—", r.rank <= 3 && r.rank !== null ? "up" : ""));
    const nameCell = node("td"), nameButton = node("button", r.name, "team-name");
    nameButton.addEventListener("click", (event) => { event.stopPropagation(); focusTeam(r.team_id); });
    nameCell.append(nameButton, node("span", r.team_id, "team-id"));
    tr.append(nameCell, node("td", num(r.score), "numeric"));
    tr.append(node("td", signed(r.score_delta), "numeric " + changeClass(r.score_delta)));
    const rankText = r.rank_delta === null ? "—" : r.rank_delta === 0 ? "持平" : r.rank_delta > 0 ? `↑ ${r.rank_delta}` : `↓ ${-r.rank_delta}`;
    tr.append(node("td", rankText, "numeric " + changeClass(r.rank_delta)), node("td", num(r.gap_leader), "numeric unchanged"), node("td", timeText(r.scored_at), "unchanged"));
    tr.addEventListener("click", () => focusTeam(r.team_id));
    return tr;
  }));
  setText("table-count", `${rows.length} / ${state.rows.length}`);
  $("table-empty").hidden = rows.length > 0;
}

function renderFocus() {
  if (!state) return;
  const row = state.rows.find((r) => r.team_id === focusId);
  $("focus-select").replaceChildren(...state.rows.map((r) => {
    const option = node("option", (r.rank ?? "—") + ". " + r.name); option.value = r.team_id; return option;
  }));
  $("focus-select").value = focusId;
  setText("focus-rank", "# " + (row?.rank ?? "—")); setText("instrument-rank", "# " + (row?.rank ?? "—"));
  setText("focus-score", num(row?.score)); setText("instrument-score", num(row?.score));
  setText("instrument-name", row?.name || "等待队伍数据");
  setText("symbol-code", (row?.team_id || "AIC") + " / " + (activeStage || "SCORE"));
  setText("focus-delta", "较上一版 " + signed(row?.score_delta)); $("focus-delta").className = changeClass(row?.score_delta);
  setText("focus-gap", row?.rank === 1 ? "当前榜首" : num(row?.gap_leader));
  setText("focus-ahead", row?.gap_ahead === null ? "—" : num(row?.gap_ahead));
  setText("focus-time", row ? "公开提交 " + timeText(row.submitted_at, true) + "\n公开打分 " + timeText(row.scored_at, true) : "等待队伍数据");
  $("focus-watch").textContent = watches.has(focusId) ? "★ 已关注 · 点击取消" : "☆ 加入关注";
  $("focus-watch").classList.toggle("selected", watches.has(focusId)); $("focus-watch").disabled = !row;
  const index = state.rows.findIndex((r) => r.team_id === focusId);
  const rivals = state.rows.filter((_, i) => i === index - 1 || i === index + 1);
  $("rivals").replaceChildren(...rivals.map((r) => {
    const line = node("div", null, "rival"), button = node("button", r.name);
    button.addEventListener("click", () => focusTeam(r.team_id));
    line.append(node("span", "#" + (r.rank ?? "—")), button, node("b", num(r.score)));
    return line;
  }));
  if (!rivals.length) $("rivals").append(node("p", "暂无相邻队伍", "rival-empty"));
}
function renderEvents() {
  const events = state.events || []; setText("event-count", events.filter((e) => e.kind !== "baseline").length);
  $("event-list").replaceChildren(...events.map((event) => {
    const line = node("div", null, "event"), timeEl = node("time", timeText(event.at, true));
    timeEl.dateTime = event.at;
    line.append(timeEl, node("span", kindNames[event.kind] || "变化", "event-kind"), node("span", event.title, "event-title"));
    return line;
  }));
  if (!events.length) $("event-list").append(node("p", "还没有有效榜单，采集成功后会建立第一份记录。", "event-empty"));
}

function renderLegend() {
  if (!state) return;
  $("legend").replaceChildren(...selected.map((id, index) => {
    const row = state.rows.find((r) => r.team_id === id), button = node("button"), dot = node("i");
    dot.style.backgroundColor = palette[index];
    button.append(dot, node("span", row?.name || id), node("small", "×"));
    button.title = "移除比较线；点击队名可加入，最多 5 队";
    button.setAttribute("aria-label", "从走势移除" + (row?.name || id));
    button.addEventListener("click", () => {
      selected = selected.filter((value) => value !== id); savePref(activeStage + ":selected", selected);
      seriesCache.clear(); resetY(); renderLegend(); renderChart();
    });
    return button;
  }));
  if (!selected.length) $("legend").append(node("span", "点击左侧队名加入比较", "add-hint"));
}
function svgNode(tag, attrs = {}, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  if (text !== undefined) el.textContent = text;
  return el;
}
function ensureViewport(bounds) {
  if (!viewport) viewport = FX.preset(bounds, rangePreset || 0);
  else if (followLatest) viewport = rangePreset !== null ? FX.preset(bounds, rangePreset)
    : FX.windowBounds(bounds.to - (viewport.to - viewport.from), bounds.to, bounds);
  else viewport = FX.windowBounds(viewport.from, viewport.to, bounds);
}
function durationText(ms) {
  if (ms >= 86400000) return (ms / 86400000).toFixed(1) + " 天";
  if (ms >= 3600000) return (ms / 3600000).toFixed(1) + " 小时";
  return Math.max(1, Math.round(ms / 60000)) + " 分钟";
}
function axisTime(t, span, full = false) {
  return new Intl.DateTimeFormat("zh-CN", {timeZone:"Asia/Shanghai",
    ...(span > 86400000 ? {month:"2-digit",day:"2-digit"} : {}),hour:"2-digit",minute:"2-digit",
    ...(full || span < 180000 ? {second:"2-digit"} : {}), hour12:false}).format(t);
}
function pathFor(points, x, y) {
  let path = "", prev = null;
  for (const p of points) {
    if (p.value === null || !Number.isFinite(p.value)) { prev = null; continue; }
    const px = x(p.t), py = y(p.value);
    path += !prev || p.t - prev.t > maxGap() ? " M" + px + "," + py : " H" + px + " V" + py;
    prev = p;
  }
  return path;
}
function syncChartControls() {
  for (const b of document.querySelectorAll("[data-range]")) b.classList.toggle("active", rangePreset !== null && Number(b.dataset.range) === rangePreset);
  for (const b of document.querySelectorAll("[data-metric]")) b.classList.toggle("active", b.dataset.metric === metric);
  $("auto-scale").classList.toggle("active", autoY); $("auto-scale").setAttribute("aria-pressed", String(autoY));
  $("live-button").classList.toggle("active", followLatest); $("live-button").setAttribute("aria-pressed", String(followLatest));
  $("basis-select").querySelector('[value="percent"]').disabled = metric !== "score";
  $("basis-select").value = basis;
}
function renderChart() {
  const svg = $("trend-chart"); svg.replaceChildren(); $("chart-tooltip").hidden = true; plot = null;
  syncChartControls();
  const observations = history?.observations || [];
  timeBounds = FX.domain(observations.map((o) => Date.parse(o.checked_at)), (state?.collector.interval_seconds || 300) * 1000);
  if (!timeBounds || !state) {
    $("chart-empty").hidden = false; $("chart-empty").classList.add("empty-data");
    $("chart-empty").textContent = "等待有效观测记录"; $("range-chart").replaceChildren(); return;
  }
  ensureViewport(timeBounds);
  const raw = getSeries(metric), transformed = new Map(), rawVisible = new Map();
  for (const id of selected) {
    const visible = FX.clip(raw.get(id) || [], viewport, maxGap());
    rawVisible.set(id, visible); transformed.set(id, FX.relative(visible, basis, metric));
  }
  const values = [...transformed.values()].flatMap((s) => s.points.filter((p) => p.value !== null).map((p) => p.value));
  const currentScale = FX.scale(values, {integer: metric === "rank", floor: metric === "rank" && basis === "absolute" ? 1 : null, manual: autoY ? null : manualY});
  const width = Math.max(270, svg.clientWidth || 900), height = Math.max(200, svg.clientHeight || 400);
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  const left = 14, right = width - (width < 500 ? 75 : 86), top = 23, bottom = height - 29;
  const x = (t) => left + (t - viewport.from) / (viewport.to - viewport.from) * (right - left);
  const invert = metric === "rank" && basis === "absolute";
  const y = (v) => invert ? top + (v - currentScale.min) / (currentScale.max - currentScale.min) * (bottom - top)
    : bottom - (v - currentScale.min) / (currentScale.max - currentScale.min) * (bottom - top);
  const unit = basis === "percent" ? "区间涨幅 %" : basis === "change" ? metric === "score" ? "区间变化 Δ 分" : metric === "rank" ? "名次改善 Δ" : "分差缩小 Δ 分"
    : metric === "score" ? "公开分数" : metric === "rank" ? "官方名次" : "距榜首 / 分";
  svg.append(svgNode("rect", {x:left,y:top,width:right-left,height:bottom-top,fill:"#0a1420"}));
  const defs = svgNode("defs"), clipPath = svgNode("clipPath", {id:"plot-clip"});
  clipPath.append(svgNode("rect", {x:left,y:top,width:right-left,height:bottom-top})); defs.append(clipPath); svg.append(defs);
  for (const tick of currentScale.ticks) {
    const py = y(tick); if (py < top - 1 || py > bottom + 1) continue;
    svg.append(svgNode("line", {x1:left,x2:right,y1:py,y2:py,stroke:Math.abs(tick)<1e-12 && basis!=="absolute" ? "#2d526b" : "#192c40","stroke-dasharray":"2 4"}));
    const label = (metric === "rank" && basis === "absolute" ? "#" : "") + tick.toFixed(currentScale.decimals);
    svg.append(svgNode("text", {x:right+9,y:py+3,class:"axis-label"}, label));
  }
  const tickCount = width < 550 ? 3 : width < 900 ? 5 : 7, span = viewport.to - viewport.from;
  for (let i = 0; i < tickCount; i++) {
    const t = viewport.from + span * i / (tickCount - 1), px = x(t);
    svg.append(svgNode("line", {x1:px,x2:px,y1:top,y2:bottom,stroke:"#15283b","stroke-dasharray":"2 5"}));
    svg.append(svgNode("text", {x:px,y:bottom+19,"text-anchor":i===0?"start":i===tickCount-1?"end":"middle"}, axisTime(t, span)));
  }
  svg.append(svgNode("line", {x1:right,x2:right,y1:top,y2:bottom,stroke:"#29435a"}));
  svg.append(svgNode("text", {x:left+2,y:12,class:"axis-unit"}, unit));
  const layer = svgNode("g", {"clip-path":"url(#plot-clip)"});
  selected.forEach((id, index) => {
    const points = transformed.get(id).points;
    layer.append(svgNode("path", {d:pathFor(points,x,y),fill:"none",stroke:palette[index],"stroke-width":id===focusId?2:1.55,"stroke-linejoin":"round",opacity:id===focusId?1:.88,"data-series":id}));
    const actual = points.filter((p) => p.value !== null && !p.synthetic);
    const markers = actual.length < 70 ? actual : actual.filter((p,i) => i===0 || i===actual.length-1 || p.value!==actual[i-1].value);
    for (const p of markers) layer.append(svgNode("circle", {cx:x(p.t),cy:y(p.value),r:actual.length===1?3.5:2.2,fill:palette[index],stroke:"#0a1420","stroke-width":1}));
  });
  svg.append(layer);
  const activeLine = selected.includes(focusId) ? focusId : selected[0], last = transformed.get(activeLine)?.points.filter((p) => p.value !== null).at(-1);
  if (last) {
    const py = y(last.value), color = palette[selected.indexOf(activeLine)];
    if (py >= top && py <= bottom) {
      svg.append(svgNode("line", {x1:left,x2:right,y1:py,y2:py,stroke:color,"stroke-dasharray":"4 5",opacity:.35}));
      svg.append(svgNode("rect", {x:right+2,y:py-9,width:width-right-5,height:18,fill:color,rx:1}));
      svg.append(svgNode("text", {x:right+7,y:py+3,class:"price-label"}, last.value.toFixed(currentScale.decimals)));
    }
  }
  const crossX = svgNode("line", {x1:0,x2:0,y1:top,y2:bottom,stroke:"#6a91ae","stroke-dasharray":"3 3",visibility:"hidden"});
  const crossY = svgNode("line", {x1:left,x2:right,y1:0,y2:0,stroke:"#6a91ae","stroke-dasharray":"3 3",visibility:"hidden"});
  svg.append(crossX,crossY);
  plot = {width,height,left,right,top,bottom,x,y,scale:currentScale,transformed,rawVisible,crossX,crossY,activeLine};
  const flat = values.length > 0 && [...rawVisible.values()].every((pts) => { const s=FX.summary(pts);return !s || Math.abs(s.high-s.low)<1e-12; });
  const noBase = basis==="percent" && values.length===0 && selected.some((id)=>transformed.get(id)?.base===0);
  $("chart-empty").hidden = values.length>0 && !flat;
  $("chart-empty").classList.toggle("empty-data", !values.length);
  $("chart-empty").textContent = !selected.length ? "点击左侧队名加入比较" : noBase ? "区间起点为 0，无法计算百分比；可切换区间变化" : !values.length ? "当前视窗没有有效观测，可缩小或平移查看" : "所选队伍在此区间的公开数值未变化";
  setText("chart-note", (basis==="absolute" ? "按可视数据自动定标" : "各队可视区间首个有效值为基准") + " · 断线留空" + (history.truncated ? " · 仅近 5000 次检查" : ""));
  renderNavigator(raw.get(focusId) || raw.get(selected[0]) || []);
  renderVisibleStats();
}
function renderNavigator(points) {
  const svg = $("range-chart"), width = Math.max(260, svg.clientWidth || 900), height = 56;
  svg.replaceChildren(); svg.setAttribute("viewBox","0 0 "+width+" "+height);
  const left=5,right=width-5,top=8,bottom=39;
  const x=(t)=>left+(t-timeBounds.from)/(timeBounds.to-timeBounds.from)*(right-left);
  const scale=FX.scale(points.map((p)=>p.value)), y=(v)=>bottom-(v-scale.min)/(scale.max-scale.min)*(bottom-top);
  svg.append(svgNode("path",{d:pathFor(points,x,y),fill:"none",stroke:"#365e7e","stroke-width":1.15}));
  const a=x(viewport.from),b=x(viewport.to);
  svg.append(svgNode("rect",{x:left,y:0,width:Math.max(0,a-left),height:41,fill:"#060d16",opacity:.6}));
  svg.append(svgNode("rect",{x:b,y:0,width:Math.max(0,right-b),height:41,fill:"#060d16",opacity:.6}));
  svg.append(svgNode("rect",{x:a,y:2,width:Math.max(1,b-a),height:39,fill:"#2e7598","fill-opacity":.14,stroke:"#367a9c","data-handle":"move"}));
  for (const [px,handle] of [[a,"left"],[b,"right"]]) {
    svg.append(svgNode("rect",{x:px-5,y:2,width:10,height:39,fill:"#1d4b65",stroke:"#4999bd","data-handle":handle,rx:1}));
    svg.append(svgNode("line",{x1:px,x2:px,y1:16,y2:27,stroke:"#81bbd7","pointer-events":"none"}));
  }
  svg.append(svgNode("text",{x:left+2,y:53,"font-size":8},timeText(new Date(timeBounds.from).toISOString())));
  svg.append(svgNode("text",{x:right-2,y:53,"text-anchor":"end","font-size":8},timeText(new Date(timeBounds.to).toISOString())));
  navPlot={width,left,right,x};
}
function renderVisibleStats() {
  if (!viewport || !state || !history) return;
  const scores = getSeries("score"), scorePoints=FX.clip(scores.get(focusId)||[],viewport,maxGap()), scoreStats=FX.summary(scorePoints);
  const rawMetric=getSeries(metric), rawStats=FX.summary(FX.clip(rawMetric.get(focusId)||[],viewport,maxGap()));
  const metricDigits=metric==="rank"?0:4;
  for (const [id,key] of [["range-open","open"],["range-high","high"],["range-low","low"],["range-close","close"]]) setText(id,num(rawStats?.[key],metricDigits));
  setText("readout-label",metric==="score"?"观测分数":metric==="rank"?"观测名次":"观测分差");
  setText("range-duration",durationText(viewport.to-viewport.from)+" / "+(rawStats?.observations||0)+" 次");
  setText("focus-best",num(scoreStats?.high));setText("focus-period-delta",signed(scoreStats?.delta));
  $("focus-period-delta").className=changeClass(scoreStats?.delta);
  setText("focus-observations",scoreStats?scoreStats.observations+" 次":"—");
  const pct=scoreStats && scoreStats.open!==0?scoreStats.delta/Math.abs(scoreStats.open)*100:null;
  setText("instrument-change","区间 "+signed(scoreStats?.delta)+" ("+signed(pct,3)+"%)");
  $("instrument-change").className=changeClass(scoreStats?.delta);
  const header=node("div",null,"analysis-row header");header.append(node("span","队伍"),node("span","区间变化"),node("span","高 − 低"),node("span","观测次数"));
  $("range-analysis").replaceChildren(header,...selected.map((id)=>{
    const r=state.rows.find((r)=>r.team_id===id),stats=FX.summary(FX.clip(scores.get(id)||[],viewport,maxGap()));
    const row=node("div",null,"analysis-row");row.append(node("span",r?.name||id),node("span",signed(stats?.delta),changeClass(stats?.delta)),node("span",stats?num(stats.high-stats.low):"—"),node("span",stats?.observations??"—"));return row;
  }));
}
function scheduleChart() {
  if (framePending) return;
  framePending=true;requestAnimationFrame(()=>{framePending=false;renderChart();});
}
function setViewport(next, user=true) {
  if (!timeBounds) return;
  viewport=FX.windowBounds(next.from,next.to,timeBounds);
  if (user) {rangePreset=null;followLatest=Math.abs(viewport.to-timeBounds.to)<2;}
  scheduleChart();
}
function zoomTime(factor,anchor) {
  if (!viewport||!timeBounds) return;
  setViewport(FX.zoom(viewport,timeBounds,factor,anchor??viewport.to));
}
function localPoint(event,svg) {
  const r=svg.getBoundingClientRect(),v=svg.viewBox.baseVal;
  return {x:(event.clientX-r.left)/r.width*v.width,y:(event.clientY-r.top)/r.height*v.height};
}
function hideHover() { $("chart-tooltip").hidden=true;plot?.crossX.setAttribute("visibility","hidden");plot?.crossY.setAttribute("visibility","hidden"); }
function showHover(event) {
  if (!plot||!history||!selected.length) return;
  const p=localPoint(event,$("trend-chart"));
  if (p.x<plot.left||p.x>plot.right||p.y<plot.top||p.y>plot.bottom) {hideHover();return;}
  const wanted=viewport.from+(p.x-plot.left)/(plot.right-plot.left)*(viewport.to-viewport.from);
  const candidates=history.observations.filter((o)=>Date.parse(o.checked_at)>=viewport.from&&Date.parse(o.checked_at)<=viewport.to);
  if (!candidates.length) {hideHover();return;}
  const observation=candidates.reduce((a,b)=>Math.abs(Date.parse(a.checked_at)-wanted)<Math.abs(Date.parse(b.checked_at)-wanted)?a:b);
  const at=Date.parse(observation.checked_at),lines=["观测 "+timeText(observation.checked_at,true)];
  if (!observation.success) lines.push("本次检查失败，没有有效成绩");
  else for (const id of selected) {
    const row=state.rows.find((r)=>r.team_id===id),point=plot.transformed.get(id)?.points.find((p)=>p.t===at&&!p.synthetic);
    lines.push((row?.name||id)+"  "+num(point?.value,Math.max(4,plot.scale.decimals))+(basis==="percent"?"%":"")+(basis!=="absolute"?"  | 原值 "+num(point?.raw,metric==="rank"?0:4):""));
  }
  const tooltip=$("chart-tooltip");tooltip.textContent=lines.join("\n");tooltip.hidden=false;
  const rect=$("chart-wrap").getBoundingClientRect();tooltip.style.left=FX.clamp(event.clientX-rect.left+14,6,Math.max(6,rect.width-280))+"px";tooltip.style.top=FX.clamp(event.clientY-rect.top+15,6,Math.max(6,rect.height-145))+"px";
  const px=plot.x(at);plot.crossX.setAttribute("x1",px);plot.crossX.setAttribute("x2",px);plot.crossX.setAttribute("visibility","visible");
  const active=plot.transformed.get(plot.activeLine)?.points.find((p)=>p.t===at&&!p.synthetic);
  if (active?.value!==null&&active?.value!==undefined){const py=plot.y(active.value);plot.crossY.setAttribute("y1",py);plot.crossY.setAttribute("y2",py);plot.crossY.setAttribute("visibility","visible");}
}
function maybeNotify() {
  if (!state.snapshot_id) return;
  const key = activeStage + ":seen", previous = readPref(key, null);
  savePref(key, state.snapshot_id);
  if (previous === null || previous === state.snapshot_id || !readPref("notifications", false)) return;
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const latestAt = state.events[0]?.at;
  const events = state.events.filter((e) => e.at === latestAt && e.kind !== "baseline");
  if (events.length) {
    try { new Notification(`${activeStage} · 公开榜单有变化`, {body: events.slice(0, 2).map((e) => e.title).join("\n"), tag: `${activeStage}:${state.snapshot_id}`}); }
    catch { /* 个别浏览器不支持桌面通知时，看板动态仍正常记录。 */ }
  }
}
function renderNotificationButton() {
  $("notify-button").textContent = readPref("notifications", false) && "Notification" in window && Notification.permission === "granted" ? "变化提醒已开启" : "开启变化提醒";
  $("notify-button").title = "网页打开时，提醒当前阶段的公开榜单变化";
}


async function load(forceHistory=false) {
  if (inFlight&&!forceHistory) return;
  const seq=++requestSeq;inFlight=true;
  try {
    const data=await api("/api/state"+(activeStage?"?stage="+encodeURIComponent(activeStage):""));
    if(seq!==requestSeq)return;
    const changedStage=!state||state.stage!==data.stage;state=data;activeStage=data.stage;
    if(changedStage){history=null;loadStagePreferences();}
    renderState(changedStage);maybeNotify();
    const nextKey=activeStage+":"+state.last_check?.id;
    if(forceHistory||historyKey!==nextKey) {
      if(activeStage) {
        const result=await api("/api/history?stage="+encodeURIComponent(activeStage)+"&hours=0");
        if(seq!==requestSeq)return;
        history=result;historyKey=nextKey;seriesCache.clear();
      }
      renderChart();
    }
  } catch(error) {
    if(seq!==requestSeq)return;
    $("error-banner").hidden=false;setText("error-banner","本地采集器未连接："+error.message+"。已保存的记录仍保留在电脑上。");
    setText("status-label","连接断开");$("status-dot").className="dot bad";setText("next-check","等待重新连接");
  } finally {if(seq===requestSeq)inFlight=false;}
}
const mainChart=$("trend-chart"),rangeChart=$("range-chart");
mainChart.addEventListener("wheel",(event)=>{
  if(!plot||!viewport)return;
  event.preventDefault();hideHover();
  const p=localPoint(event,mainChart),factor=Math.exp(FX.clamp(event.deltaY*.002,-.65,.65));
  if(event.shiftKey||p.x>plot.right) {
    autoY=false;const ratio=FX.clamp((plot.bottom-p.y)/(plot.bottom-plot.top),0,1);
    const anchor=plot.scale.min+(plot.scale.max-plot.scale.min)*(metric==="rank"&&basis==="absolute"?1-ratio:ratio);
    manualY=FX.zoomScale(plot.scale,factor,anchor);scheduleChart();
  } else if(Math.abs(event.deltaX)>Math.abs(event.deltaY))setViewport(FX.pan(viewport,timeBounds,event.deltaX/(plot.right-plot.left)*(viewport.to-viewport.from)));
  else zoomTime(factor,viewport.from+FX.clamp((p.x-plot.left)/(plot.right-plot.left),0,1)*(viewport.to-viewport.from));
},{passive:false});
mainChart.addEventListener("pointerdown",(event)=>{
  if(!plot||event.button!==0)return;
  const p=localPoint(event,mainChart);if(p.y<plot.top||p.y>plot.bottom)return;
  chartDrag={mode:p.x>plot.right?"scale":"pan",x:p.x,y:p.y,view:{...viewport},scale:{min:plot.scale.min,max:plot.scale.max},plotWidth:plot.right-plot.left};
  hideHover();mainChart.setPointerCapture(event.pointerId);mainChart.style.cursor=chartDrag.mode==="scale"?"ns-resize":"grabbing";event.preventDefault();
});
mainChart.addEventListener("pointermove",(event)=>{
  if(!chartDrag){showHover(event);return;}
  const p=localPoint(event,mainChart),d=chartDrag;
  if(d.mode==="scale"){autoY=false;manualY=FX.zoomScale(d.scale,Math.exp(FX.clamp((p.y-d.y)*.012,-6,6)));scheduleChart();}
  else setViewport(FX.pan(d.view,timeBounds,-(p.x-d.x)/d.plotWidth*(d.view.to-d.view.from)));
});
function endChartDrag(event){chartDrag=null;if(mainChart.hasPointerCapture(event.pointerId))mainChart.releasePointerCapture(event.pointerId);mainChart.style.cursor="crosshair";}
mainChart.addEventListener("pointerup",endChartDrag);mainChart.addEventListener("pointercancel",endChartDrag);mainChart.addEventListener("pointerleave",()=>{if(!chartDrag)hideHover();});
mainChart.addEventListener("dblclick",(event)=>{const p=localPoint(event,mainChart);if(plot&&p.x>plot.right){resetY();renderChart();}else resetView();});
rangeChart.addEventListener("pointerdown",(event)=>{
  if(!navPlot||!viewport||event.button!==0)return;
  const p=localPoint(event,rangeChart),t=timeBounds.from+(p.x-navPlot.left)/(navPlot.right-navPlot.left)*(timeBounds.to-timeBounds.from);
  let handle=event.target.getAttribute("data-handle")||"move";
  if(t<viewport.from||t>viewport.to){const span=viewport.to-viewport.from;viewport=FX.windowBounds(t-span/2,t+span/2,timeBounds);rangePreset=null;followLatest=false;}
  navDrag={handle,x:p.x,view:{...viewport},width:navPlot.right-navPlot.left,bounds:{...timeBounds}};
  rangeChart.setPointerCapture(event.pointerId);scheduleChart();event.preventDefault();
});
rangeChart.addEventListener("pointermove",(event)=>{
  if(!navDrag)return;
  const p=localPoint(event,rangeChart),d=navDrag,delta=(p.x-d.x)/d.width*(d.bounds.to-d.bounds.from),minSpan=Math.min(60000,d.bounds.to-d.bounds.from);
  if(d.handle==="left")setViewport({from:FX.clamp(d.view.from+delta,d.bounds.from,d.view.to-minSpan),to:d.view.to});
  else if(d.handle==="right")setViewport({from:d.view.from,to:FX.clamp(d.view.to+delta,d.view.from+minSpan,d.bounds.to)});
  else setViewport(FX.pan(d.view,d.bounds,delta));
});
function endNavDrag(event){navDrag=null;if(rangeChart.hasPointerCapture(event.pointerId))rangeChart.releasePointerCapture(event.pointerId);}
rangeChart.addEventListener("pointerup",endNavDrag);rangeChart.addEventListener("pointercancel",endNavDrag);
function resetView(){rangePreset=0;followLatest=true;viewport=null;resetY();renderChart();}
$("zoom-in").addEventListener("click",()=>zoomTime(.6));
$("zoom-out").addEventListener("click",()=>zoomTime(1/.6));
$("reset-chart").addEventListener("click",resetView);
$("auto-scale").addEventListener("click",()=>{autoY=!autoY;manualY=autoY?null:plot?{min:plot.scale.min,max:plot.scale.max}:null;renderChart();});
$("live-button").addEventListener("click",()=>{followLatest=true;if(viewport&&timeBounds)viewport=FX.windowBounds(timeBounds.to-(viewport.to-viewport.from),timeBounds.to,timeBounds);renderChart();});
$("expand-chart").addEventListener("click",()=>{const expanded=document.body.classList.toggle("chart-focus");$("expand-chart").setAttribute("aria-pressed",String(expanded));requestAnimationFrame(renderChart);});
$("solo-focus").addEventListener("click",()=>{if(!focusId)return;selected=[focusId];savePref(activeStage+":selected",selected);seriesCache.clear();resetY();renderLegend();renderChart();});
for(const button of document.querySelectorAll("[data-range]"))button.addEventListener("click",()=>{
  rangePreset=Number(button.dataset.range);followLatest=true;viewport=null;renderChart();
});
for(const button of document.querySelectorAll("[data-metric]"))button.addEventListener("click",()=>{
  metric=button.dataset.metric;if(metric!=="score"&&basis==="percent")basis="change";resetY();renderChart();
});
$("basis-select").addEventListener("change",(event)=>{basis=event.target.value;savePref(activeStage+":fx-basis",basis);resetY();renderChart();});
mainChart.addEventListener("keydown",(event)=>{
  if(!viewport)return;
  if(["+","=","-","ArrowLeft","ArrowRight","0","a","A","r","R"].includes(event.key))event.preventDefault();
  if(event.key==="+"||event.key==="=")zoomTime(.6);
  else if(event.key==="-")zoomTime(1/.6);
  else if(event.key==="ArrowLeft"||event.key==="ArrowRight")setViewport(FX.pan(viewport,timeBounds,(event.key==="ArrowLeft"?-1:1)*(viewport.to-viewport.from)*.2));
  else if(event.key==="0")resetView();
  else if(event.key.toLowerCase()==="a"){resetY();renderChart();}
  else if(event.key.toLowerCase()==="r")$("live-button").click();
});
document.addEventListener("keydown",(event)=>{if(event.key==="Escape"&&document.body.classList.contains("chart-focus"))$("expand-chart").click();});
for(const button of document.querySelectorAll("[data-tab]"))button.addEventListener("click",()=>{
  for(const b of document.querySelectorAll("[data-tab]")){b.classList.toggle("active",b===button);$(b.dataset.tab).hidden=b!==button;}
});
$("search").addEventListener("input",renderTable);
$("watch-search").addEventListener("input",()=>renderMarket());
$("watch-filter").addEventListener("click",()=>{showWatched=!showWatched;renderMarket();renderTable();});
$("focus-select").addEventListener("change",(event)=>focusTeam(event.target.value));
$("focus-watch").addEventListener("click",()=>toggleWatch(focusId));
for(const button of document.querySelectorAll("[data-sort]"))button.addEventListener("click",()=>{
  if(sortKey===button.dataset.sort)sortDirection*=-1;else{sortKey=button.dataset.sort;sortDirection=sortKey==="score"?-1:1;}renderTable();
});
$("refresh-button").addEventListener("click",async()=>{
  $("refresh-button").disabled=true;
  try{const result=await api("/api/refresh",true);toast(result.accepted?"已安排检查，新数据自动显示":result.busy?"正在采集，请稍候":"刚检查过，"+result.retry_after+" 秒后可再次检查");await load(true);if(result.accepted){setTimeout(()=>load(true),4500);setTimeout(()=>load(true),15000);}}
  catch{toast("采集器未连接，请检查启动状态");}
  finally{$("refresh-button").disabled=false;}
});
$("notify-button").addEventListener("click",async()=>{
  if(!("Notification" in window)){toast("此浏览器不支持桌面通知，变化仍会保存");return;}
  if(readPref("notifications",false)){savePref("notifications",false);toast("已关闭桌面提醒");}
  else{const permission=await Notification.requestPermission();savePref("notifications",permission==="granted");toast(permission==="granted"?"已开启：网页打开时提醒当前阶段的变化":"未获通知权限，变化仍会保存");}renderNotificationButton();
});
document.addEventListener("visibilitychange",()=>{if(!document.hidden)load(true);});
let resizeTimer;
window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderChart,90);});
renderNotificationButton();load();
setInterval(()=>load(),15000);
function tickClock(){
  setText("terminal-clock",new Intl.DateTimeFormat("zh-CN",{timeZone:"Asia/Shanghai",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(new Date())+" CST");
  if(state&&$("status-dot").className!=="dot bad")renderStatus();
  document.querySelector(".statusbar .dot").className=$("status-dot").className;
}
tickClock();setInterval(tickClock,1000);
