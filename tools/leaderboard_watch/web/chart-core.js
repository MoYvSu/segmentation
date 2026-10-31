/* 图表计算独立于页面，便于验证微小分差、时间缩放和断点。 */
(function (root) {
  "use strict";
  const finite = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  function domain(times, interval = 300000) {
    const valid = times.filter(Number.isFinite);
    if (!valid.length) return null;
    const from = Math.min(...valid), to = Math.max(...valid);
    return to > from ? {from, to} : {from: from - interval, to};
  }
  function windowBounds(from, to, bounds, minimum = 60000) {
    const entire = bounds.to - bounds.from;
    const span = clamp(to - from, Math.min(minimum, entire), entire);
    const start = clamp((from + to - span) / 2, bounds.from, bounds.to - span);
    return {from: start, to: start + span};
  }
  function zoom(view, bounds, factor, anchor = (view.from + view.to) / 2) {
    const ratio = clamp((anchor - view.from) / (view.to - view.from), 0, 1);
    const span = (view.to - view.from) * factor;
    return windowBounds(anchor - span * ratio, anchor + span * (1 - ratio), bounds);
  }
  function pan(view, bounds, delta) {
    const span = view.to - view.from;
    const from = clamp(view.from + delta, bounds.from, bounds.to - span);
    return {from, to: from + span};
  }
  function preset(bounds, duration) {
    return duration ? {from: Math.max(bounds.from, bounds.to - duration), to: bounds.to} : {...bounds};
  }
  function metricValue(row, rows, metric) {
    if (!row) return null;
    if (metric === "rank") return finite(row.rank) ? Number(row.rank) : null;
    if (!finite(row.score)) return null;
    if (metric === "score") return Number(row.score);
    const leader = rows.find((r) => finite(r.score));
    return leader ? Number(leader.score) - Number(row.score) : null;
  }
  function series(history, ids, metric) {
    const snapshots = new Map((history?.snapshots || []).map((s) => [s.id, s]));
    const output = new Map(ids.map((id) => [id, []]));
    for (const observation of history?.observations || []) {
      const t = Date.parse(observation.checked_at);
      if (!Number.isFinite(t)) continue;
      const snapshot = snapshots.get(observation.snapshot_id);
      for (const id of ids) {
        const row = snapshot?.rows.find((r) => r.team_id === id);
        const value = observation.success && snapshot ? metricValue(row, snapshot.rows, metric) : null;
        output.get(id).push({t, value, observedAt: t, synthetic: false, success: Boolean(observation.success)});
      }
    }
    for (const points of output.values()) points.sort((a, b) => a.t - b.t);
    return output;
  }
  function clip(points, view, maxGap) {
    const result = points.filter((p) => p.t >= view.from && p.t <= view.to).map((p) => ({...p}));
    // 仅在两次相邻有效检查之间延续阶梯值。失败和长时间缺失都不补线。
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      if (!finite(a.value) || !finite(b.value) || b.t - a.t > maxGap) continue;
      if (a.t < view.from && b.t > view.from) result.unshift({...a, t: view.from, synthetic: true});
      if (a.t < view.to && b.t > view.to) result.push({...a, t: view.to, synthetic: true});
    }
    return result.sort((a, b) => a.t - b.t);
  }
  function relative(points, basis, metric) {
    const first = points.find((p) => finite(p.value));
    const base = first?.value ?? null;
    const direction = metric === "score" ? 1 : -1;
    return {base, points: points.map((p) => ({...p, raw: p.value,
      value: !finite(p.value) || base === null ? null : basis === "absolute" ? p.value
        : basis === "percent" ? base === 0 ? null : (p.value - base) / Math.abs(base) * 100
          : (p.value - base) * direction}))};
  }
  function niceStep(value) {
    if (!(value > 0)) return 1;
    const power = 10 ** Math.floor(Math.log10(value)), fraction = value / power;
    return ([1, 2, 2.5, 5, 10].find((n) => n >= fraction) || 10) * power;
  }
  function ticksFor(min, max, integer = false) {
    const step = integer ? Math.max(1, Math.ceil((max - min) / 5)) : niceStep((max - min) / 5);
    let decimals = Math.max(0, Math.ceil(-Math.log10(step)));
    if (Math.abs(step * 10 ** decimals - Math.round(step * 10 ** decimals)) > 1e-6) decimals++;
    decimals = Math.min(10, Math.max(integer ? 0 : 4, decimals));
    const ticks = [];
    for (let i = Math.ceil(min / step - 1e-8); i * step <= max + step * 1e-8 && ticks.length < 12; i++) ticks.push(Number((i * step).toFixed(decimals)));
    return {min, max, ticks, decimals, step};
  }
  function scale(values, {integer = false, floor = null, manual = null} = {}) {
    if (manual && finite(manual.min) && finite(manual.max) && manual.max > manual.min) return ticksFor(manual.min, manual.max, integer);
    const valid = values.filter(finite).map(Number);
    let min = valid.length ? Math.min(...valid) : 0, max = valid.length ? Math.max(...valid) : 1;
    const width = max - min;
    // 有变化时按真实波动幅度缩放，去掉固定 0.2 分的最小留白。
    const pad = integer ? Math.max(.4, width * .1) : width > 1e-12 ? Math.max(width * .12, Math.abs(max) * 1e-14) : .0002;
    min -= pad; max += pad;
    if (integer) { min = Math.floor(min); max = Math.ceil(max); }
    if (floor !== null) min = Math.max(floor, min);
    if (max <= min) max = min + (integer ? 1 : .0004);
    const step = integer ? Math.max(1, Math.ceil((max - min) / 5)) : niceStep((max - min) / 5);
    min = Math.floor(min / step) * step; max = Math.ceil(max / step) * step;
    if (floor !== null) min = Math.max(floor, min);
    return ticksFor(min, max, integer);
  }
  function zoomScale(current, factor, anchor = (current.min + current.max) / 2) {
    const span = clamp((current.max - current.min) * factor, 1e-9, 1e9);
    const ratio = (anchor - current.min) / (current.max - current.min);
    return {min: anchor - span * ratio, max: anchor + span * (1 - ratio)};
  }
  function summary(points) {
    const valid = points.filter((p) => finite(p.value));
    if (!valid.length) return null;
    const values = valid.map((p) => p.value);
    return {open: values[0], close: values.at(-1), high: Math.max(...values), low: Math.min(...values), delta: values.at(-1) - values[0], observations: valid.filter((p) => !p.synthetic).length};
  }
  const api = {clamp, domain, windowBounds, zoom, pan, preset, metricValue, series, clip, relative, scale, zoomScale, summary};
  root.LeaderboardChart = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(globalThis);
