"use strict";

const $ = (id) => document.getElementById(id);
let state = { job: null, applied: {}, selected: null, driftKey: null,
              validationLog: [] };

// ---- tab switching -------------------------------------------------------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $("tab-sim").classList.toggle("hidden", btn.dataset.tab !== "sim");
    $("tab-upload").classList.toggle("hidden", btn.dataset.tab !== "upload");
  });
});

function simParams() {
  const num = (id) => parseFloat($(id).value);
  const drops = $("dropped_events").value.split(",").map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isInteger(n));
  const channels = Math.max(1, Math.min(2, num("channels")));
  const steps = [];
  if (num("step_time") > 0 && num("step_delta") !== 0) {
    steps.push([num("step_time"), num("step_delta") / 1000]);
  }
  const mis = [];
  if (num("mis_idx") >= 0 && num("mis_shift") !== 0) {
    mis.push([num("mis_idx"), num("mis_shift") / 1000]);
  }
  return {
    n_events: num("n_events"), source_fps: num("source_fps"),
    recorded_fps: num("recorded_fps"), channels,
    audio_delay: [num("ad0") / 1000, num("ad1") / 1000].slice(0, channels),
    video_delay: num("vd") / 1000, audio_noise: num("audio_noise"),
    brightness_noise: num("brightness_noise"),
    exposure_drift: num("exposure_drift"),
    echo_delay: num("echo_delay") / 1000, echo_gain: num("echo_gain"),
    dropped_events: drops,
    clock_drift_ppm: num("clock_drift_ppm"),
    delay_steps: steps, misaligned_events: mis,
  };
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

$("run-sim").addEventListener("click", async () => {
  setStatus("正在生成媒体并运行校准内核…", "");
  const r = await postJSON("/api/simulate", simParams());
  if (r.job_id) pollJob(r.job_id);
});

$("run-upload").addEventListener("click", async () => {
  const file = $("rec-file").files[0];
  if (!file) { setStatus("请先选择录制文件", "error"); return; }
  const fd = new FormData();
  fd.append("params", new Blob([JSON.stringify({ channels: 2 })],
    { type: "application/json" }));
  fd.append("recording", file);
  setStatus("上传并经 ffmpeg 解码分析…", "");
  const res = await fetch("/api/analyze", { method: "POST", body: fd });
  const r = await res.json();
  if (r.error) { setStatus(r.error, "error"); return; }
  pollJob(r.job_id);
});

let pollTimer = null;
function pollJob(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (!job.id) return;
    if (job.status === "queued") { setStatus("处理中…", ""); return; }
    clearInterval(pollTimer);
    state.job = job;
    if (job.status === "failed") {
      setStatus("校准失败：" + job.error, "error");
      setBadge("fail", "失败");
      return;
    }
    setStatus("校准完成", "ok");
    render();
  }, 700);
}

function setStatus(msg, cls) {
  const el = $("run-status");
  el.textContent = msg;
  el.className = "status " + cls;
}

function setBadge(kind, text) {
  const b = $("verdict-badge");
  b.className = "badge " + kind;
  b.textContent = text;
}

// ---- result rendering ----------------------------------------------------
function render() {
  const job = state.job;
  if (!job || !job.result) return;
  const r = job.result;
  setBadge(r.verdict, { pass: "通过", warn: "警告", fail: "失败" }[r.verdict]);

  $("reasons").innerHTML = r.reasons.length
    ? r.reasons.map((x) => `<div class="reason">⚠ ${x}</div>`).join("")
    : `<div class="reason" style="color:#57d98a">所有检查通过</div>`;

  const cards = [];
  r.audio.forEach((a) => {
    cards.push(metricCard(`音频通道 ${a.channel}`, [
      ["估计延迟", `${ms(a.offset)} ms`],
      ["95% 区间", `${ms(a.offset_lo)} ~ ${ms(a.offset_hi)} ms`],
      ["残差抖动σ", `${ms(a.jitter_std)} ms`],
      ["匹配/计划", `${a.n_matched}/${a.n_scheduled}`],
      ["漏检", a.n_dropped], ["重复峰", a.n_duplicates],
    ]));
  });
  const v = r.video;
  cards.push(metricCard("视频（帧边界区间，不宣称亚帧精度）", [
    ["估计延迟中值", `${ms(v.offset)} ms`],
    ["观测区间", `${ms(v.offset_lo)} ~ ${ms(v.offset_hi)} ms`],
    ["帧抖动σ", `${ms(v.jitter_std)} ms`],
    ["匹配/计划", `${v.n_matched}/${v.n_scheduled}`],
    ["漏检", v.n_dropped], ["拒绝候选", v.n_rejected],
  ]));
  $("metrics").innerHTML = cards.join("");

  $("av-offsets").innerHTML = "<h2>音画偏移（音频 − 视频）</h2>" +
    r.av_offsets.map((o) => `
      <div class="metric-card"><h3>通道 ${o.channel}</h3>
      <div class="metric-row">
        <span>偏移 <b>${o.audio_minus_video_ms.toFixed(2)} ms</b></span>
        <span>95% [${o.ci95_lo_ms.toFixed(2)}, ${o.ci95_hi_ms.toFixed(2)}] ms</span>
        <span>${o.interpretation}</span>
      </div></div>`).join("");

  $("origin-note").innerHTML = r.origin_known
    ? '<div class="origin">录制起点已知：可估计各通道相对媒体起点的绝对呈现延迟。</div>'
    : '<div class="origin">⚠ 录制起点未知：仅测量音画相对偏移，不宣称屏幕/扬声器绝对延迟，也不使用任何命令发出时刻。</div>';

  renderAdjust(r, job);
  renderDrift(r);
  plotAll(job.series);
  renderMedia(job.media);
}

function metricCard(title, rows) {
  return `<div class="metric-card"><h3>${title}</h3><div class="metric-row">` +
    rows.map(([k, val]) => `<span>${k} <b>${val}</b></span>`).join("") +
    "</div></div>";
}
const ms = (s) => (s * 1000).toFixed(2);

// ---- drift / jump analysis ----------------------------------------------
function driftKeys(r) {
  return Object.keys(r.drift || {});
}

function renderDrift(r) {
  const keys = driftKeys(r);
  const panel = $("drift-panel");
  if (!keys.length) { panel.style.display = "none"; return; }
  panel.style.display = "";
  if (!state.driftKey || !keys.includes(state.driftKey)) {
    state.driftKey = keys[0];
  }
  $("drift-channel-tabs").innerHTML = keys.map((k) =>
    `<button class="tab ${k === state.driftKey ? "active" : ""}"
      data-drift="${k}">${k}</button>`).join("");
  document.querySelectorAll("[data-drift]").forEach((b) =>
    b.addEventListener("click", () => {
      state.driftKey = b.dataset.drift; renderDrift(r);
    }));
  const da = r.drift[state.driftKey];
  $("drift-notes").innerHTML = da.notes.length
    ? da.notes.map((n) => `<div class="reason">ℹ ${n}</div>`).join("")
    : "";

  plotDrift(da);

  $("drift-segments").innerHTML = "<h3>分段参数</h3><table><tr>" +
    "<th>区间 (s)</th><th>状态</th><th>偏移@t_ref (ms)</th>" +
    "<th>速率 (ppm)</th><th>拟合/留出</th><th>判据</th></tr>" +
    da.segments.map((s) => s.status === "ok"
      ? `<tr><td>${s.t_lo.toFixed(2)} ~ ${s.t_hi.toFixed(2)}</td><td>有效</td>
         <td>${ms(s.offset)} ± ${ms(s.offset_ci95)}</td>
         <td>${(s.slope * 1e6).toFixed(1)} ± ${(s.slope_ci95 * 1e6).toFixed(1)}</td>
         <td>${s.n_fit}/${s.n_holdout}</td><td>${s.classification}</td></tr>`
      : `<tr><td>${s.t_lo.toFixed(2)} ~ ${s.t_hi.toFixed(2)}</td>
         <td class="warn-text">数据不足</td><td colspan="3">不给确定校准参数</td>
         <td>${s.n_fit}/${s.n_holdout}</td><td>${s.classification}</td></tr>`
    ).join("") + "</table>" +
    (da.jumps.length ? "<h3>跳变</h3><table><tr><th>时刻 (s)</th>" +
      "<th>阶跃 (ms)</th><th>说明</th></tr>" +
      da.jumps.map((j) => `<tr><td>${j.time.toFixed(2)}</td>
        <td>${(j.step * 1000).toFixed(1)} ± ${(j.step_ci95 * 1000).toFixed(1)}</td>
        <td>${j.note}</td></tr>`).join("") + "</table>" : "");

  $("drift-excluded").innerHTML = da.excluded.length
    ? "<h3>剔除记录（异常点留痕，未静默删除）</h3><table><tr><th>事件</th>" +
      "<th>时刻 (s)</th><th>残差 (ms)</th><th>剔除依据</th></tr>" +
      da.excluded.map((e) => `<tr><td>${e.event_index}</td>
        <td>${e.time.toFixed(3)}</td><td>${(e.residual * 1000).toFixed(2)}</td>
        <td>${e.reason}</td></tr>`).join("") + "</table>"
    : "";

  const vp = da.validation_points;
  const rmsText = da.rms_after == null
    ? "无有效留出点"
    : `RMS ${ms(da.rms_before)} → ${ms(da.rms_after)} ms`;
  $("drift-validation").innerHTML =
    `<h3>独立留出验证（未参与拟合）</h3>
     <div class="metric-row"><span>校正前 → 校正后 <b>${rmsText}</b></span>
     <span>覆盖范围 <b>${da.t_lo.toFixed(2)} ~ ${da.t_hi.toFixed(2)} s</b>，范围外为外推</span></div>` +
    (vp.length ? "<table><tr><th>事件</th><th>时刻 (s)</th><th>校正前 (ms)</th>" +
      "<th>校正后 (ms)</th><th>位置</th></tr>" +
      vp.map((p) => `<tr class="${p.status === "ok" ? "" : "warn-text"}">
        <td>${p.event_index}</td><td>${p.time.toFixed(2)}</td>
        <td>${(p.residual_before * 1000).toFixed(2)}</td>
        <td>${p.residual_after == null ? "—" : (p.residual_after * 1000).toFixed(2)}</td>
        <td>${{ ok: "覆盖内", gap: "空洞", extrapolated: "外推",
                 insufficient: "数据不足" }[p.status]}</td></tr>`).join("") +
      "</table>" : "");

  $("export-params").disabled = false;
}

function plotDrift(da) {
  const cv = $("drift-canvas");
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const t0 = da.t_lo, t1 = da.t_hi;
  if (!(t1 > t0)) return;
  const all = da.fit_series.map((p) => p[1])
    .concat(da.validation_points.map((p) => p.residual_before))
    .concat(da.excluded.map((e) => e.residual));
  if (!all.length) return;
  let y0 = Math.min(...all), y1 = Math.max(...all);
  const pad = Math.max((y1 - y0) * 0.15, 0.001);
  y0 -= pad; y1 += pad;
  const x = (t) => ((t - t0) / (t1 - t0)) * (w - 8) + 4;
  const y = (r) => h - 16 - ((r - y0) / (y1 - y0)) * (h - 32);
  // gap shading: no parameters claimed there
  da.segments.forEach((s) => (s.gaps || []).forEach(([lo, hi]) => {
    ctx.fillStyle = "rgba(255,209,102,0.15)";
    ctx.fillRect(x(lo), 0, Math.max(2, x(hi) - x(lo)), h);
  }));
  // fitted segment lines
  da.segments.forEach((s) => {
    if (s.status !== "ok") return;
    ctx.strokeStyle = "#57d98a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const f = (t) => s.offset + s.slope * (t - s.t_ref);
    ctx.moveTo(x(s.t_lo), y(f(s.t_lo)));
    ctx.lineTo(x(s.t_hi), y(f(s.t_hi)));
    ctx.stroke();
  });
  // jump boundaries
  da.jumps.forEach((j) => {
    ctx.strokeStyle = "#ffd166";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(j.time), 0); ctx.lineTo(x(j.time), h);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  // fit points
  ctx.fillStyle = "#4cc9f0";
  da.fit_series.forEach(([t, r]) => {
    ctx.beginPath(); ctx.arc(x(t), y(r), 2.5, 0, 7); ctx.fill();
  });
  // holdout validation points (hollow)
  ctx.strokeStyle = "#b8c4d4";
  da.validation_points.forEach((p) => {
    ctx.beginPath(); ctx.arc(x(p.time), y(p.residual_before), 3.5, 0, 7);
    ctx.stroke();
  });
  // excluded outliers (red cross)
  ctx.strokeStyle = "#ef476f";
  da.excluded.forEach((e) => {
    const px = x(e.time), py = y(e.residual);
    ctx.beginPath();
    ctx.moveTo(px - 4, py - 4); ctx.lineTo(px + 4, py + 4);
    ctx.moveTo(px + 4, py - 4); ctx.lineTo(px - 4, py + 4);
    ctx.stroke();
  });
}

$("export-params").addEventListener("click", async () => {
  const st = $("export-status");
  st.className = "status";
  const res = await fetch("/api/export");
  if (!res.ok) {
    const j = await res.json();
    st.className = "status error";
    st.textContent = "导出失败：" + (j.error || res.status);
    return;
  }
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "playback_params.json";
  a.click();
  URL.revokeObjectURL(a.href);
  st.className = "status ok";
  st.textContent = "已导出参数文件（仅生成文件，未修改任何设备设置）。";
});

// ---- kernel-driven delay adjustment -------------------------------------
function renderAdjust(r, job) {
  const allowed = r.verdict !== "fail";
  const box = $("adjust-controls");
  box.innerHTML = "";
  Object.entries(r.suggested_delay_ms).forEach(([key, sug]) => {
    const wrap = document.createElement("label");
    wrap.textContent = key;
    const inp = document.createElement("input");
    inp.type = "number"; inp.id = "dly_" + key; inp.step = "0.1";
    inp.value = (state.applied[key] ?? sug).toFixed(2);
    inp.disabled = !allowed;
    wrap.appendChild(inp);
    box.appendChild(wrap);
  });
  $("apply-delay").disabled = !allowed;
  if (!allowed) {
    $("applied-feedback").className = "status error";
    $("applied-feedback").textContent =
      "校准未通过，延迟调整已锁定——失败状态反馈回内核，请先解决对齐问题。";
  }
}

$("apply-delay").addEventListener("click", async () => {
  const delay_ms = {};
  document.querySelectorAll("#adjust-controls input").forEach((inp) => {
    delay_ms[inp.id.replace("dly_", "")] = parseFloat(inp.value);
  });
  const r = await postJSON("/api/apply_delay", {
    job_id: state.job.id, delay_ms,
  });
  const fb = $("applied-feedback");
  if (r.ok) {
    state.applied = r.applied_delay;
    state.validationLog = r.validation_log || state.validationLog;
    fb.className = "status ok";
    fb.textContent = "已应用修正（内核确认）：" + JSON.stringify(r.applied_delay);
    renderCorrected(r.corrected, r.anomalies);
    renderValidationLog();
    plotAll(state.job.series);  // playback markers move by the applied delay
  } else {
    fb.className = "status error";
    fb.textContent = "被内核拒绝：" + r.error;
  }
});

function renderCorrected(corrected, anomalies) {
  if (!corrected) return;
  let html = "<h3>修正后测量偏移（实际运算）</h3><table><tr>" +
    "<th>对象</th><th>修正后偏移 (ms)</th><th>95% 半宽</th><th>判定</th></tr>";
  corrected.av.forEach((c) => {
    const bad = Math.abs(c.corrected_ms) > c.ci95_half_ms;
    html += `<tr class="${bad ? "warn-text" : ""}"><td>音画 ch${c.channel}</td>
      <td>${c.corrected_ms.toFixed(2)}</td><td>${c.ci95_half_ms.toFixed(2)}</td>
      <td>${bad ? "⚠ 异常" : "在区间内"}</td></tr>`;
  });
  corrected.channels.forEach((c) => {
    html += `<tr><td>${c.kind === "video" ? "视频" : "音频 ch" + c.channel}
      （绝对）</td><td>${c.corrected_ms.toFixed(2)}</td>
      <td colspan="2">已应用 ${c.applied_ms.toFixed(2)} ms</td></tr>`;
  });
  html += "</table>";
  if (anomalies && anomalies.length) {
    html += anomalies.map((a) => `<div class="reason">⚠ ${a}</div>`).join("");
  }
  $("corrected-offsets").innerHTML = html;
}

function renderValidationLog() {
  const log = state.validationLog || [];
  if (!log.length) { $("validation-log").innerHTML = ""; return; }
  let html = "<h3>校准前后独立验证记录</h3><table><tr><th>时间</th>" +
    "<th>应用延迟 (ms)</th><th>通道</th><th>留出验证 RMS 前→后 (ms)</th>" +
    "<th>异常</th></tr>";
  log.forEach((e) => {
    const hv = e.holdout_validation || {};
    const keys = Object.keys(hv);
    const first = keys.length ? hv[keys[0]] : null;
    html += `<tr><td>${e.time}</td>
      <td>${JSON.stringify(e.applied_delay)}</td>
      <td>${keys.join(", ") || "—"}</td>
      <td>${first && first.rms_after_ms != null
            ? `${first.rms_before_ms.toFixed(2)} → ${first.rms_after_ms.toFixed(2)}`
            : "—"}</td>
      <td>${(e.anomalies || []).length ? e.anomalies.join("；") : "无"}</td></tr>`;
  });
  $("validation-log").innerHTML = html + "</table>";
}

// ---- canvas plots --------------------------------------------------------
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth - 36;
  cv.width = w * dpr; cv.style.width = w + "px";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w, h: cv.height };
}

function appliedShift(kind, channel) {
  // applied delay (s) by which this path's playback markers move
  const key = kind === "video" ? "video_delay_ms" : `audio_ch${channel}_delay_ms`;
  return (state.applied[key] || 0) / 1000;
}

function plotAll(series) {
  if (!series) return;
  plotWave(series);
  plotBrightness(series);
}

function plotWave(series) {
  const cv = $("wave-canvas");
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const colors = ["#4cc9f0", "#f78c6b"];
  const wf = series.waveforms[0];
  const tmax = wf.t[wf.t.length - 1];
  const x = (t) => (t / tmax) * w;
  const mid = h / 2;
  ctx.strokeStyle = "#2a3442";
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  series.waveforms.forEach((ch, ci) => {
    ctx.strokeStyle = colors[ci % 2];
    ctx.lineWidth = 1;
    ctx.beginPath();
    ch.t.forEach((t, i) => {
      const px = x(t), py = mid - ch.v[i] * (h * 0.42);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
  });
  // audio detection markers (all channels overlaid)
  const marks = [];
  series.audio_detections.forEach((dets) =>
    dets.forEach((d) => !d.rejected && marks.push(d)));
  drawMarks(ctx, marks, x, h, "audio");
  // corrected playback markers: shifted by the applied delay (real arithmetic)
  marks.forEach((d) => {
    const sh = appliedShift("audio", d.channel);
    if (sh) drawCorrected(ctx, x(d.time + sh), h);
  });
  cv._marks = marks; cv._x = x;
}

function plotBrightness(series) {
  const cv = $("bright-canvas");
  const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const b = series.brightness;
  const tmax = b.t[b.t.length - 1];
  const x = (t) => (t / tmax) * w;
  const y = (v) => h - v * (h - 24) - 10;
  ctx.strokeStyle = "#b8c4d4";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  b.t.forEach((t, i) => {
    const px = x(t), py = y(b.v[i]);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
  const marks = series.video_detections.filter((d) => !d.rejected);
  // frame brackets: show the observation interval explicitly
  marks.forEach((d) => {
    ctx.fillStyle = "rgba(239,71,111,0.18)";
    ctx.fillRect(x(d.lo), 0, Math.max(2, x(d.hi) - x(d.lo)), h);
    ctx.strokeStyle = "#ef476f";
    ctx.beginPath();
    ctx.moveTo(x(d.lo), 0); ctx.lineTo(x(d.lo), h);
    ctx.moveTo(x(d.hi), 0); ctx.lineTo(x(d.hi), h);
    ctx.stroke();
  });
  drawMarks(ctx, marks, x, h, "video");
  marks.forEach((d) => {
    const sh = appliedShift("video", -1);
    if (sh) drawCorrected(ctx, x(d.time + sh), h);
  });
  cv._marks = marks; cv._x = x;
}

function drawMarks(ctx, marks, x, h, kind) {
  marks.forEach((d) => {
    ctx.fillStyle = d.sync ? "#ffd166" : kind === "video" ? "#ef476f" : "#4cc9f0";
    const px = x(d.time);
    ctx.beginPath();
    ctx.moveTo(px, 0); ctx.lineTo(px + 5, 8); ctx.lineTo(px - 5, 8);
    ctx.closePath(); ctx.fill();
  });
}

function drawCorrected(ctx, px, h) {
  ctx.fillStyle = "#57d98a";
  ctx.beginPath();
  ctx.moveTo(px, h); ctx.lineTo(px + 5, h - 8); ctx.lineTo(px - 5, h - 8);
  ctx.closePath(); ctx.fill();
}

// ---- event selection -----------------------------------------------------
["wave-canvas", "bright-canvas"].forEach((id) => {
  $(id).addEventListener("click", (e) => {
    const cv = $(id);
    if (!cv._marks) return;
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    let best = null, bestDist = 12;
    cv._marks.forEach((d) => {
      const dist = Math.abs(cv._x(d.time) - px);
      if (dist < bestDist) { bestDist = dist; best = d; }
    });
    if (best) showDetail(best, id);
  });
});

function showDetail(d, source) {
  state.selected = d;
  const width = (d.hi - d.lo) * 1000;
  $("event-detail").innerHTML = `
    <table>
      <tr><th>来源</th><td>${source === "wave-canvas" ? "音频脉冲" : "视频闪光"}</td></tr>
      <tr><th>通道</th><td>${d.channel === undefined || d.channel < 0 ? "—" : d.channel}</td></tr>
      <tr><th>SYNC 锚点</th><td>${d.sync ? "是" : "否"}</td></tr>
      <tr><th>检测时刻</th><td>${(d.time * 1000).toFixed(3)} ms</td></tr>
      <tr><th>观测区间</th><td>${(d.lo * 1000).toFixed(3)} ~ ${(d.hi * 1000).toFixed(3)} ms（宽 ${width.toFixed(2)} ms）</td></tr>
      <tr><th>强度分数</th><td>${d.score.toFixed(3)}</td></tr>
      <tr><th>精度说明</th><td>${source === "bright-canvas"
        ? "受帧率限制，闪光时刻只落在两帧边界之间，不宣称亚帧精度。"
        : "采样级互相关+亚样本插值，不确定性约为亚毫秒。"}</td></tr>
    </table>`;
}

function renderMedia(media) {
  if (!media) return;
  $("media-links").innerHTML =
    `<a class="media-link" href="${media.mp4}" target="_blank">▶ 校准 MP4</a>
     <a class="media-link" href="${media.wav}" target="_blank">♪ 校准 WAV</a>`;
}

// restore last state on load
fetch("/api/state").then((r) => r.json()).then((s) => {
  state.applied = s.applied_delay || {};
  state.validationLog = s.validation_log || [];
  if (s.job && s.job.status === "done") { state.job = s.job; render(); }
  renderValidationLog();
});
