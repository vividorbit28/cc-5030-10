"use strict";

const $ = (id) => document.getElementById(id);
let state = { job: null, applied: {}, selected: null };

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
  return {
    n_events: num("n_events"), source_fps: num("source_fps"),
    recorded_fps: num("recorded_fps"), channels,
    audio_delay: [num("ad0") / 1000, num("ad1") / 1000].slice(0, channels),
    video_delay: num("vd") / 1000, audio_noise: num("audio_noise"),
    brightness_noise: num("brightness_noise"),
    exposure_drift: num("exposure_drift"),
    echo_delay: num("echo_delay") / 1000, echo_gain: num("echo_gain"),
    dropped_events: drops,
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
  plotAll(job.series);
  renderMedia(job.media);
}

function metricCard(title, rows) {
  return `<div class="metric-card"><h3>${title}</h3><div class="metric-row">` +
    rows.map(([k, val]) => `<span>${k} <b>${val}</b></span>`).join("") +
    "</div></div>";
}
const ms = (s) => (s * 1000).toFixed(2);

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
    fb.className = "status ok";
    fb.textContent = "已应用修正（内核确认）：" + JSON.stringify(r.applied_delay);
  } else {
    fb.className = "status error";
    fb.textContent = "被内核拒绝：" + r.error;
  }
});

// ---- canvas plots --------------------------------------------------------
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth - 36;
  cv.width = w * dpr; cv.style.width = w + "px";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w, h: cv.height };
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
  if (s.job && s.job.status === "done") { state.job = s.job; render(); }
});
