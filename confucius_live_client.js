const API = "/api/confucius";
const WS_PROTOCOL = location.protocol === "https:" ? "wss:" : "ws:";
const MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024;
const MAX_FILE_SECONDS = 30 * 60;
const MAX_FILE_PCM_BYTES = 128 * 1024 * 1024;
const MAX_FILE_METADATA_BYTES = 16 * 1024;
const byId = (id) => document.getElementById(id);
let active = null;
let fileBusy = false;

function setStatus(text, tone = "idle") {
  const element = byId("confucius-live-status");
  if (!element) return;
  element.textContent = text;
  element.dataset.tone = tone;
}

function setButtons(running, settling = false) {
  const start = byId("confucius-live-start");
  const stop = byId("confucius-live-stop");
  const cancel = byId("confucius-live-cancel");
  const fileInput = byId("confucius-file-input");
  const fileTranscribe = byId("confucius-file-transcribe");
  if (start) start.disabled = running || fileBusy;
  if (stop) stop.disabled = !running || settling;
  if (cancel) cancel.disabled = !running || settling;
  if (fileInput) fileInput.disabled = running || fileBusy;
  if (fileTranscribe) fileTranscribe.disabled = running || fileBusy;
}

function writeCaption(stable, preview = "") {
  if (byId("confucius-live-stable")) byId("confucius-live-stable").textContent = stable || "";
  if (byId("confucius-live-preview")) byId("confucius-live-preview").textContent = preview || "";
}

function withoutClearedPrefix(value, prefix) {
  const text = String(value || "");
  const cleared = String(prefix || "");
  if (!cleared || !text.startsWith(cleared)) return text;
  return text.slice(cleared.length).replace(/^\s+/, "");
}

function writeSessionCaption(state, stable, preview = "") {
  state.rawStable = String(stable || "");
  state.rawPreview = String(preview || "");
  let visibleStable = withoutClearedPrefix(state.rawStable, state.clearedStable);
  if (visibleStable && state.clearedPreview) {
    visibleStable = withoutClearedPrefix(visibleStable, state.clearedPreview);
  }
  const visiblePreview = withoutClearedPrefix(state.rawPreview, state.clearedPreview);
  writeCaption(visibleStable, visiblePreview);
}

function clearTranscript() {
  if (active) {
    active.clearedStable = String(active.rawStable || "");
    active.clearedPreview = String(active.rawPreview || "");
  }
  writeCaption("", "");
  setStatus(active && !active.stopping ? "字幕已清理；实时识别继续中" : "字幕已清理", "ready");
}

async function jsonFetch(path, options = {}) {
  const response = await fetch(API + path, {cache: "no-store", ...options});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.error || payload;
    throw new Error(detail.message || detail.code || `HTTP ${response.status}`);
  }
  return payload;
}

async function refreshStatus() {
  try {
    const status = await jsonFetch("/status?verify_hash=true");
    const summary = byId("confucius-component-summary");
    if (summary) summary.textContent = status.message || status.code || "";
    const accept = byId("confucius-license-accept");
    if (accept) accept.hidden = !status.available || status.licenseAccepted;
    if (!active) setStatus(status.ready ? "就绪：Confucius4-R2T2" : status.message, status.ready ? "ready" : "warn");
  } catch (error) {
    if (!active) setStatus(`组件检查失败：${error.message}`, "error");
  }
}

function cleanup(state) {
  state.worklet?.disconnect();
  state.source?.disconnect();
  state.silent?.disconnect();
  state.media?.getTracks().forEach((track) => track.stop());
  state.context?.close().catch(() => {});
  state.worklet = state.source = state.silent = state.media = state.context = null;
}

function requireOpen(state) {
  if (state.ws?.readyState !== WebSocket.OPEN || state.stopping) {
    throw state.connectionError || new Error("实时连接已关闭");
  }
}

function sendPcm(state, pcm) {
  if (state.ws?.readyState !== WebSocket.OPEN || state.stopping) return;
  if (state.sent - state.acked > 3 * 16000 || state.ws.bufferedAmount > 3 * 16000 * 4) {
    state.connectionError = new Error("音频积压超过 3 秒");
    setStatus(`麦克风已停止：${state.connectionError.message}`, "error");
    cleanup(state);
    state.stopping = true;
    setButtons(true, true);
    state.ws.send(JSON.stringify({type: "cancel"}));
    return;
  }
  const payload = new ArrayBuffer(8 + pcm.byteLength);
  const view = new DataView(payload);
  view.setUint32(0, state.seq, true);
  view.setUint32(4, state.sent, true);
  new Float32Array(payload, 8).set(pcm);
  state.ws.send(payload);
  state.sent += pcm.length;
  state.seq++;
}

async function cancelHttp(state) {
  if (!state.sid || !state.browserToken) return false;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`${API}/live/${encodeURIComponent(state.sid)}/cancel`, {
      method: "POST",
      headers: {"X-R2T2-Session-Token": state.browserToken},
      signal: controller.signal,
    });
    return response.ok;
  } catch (_) {
    // The authenticated socket may already have cancelled the worker session.
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function start() {
  if (active || fileBusy) return;
  const state = {seq: 0, sent: 0, acked: 0, stopping: false, finalizing: false,
    worklet: null, source: null, silent: null, media: null, context: null, ws: null,
    flushed: null, connectionError: null, cancelRequested: false, terminalReceived: false,
    rawStable: "", rawPreview: "", clearedStable: "", clearedPreview: ""};
  active = state;
  setButtons(true);
  setStatus("正在加载 Confucius Q8 模型…", "busy");
  try {
    const created = await jsonFetch("/live/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        language: byId("confucius-live-language")?.value || "Auto",
        context: byId("confucius-live-context")?.value || "",
        hotwords: byId("confucius-live-hotwords")?.value || "",
        stream_chunk_ms: Number(byId("confucius-live-chunk")?.value || 320),
        min_segment_seconds: Number(byId("confucius-live-min-segment")?.value || 8),
      }),
    });
    state.sid = created.session_id;
    state.browserToken = created.browser_token;
    if (state.cancelRequested) {
      state.stopping = true;
      const cancelled = await cancelHttp(state);
      state.terminalReceived = cancelled;
      if (active === state) active = null;
      setButtons(false);
      setStatus(cancelled ? "已取消" : "取消确认失败；会话将由服务自动清理", cancelled ? "idle" : "error");
      return;
    }
    const ws = new WebSocket(`${WS_PROTOCOL}//${location.host}${API}/live/${encodeURIComponent(state.sid)}/stream`);
    state.ws = ws;
    ws.binaryType = "arraybuffer";
    const ready = new Promise((resolve, reject) => {
      let readyResolved = false;
      ws.onopen = () => ws.send(JSON.stringify({browser_token: state.browserToken}));
      ws.onerror = () => reject(state.connectionError ||= new Error("WebSocket 连接失败"));
      ws.onmessage = (message) => {
        let event;
        try {
          event = JSON.parse(message.data);
        } catch (_) {
          state.connectionError = new Error("服务返回了无法解析的实时字幕消息");
          state.terminalReceived = true;
          state.stopping = true;
          cleanup(state);
          setButtons(true, true);
          setStatus(`协议错误：${state.connectionError.message}`, "error");
          if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: "cancel"}));
          if (!readyResolved) reject(state.connectionError);
          return;
        }
        if (event.type === "ready") { readyResolved = true; resolve(); }
        if (event.type === "ack") {
          state.acked = Number(event.ack_sample || 0);
          for (const update of event.events || []) writeSessionCaption(state, update.stable_text, update.preview_text);
          setStatus(state.sent - state.acked > 16000 ? "录音中（积压超过 1 秒）" : "录音中", "recording");
        }
        if (event.type === "final") {
          writeSessionCaption(state, event.text || "", "");
          setStatus(event.truncated ? "已完成但达到输出上限，请复核文本" :
            event.quality_status === "standard" ? "识别完成" : "识别完成，存在强制分段，请复核", "ready");
          state.stopping = true;
          state.terminalReceived = true;
        }
        if (event.type === "error" || event.type === "cancelled") {
          const reason = event.message || event.code || "已取消";
          if (event.type === "error") state.connectionError = new Error(reason);
          const failed = event.type === "error" || Boolean(state.connectionError);
          setStatus(`${failed ? "错误" : "已取消"}：${state.connectionError?.message || reason}`, failed ? "error" : "idle");
          state.stopping = true;
          state.terminalReceived = true;
          if (!readyResolved) reject(state.connectionError);
        }
      };
      ws.onclose = () => {
        cleanup(state);
        if (!state.terminalReceived) {
          const phase = state.finalizing ? "完成结果前" : state.cancelRequested ? "取消确认前" : "";
          setStatus(`${phase ? `${phase}连接中断` : "连接中断"}${state.connectionError ? `：${state.connectionError.message}` : ""}`, "error");
        }
        if (active === state) active = null;
        setButtons(false);
        if (!readyResolved) reject(state.connectionError || new Error("麦克风就绪前连接已关闭"));
      };
    });
    await ready;
    if (state.cancelRequested) {
      state.stopping = true;
      setButtons(true, true);
      ws.send(JSON.stringify({type: "cancel"}));
      return;
    }
    requireOpen(state);
    state.media = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    }});
    requireOpen(state);
    state.context = new AudioContext();
    await Promise.race([
      state.context.audioWorklet.addModule(`${API}/assets/r2t2-worklet.js`),
      new Promise((_, reject) => setTimeout(() => reject(new Error("AudioWorklet 加载超时")), 10000)),
    ]);
    requireOpen(state);
    state.source = state.context.createMediaStreamSource(state.media);
    state.worklet = new AudioWorkletNode(state.context, "t8-confucius-capture");
    state.silent = state.context.createGain();
    state.silent.gain.value = 0;
    state.worklet.port.onmessage = (message) => {
      if (message.data?.type === "pcm") sendPcm(state, message.data.samples);
      if (message.data?.type === "flushed") state.flushed?.();
    };
    state.source.connect(state.worklet);
    state.worklet.connect(state.silent).connect(state.context.destination);
    setStatus("录音中", "recording");
  } catch (error) {
    state.stopping = true;
    cleanup(state);
    state.ws?.close();
    const cancelled = await cancelHttp(state);
    state.terminalReceived = state.terminalReceived || cancelled;
    if (active === state) active = null;
    setButtons(false);
    if (state.cancelRequested && !state.connectionError) setStatus("已取消", "idle");
    else setStatus(`启动失败：${error.message}`, "error");
  }
}

async function decodeAudioFile(file) {
  if (!file) throw new Error("请先选择音频文件");
  if (file.size <= 0) throw new Error("所选音频文件为空");
  if (file.size > MAX_SOURCE_FILE_BYTES) throw new Error("音频文件超过 256 MiB 上限");
  const DecodeContext = window.AudioContext || window.webkitAudioContext;
  const RenderContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!DecodeContext || !RenderContext) throw new Error("当前浏览器不支持本地音频解码");
  const decoder = new DecodeContext();
  let decoded;
  try {
    decoded = await decoder.decodeAudioData(await file.arrayBuffer());
  } finally {
    await decoder.close().catch(() => {});
  }
  if (!Number.isFinite(decoded.duration) || decoded.duration <= 0) throw new Error("无法读取有效音频时长");
  if (decoded.duration > MAX_FILE_SECONDS) throw new Error("音频超过 30 分钟上限");
  const frames = Math.max(1, Math.ceil(decoded.duration * 16000));
  const renderer = new RenderContext(1, frames, 16000);
  const source = renderer.createBufferSource();
  source.buffer = decoded;
  source.connect(renderer.destination);
  source.start(0);
  const rendered = await renderer.startRendering();
  const pcm = new Float32Array(rendered.getChannelData(0));
  if (!pcm.length || pcm.byteLength > MAX_FILE_PCM_BYTES) throw new Error("解码后的音频超过本地转写上限");
  return pcm;
}

async function transcribeFile() {
  if (active) return setStatus("请先停止实时字幕，再转写音频文件", "warn");
  if (fileBusy) return;
  const file = byId("confucius-file-input")?.files?.[0];
  fileBusy = true;
  setButtons(false, true);
  setStatus("正在本地解码音频…", "busy");
  try {
    const pcm = await decodeAudioFile(file);
    const metadata = new TextEncoder().encode(JSON.stringify({
      language: byId("confucius-live-language")?.value || "Auto",
      context: byId("confucius-live-context")?.value || "",
      hotwords: byId("confucius-live-hotwords")?.value || "",
    }));
    if (metadata.byteLength > MAX_FILE_METADATA_BYTES) {
      throw new Error("内容背景和重点词过长，请精简后重试");
    }
    const payload = new ArrayBuffer(4 + metadata.byteLength + pcm.byteLength);
    const view = new DataView(payload);
    view.setUint32(0, metadata.byteLength, true);
    new Uint8Array(payload, 4, metadata.byteLength).set(metadata);
    new Uint8Array(payload, 4 + metadata.byteLength).set(
      new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)
    );
    setStatus("正在使用本地模型转写音频…", "busy");
    const result = await jsonFetch("/file/transcribe", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream"},
      body: payload,
    });
    writeCaption(result.text || "", "");
    const backend = result.backend === "confucius_r2t2" ? "Confucius4-R2T2" : "Whisper 回退";
    setStatus(`文件转写完成（${backend}）`, "ready");
  } catch (error) {
    setStatus(`文件转写失败：${error.message}`, "error");
  } finally {
    fileBusy = false;
    setButtons(Boolean(active), Boolean(active && (active.stopping || active.finalizing)));
  }
}

async function stop(cancel = false) {
  const state = active;
  if (!state || state.stopping || state.finalizing) return;
  if (state.ws?.readyState !== WebSocket.OPEN) {
    state.cancelRequested = true;
    setButtons(true, true);
    setStatus("正在取消启动…", "busy");
    return;
  }
  if (!state.worklet) {
    state.cancelRequested = true;
    state.stopping = true;
    setButtons(true, true);
    state.ws.send(JSON.stringify({type: "cancel"}));
    setStatus("正在取消尚未开始的录音…", "busy");
    return;
  }
  if (cancel) {
    state.cancelRequested = true;
    state.stopping = true;
    cleanup(state);
    setButtons(true, true);
    state.ws.send(JSON.stringify({type: "cancel"}));
    setStatus("正在取消…", "busy");
    return;
  }
  state.finalizing = true;
  setButtons(true, true);
  setStatus("正在完成最后一段…", "busy");
  try {
    if (!state.worklet) throw new Error("麦克风尚未开始录音");
    await Promise.race([
      new Promise((resolve) => { state.flushed = resolve; state.worklet.port.postMessage({type: "flush"}); }),
      new Promise((_, reject) => setTimeout(() => reject(new Error("麦克风刷新超时")), 5000)),
    ]);
    requireOpen(state);
    cleanup(state);
    state.finalizing = false;
    state.stopping = true;
    state.ws.send(JSON.stringify({type: "finish", last_seq: state.seq - 1, total_samples: state.sent}));
  } catch (error) {
    state.finalizing = false;
    state.stopping = true;
    cleanup(state);
    if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({type: "cancel"}));
    setStatus(`结束失败：${error.message}`, "error");
  } finally {
    state.flushed = null;
  }
}

function downloadTranscript() {
  const text = byId("confucius-live-stable")?.textContent || "";
  if (!text.trim()) return setStatus("当前没有可保存的识别文本", "warn");
  const blob = new Blob([text], {type: "text/plain;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `Confucius-ASR-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function bind() {
  if (!byId("confucius-live-start") || byId("confucius-live-start").dataset.bound) return;
  byId("confucius-live-start").dataset.bound = "1";
  byId("confucius-live-start").addEventListener("click", start);
  byId("confucius-live-stop")?.addEventListener("click", () => stop(false));
  byId("confucius-live-cancel")?.addEventListener("click", () => stop(true));
  byId("confucius-live-copy")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(byId("confucius-live-stable")?.textContent || "");
      setStatus("识别文本已复制", "ready");
    } catch (error) {
      setStatus(`复制失败：${error.message || "系统剪贴板不可用"}`, "error");
    }
  });
  byId("confucius-live-download")?.addEventListener("click", downloadTranscript);
  byId("confucius-live-clear")?.addEventListener("click", clearTranscript);
  byId("confucius-file-transcribe")?.addEventListener("click", transcribeFile);
  byId("confucius-file-input")?.addEventListener("change", () => {
    const file = byId("confucius-file-input")?.files?.[0];
    if (file && !active && !fileBusy) setStatus(`已选择：${file.name}`, "idle");
  });
  byId("confucius-component-refresh")?.addEventListener("click", refreshStatus);
  byId("confucius-license-accept")?.addEventListener("click", async () => {
    try {
      await jsonFetch("/license", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({accepted: true})});
      await refreshStatus();
    } catch (error) {
      setStatus(`许可保存失败：${error.message}`, "error");
    }
  });
  setButtons(false);
  refreshStatus();
}

const observer = new MutationObserver(bind);
observer.observe(document.documentElement, {childList: true, subtree: true});
bind();
window.addEventListener("beforeunload", () => {
  if (active) {
    cleanup(active);
    if (active.sid && active.browserToken && !active.terminalReceived) {
      fetch(`${API}/live/${encodeURIComponent(active.sid)}/cancel`, {
        method: "POST",
        headers: {"X-R2T2-Session-Token": active.browserToken},
        keepalive: true,
      }).catch(() => {});
    }
    active.ws?.close();
  }
});
