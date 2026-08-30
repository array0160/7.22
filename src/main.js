import "./style.css";
import * as ort from "onnxruntime-web/webgpu";
import {
  DetectionService,
  RecognitionService,
  TextImageUnwarpingService,
  getTextDetectionPresetOptions,
  getTextImageUnwarpingPresetOptions,
  getTextRecognitionPresetOptions,
  normalizeInputToRgb,
} from "paddleocr";

const UVDOC_URL =
  "https://huggingface.co/PaddlePaddle/UVDoc_onnx/resolve/main/inference.onnx";
const DET_URL =
  "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx";

const REC_MODELS = {
  server: {
    label: "PP-OCRv5 server recognizer",
    preset: "PP-OCRv5_server_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx/resolve/main/inference.onnx",
  },
  mobile: {
    label: "PP-OCRv5 mobile recognizer",
    preset: "PP-OCRv5_mobile_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx",
  },
};

const V6_RESCUE_MODELS = {
  small6: {
    key: "small6",
    label: "PP-OCRv6 small recognizer",
    preset: "PP-OCRv6_small_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx",
  },
  medium6: {
    key: "medium6",
    label: "PP-OCRv6 medium recognizer",
    preset: "PP-OCRv6_medium_rec",
    url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx",
  },
};

const V6_DICT_URL =
  "https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/ppocr/utils/dict/ppocrv6_dict.txt";

const DIRECT_V6_DET = {
  label: "PP-OCRv6 small detector · Direct",
  preset: "PP-OCRv6_small_det",
  url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx",
};

const DIRECT_V6_REC = {
  label: "PP-OCRv6 small recognizer · Direct",
  preset: "PP-OCRv6_small_rec",
  url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx",
};

const DICT_URL =
  "https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/ppocr/utils/dict/ppocrv5_dict.txt";

const CNN_SR_MODEL = {
  label: "Real-ESRGAN general x4v3",
  // Pinned, browser-friendly ONNX redistribution of the official BSD-3 model.
  // Input: float32 [1,3,H,W] RGB 0..1; output: [1,3,4H,4W].
  url: "https://cdn.jsdelivr.net/gh/NovareOrbis/nova-ai-models@v3/realesr-general-x4v3.onnx",
};

const MODEL_CACHE = "bookocr-models-v7-2";
const MAX_PAGE_SIDE = 2400;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  bitmap: null,
  rightInput: null,
  leftInput: null,
  rightFlat: null,
  leftFlat: null,
  uvdoc: null,
  uvSession: null,
  detector: null,
  detSession: null,
  rightDetection: null,
  leftDetection: null,
  rightStrips: [],
  leftStrips: [],
  recognizer: null,
  recSession: null,
  recPreset: null,
  recDictionary: null,
  rightRecognition: [],
  leftRecognition: [],
  webGpuAvailable: null,
  backends: {},
  recModelBuffer: null,
  recSpec: null,
  recForcedWasm: false,
  fullInput: null,
  routedMode: null,
  routerAnalysis: null,
  generalPages: [],
  generalRecognition: [],
  excludedColumns: {
    right: new Set(),
    left: new Set(),
  },
  overlayHitboxes: {
    right: [],
    left: [],
  },
  rescueSuggestionHitboxes: {
    right: [],
    left: [],
  },
  selectedColumn: null,
  columnOverrides: {
    right: new Map(),
    left: new Map(),
  },
  editorTimer: null,
  editorRunToken: 0,
  overlayDrag: null,

  // v6.0 selective CNN small-text rescue.
  srSession: null,
  srModelBuffer: null,
  srForcedWasm: false,
  recPerf: null,

  // v6.1: independent heavy recognizer used only for hard-case comparison.
  serverRescueRecognizer: null,
  serverRescueSession: null,
  serverRescueModelBuffer: null,

  // v6.2: latest PP-OCRv6 hard-case recognizer.
  v6RescueRecognizer: null,
  v6RescueSession: null,
  v6RescueModelBuffer: null,
  v6RescueDictionary: null,
  v6RescueKey: null,

  // v7.0: standard OCR geometry path.
  // Detector quads are fed directly to RecognitionService; no PCA/V3 crop.
  directV6Detector: null,
  directV6Recognizer: null,
  directV6DetSession: null,
  directV6RecSession: null,
  directV6DetBuffer: null,
  directV6RecBuffer: null,
  directV6Dictionary: null,
  directV6ForcedWasm: false,
  directRightResults: [],
  directLeftResults: [],
  directRightColumns: [],
  directLeftColumns: [],
};

let webGpuProbePromise = null;

async function probeWebGpu() {
  if (state.webGpuAvailable !== null) {
    return state.webGpuAvailable;
  }

  if (!("gpu" in navigator)) {
    state.webGpuAvailable = false;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    state.webGpuAvailable = !!adapter;
    return state.webGpuAvailable;
  } catch (error) {
    console.warn("WebGPU probe failed:", error);
    state.webGpuAvailable = false;
    return false;
  }
}

function updateRuntimeBadge() {
  const badge = $("runtimeBadge");
  const values = Object.values(state.backends);

  if (values.includes("webgpu")) {
    badge.textContent = "GPU / WebGPU";
    badge.title =
      "至少一個 ONNX 模型已用 WebGPU session 啟動；不支援的節點可由 WASM fallback。";
    return;
  }

  if (values.includes("wasm")) {
    badge.textContent = "CPU / WASM";
    badge.title = "目前模型使用 WASM/CPU。";
    return;
  }

  if (state.webGpuAvailable === true) {
    badge.textContent = "WebGPU 可用";
    badge.title = "下一個模型 session 會優先嘗試 GPU。";
  } else if (state.webGpuAvailable === false) {
    badge.textContent = "CPU / WASM";
    badge.title = "目前瀏覽器沒有可用的 WebGPU adapter。";
  } else {
    badge.textContent = "偵測 GPU…";
  }
}

function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

function freshRecPerf() {
  return {
    startedAt: performance.now(),
    cnnRuns: 0,
    cnnInferenceMs: 0,
    cnnSetupMs: 0,
    cnnFailures: 0,
    serverRuns: 0,
    serverInferenceMs: 0,
    serverSetupMs: 0,
    serverFailures: 0,
    v6Runs: 0,
    v6InferenceMs: 0,
    v6SetupMs: 0,
    v6Failures: 0,
  };
}

function formatDuration(ms) {
  const value = Math.max(0, Number(ms) || 0);
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} 秒`;
}

function updateRecTiming(totalMs = null) {
  const el = $("recTiming");
  if (!el) return;

  const perf = state.recPerf;
  if (!perf) {
    el.textContent = "上次 Recognition：尚未執行";
    return;
  }

  const total =
    totalMs === null
      ? performance.now() - perf.startedAt
      : totalMs;

  const parts = [
    `總計 ${formatDuration(total)}`,
  ];

  if (perf.v6Runs > 0 || perf.v6SetupMs > 0) {
    parts.push(
      `PP-OCRv6 ${perf.v6Runs} 欄 / ${formatDuration(perf.v6InferenceMs)}`,
    );
  }

  if (perf.v6SetupMs > 0) {
    parts.push(
      `v6 首次載入/建 session ${formatDuration(perf.v6SetupMs)}`,
    );
  }

  if (perf.v6Failures > 0) {
    parts.push(`v6 fallback ${perf.v6Failures} 次`);
  }

  if (perf.serverRuns > 0 || perf.serverSetupMs > 0) {
    parts.push(
      `Server ${perf.serverRuns} 欄 / ${formatDuration(perf.serverInferenceMs)}`,
    );
  }

  if (perf.serverSetupMs > 0) {
    parts.push(
      `Server 首次載入/建 session ${formatDuration(perf.serverSetupMs)}`,
    );
  }

  if (perf.serverFailures > 0) {
    parts.push(`Server fallback ${perf.serverFailures} 次`);
  }

  if (perf.cnnRuns > 0 || perf.cnnSetupMs > 0) {
    parts.push(`CNN ${perf.cnnRuns} 欄 / ${formatDuration(perf.cnnInferenceMs)}`);
  }

  if (perf.cnnSetupMs > 0) {
    parts.push(`CNN 首次載入/建 session ${formatDuration(perf.cnnSetupMs)}`);
  }

  if (perf.cnnFailures > 0) {
    parts.push(`CNN fallback ${perf.cnnFailures} 次`);
  }

  el.textContent = `本次 Recognition：${parts.join(" · ")}`;
}

async function createSessionAuto(modelBuffer, modelName) {
  const canGpu = await probeWebGpu();
  const mb = (modelBuffer.byteLength / 1024 / 1024).toFixed(1);

  if (canGpu) {
    try {
      setStatus(
        `建立 ${modelName} GPU session…`,
        `模型 ${mb} MB 已下載完成；正在 WebGPU 編譯與配置權重。`,
        86,
      );
      await nextPaint();

      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });

      state.backends[modelName] = "webgpu";
      updateRuntimeBadge();
      console.info(`${modelName}: WebGPU session ready`);
      return session;
    } catch (error) {
      console.warn(`${modelName}: WebGPU failed; falling back to WASM`, error);
      setStatus(
        `${modelName} GPU 啟動失敗，改用 CPU / WASM…`,
        error instanceof Error ? error.message : String(error),
        86,
      );
      await nextPaint();
    }
  }

  setStatus(
    `建立 ${modelName} CPU session…`,
    `模型 ${mb} MB 已下載完成；正在建立 WASM session。`,
    86,
  );
  await nextPaint();

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.backends[modelName] = "wasm";
  updateRuntimeBadge();
  console.info(`${modelName}: WASM session ready`);
  return session;
}

async function createWasmSession(modelBuffer, modelName) {
  const mb = (modelBuffer.byteLength / 1024 / 1024).toFixed(1);
  setStatus(
    `建立 ${modelName} CPU / WASM session…`,
    `模型 ${mb} MB；正在切換到 CPU fallback。`,
    88,
  );
  await nextPaint();

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  state.backends[modelName] = "wasm";
  updateRuntimeBadge();
  return session;
}

function setStatus(text, detail = "", progress = null) {
  $("status").textContent = text;
  $("detail").textContent = detail;
  if (progress != null) {
    $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function showCanvas(canvasId, emptyId) {
  $(canvasId).style.display = "block";
  $(emptyId).style.display = "none";
}

function clearCanvas(canvasId, emptyId, text) {
  const c = $(canvasId);
  c.width = 1;
  c.height = 1;
  c.style.display = "none";
  $(emptyId).style.display = "block";
  $(emptyId).textContent = text;
}

function drawBitmapCropToCanvas(bitmap, sx, sy, sw, sh, canvas) {
  const scale = Math.min(1, MAX_PAGE_SIDE / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
}

function canvasToPixels(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    width: canvas.width,
    height: canvas.height,
    data: new Uint8Array(
      imageData.data.buffer.slice(
        imageData.data.byteOffset,
        imageData.data.byteOffset + imageData.data.byteLength,
      ),
    ),
  };
}

function pixelsToCanvas(image, canvas) {
  const { width, height, data } = image;
  canvas.width = width;
  canvas.height = height;

  let rgba;
  if (data.length === width * height * 4) {
    rgba = new Uint8ClampedArray(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    );
  } else if (data.length === width * height * 3) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let si = 0, di = 0; si < data.length; ) {
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = data[si++];
      rgba[di++] = 255;
    }
  } else if (data.length === width * height) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i++) {
      const v = data[i];
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = v;
      rgba[j++] = 255;
    }
  } else {
    throw new Error(
      `未知像素格式：${data.length} bytes for ${width}x${height}`,
    );
  }

  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
}

async function fetchArrayBufferCached(url, modelName, progressBase, progressSpan) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;

  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      setStatus(`載入 ${modelName}…`, `${modelName} 已從瀏覽器快取取得。`, progressBase + progressSpan);
      return cached.arrayBuffer();
    }
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`${modelName} 下載失敗：HTTP ${response.status}`);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    if (cache) {
      await cache.put(url, new Response(buffer.slice(0)));
    }
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;

    const ratio = total ? received / total : 0.2;
    setStatus(
      `下載 ${modelName}…`,
      total
        ? `${(received/1024/1024).toFixed(1)} / ${(total/1024/1024).toFixed(1)} MB`
        : `${(received/1024/1024).toFixed(1)} MB`,
      progressBase + progressSpan * ratio,
    );
  }

  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (cache) {
    await cache.put(url, new Response(merged.slice().buffer));
  }
  return merged.buffer;
}

async function ensureUvDoc() {
  if (state.uvdoc) return state.uvdoc;

  const model = await fetchArrayBufferCached(UVDOC_URL, "UVDoc", 2, 45);
  setStatus("建立 UVDoc session…", "ONNX Runtime Web / WASM", 50);

  state.uvSession = await createSessionAuto(model, "UVDoc");

  state.uvdoc = new TextImageUnwarpingService(
    ort,
    state.uvSession,
    getTextImageUnwarpingPresetOptions("UVDoc"),
  );
  return state.uvdoc;
}

const DETECTOR_EDGE_GUARD_PERCENT = 2.5;

function detectorRuntimeOptions() {
  return {
    // These map to the Colab/PaddleOCR values we settled on.
    textPixelThreshold: Number($("pixelThreshold").value),
    boxScoreThreshold: Number($("boxThreshold").value),
    unclipRatio: Number($("unclipRatio").value),
    maxSideLength: Number($("maxSideLength").value),
    limitType: "max",
    maxSideLimit: 4000,

    // Paddle's official PP-OCRv5 inference config decodes BGR.
    channelOrder: "bgr",

    // Keep raw DB boxes as geometry hints. Do not add service-level padding.
    paddingBoxVertical: 0,
    paddingBoxHorizontal: 0,
    minimumAreaThreshold: 20,
    maxCandidates: 1000,
    boxType: "quad",
  };
}

async function ensureDetector() {
  if (state.detector) return state.detector;

  const model = await fetchArrayBufferCached(DET_URL, "PP-OCRv5 mobile detector", 50, 30);
  setStatus("建立 Detector session…", "PP-OCRv5_mobile_det ONNX", 82);

  state.detSession = await createSessionAuto(
    model,
    "PP-OCRv5 Detector",
  );

  state.detector = new DetectionService(
    ort,
    state.detSession,
    detectorRuntimeOptions(),
  );

  return state.detector;
}


function bitmapToPixels(bitmap, maxSide = 2400) {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvasToPixels(canvas);
}

function boxBounds(box) {
  const pts = boxPoints(box);
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
  };
}

async function detectRawImage(image, statusName = "圖片") {
  const detector = await ensureDetector();
  const detectorImage = normalizeInputToRgb(image);

  return detector.run(detectorImage, {
    ...detectorRuntimeOptions(),
    onProgress(event) {
      const stageName = {
        preprocess: "前處理",
        infer: "模型推理",
        postprocess: "後處理",
      }[event.stage] || event.stage;

      setStatus(
        `Auto 分析：${statusName} ${stageName}…`,
        event.detectedCount != null ? `偵測到 ${event.detectedCount} 個文字區塊` : "",
        18,
      );
    },
  });
}

function analyzeLayoutFromBoxes(boxes, image) {
  let verticalWeight = 0;
  let horizontalWeight = 0;
  let mixedWeight = 0;

  for (const box of boxes) {
    const b = boxBounds(box);
    const areaWeight = Math.max(1, Math.sqrt(b.width * b.height));

    if (b.height >= b.width * 1.55) {
      verticalWeight += areaWeight;
    } else if (b.width >= b.height * 1.55) {
      horizontalWeight += areaWeight;
    } else {
      mixedWeight += areaWeight;
    }
  }

  const total = verticalWeight + horizontalWeight + mixedWeight || 1;
  const verticalRatio = verticalWeight / total;
  const horizontalRatio = horizontalWeight / total;
  const mixedRatio = mixedWeight / total;
  const aspect = image.width / Math.max(1, image.height);
  const looksLikeSpread = aspect >= 1.16;

  let mode;
  let reason;

  if (
    verticalRatio >= 0.52 &&
    verticalRatio >= horizontalRatio * 1.22
  ) {
    if (looksLikeSpread) {
      mode = "traditional";
      reason = "文字框大多高瘦，而且圖片接近跨頁比例。";
    } else {
      mode = "vertical-single";
      reason = "文字框大多高瘦，但圖片比較像單頁／單張直排內容。";
    }
  } else if (
    horizontalRatio >= 0.50 &&
    horizontalRatio >= verticalRatio * 1.18
  ) {
    mode = "horizontal";
    reason = looksLikeSpread
      ? "文字框大多橫向，而且圖片像攤開書籍。"
      : "文字框大多橫向，適合一般橫排文件流程。";
  } else {
    mode = "general";
    reason = "文字框方向較混合，先用一般圖片／招牌 OCR。";
  }

  const winning = Math.max(verticalRatio, horizontalRatio, mixedRatio);
  const confidence = Math.round(Math.min(99, 55 + winning * 44));

  return {
    mode,
    confidence,
    verticalRatio,
    horizontalRatio,
    mixedRatio,
    looksLikeSpread,
    boxCount: boxes.length,
    reason,
  };
}

function displayModeName(mode) {
  return {
    auto: "Auto",
    traditional: "傳統直書",
    "vertical-single": "直排單頁",
    horizontal: "橫排書籍／文件",
    general: "一般圖片／招牌",
  }[mode] || mode;
}

function updateModeUi(mode, analysis = null) {
  const isTraditional = mode === "traditional";
  const isBookish = isTraditional || mode === "horizontal";

  $("verticalPipeline").classList.toggle("hidden", !isTraditional);
  $("generalPipeline").classList.toggle("hidden", isTraditional);
  $("verticalSteps").classList.toggle("hidden", !isTraditional);

  document.querySelectorAll(".book-only-control").forEach((el) => {
    el.classList.toggle("dimmed", !isBookish && $("ocrMode").value !== "auto");
  });

  if (mode === "traditional") {
    $("readingOrderLabel").textContent = "右頁 → 左頁；右欄 → 左欄";
    $("readingOrderHint").textContent = "Column 01 現在也會真的顯示在畫面最右邊。";
    $("fullTextOrder").textContent =
      "傳統直書：右頁 → 左頁；每頁最右欄 → 最左欄；每欄上 → 下。";
  } else if (mode === "vertical-single") {
    $("readingOrderLabel").textContent = "最右直欄 → 最左直欄";
    $("readingOrderHint").textContent = "單頁直排不切成左右跨頁。";
    $("fullTextOrder").textContent =
      "直排單頁：文字區塊依右 → 左排序；高瘦區塊會自動旋正後辨識。";
  } else if (mode === "horizontal") {
    $("readingOrderLabel").textContent = "上 → 下；每行左 → 右";
    $("readingOrderHint").textContent = analysis?.looksLikeSpread
      ? "若為跨頁橫排書籍，左頁先於右頁。"
      : "單頁橫排依一般閱讀順序。";
    $("fullTextOrder").textContent =
      analysis?.looksLikeSpread
        ? "橫排跨頁：左頁 → 右頁；每頁上 → 下、左 → 右。"
        : "橫排單頁：上 → 下；同一行左 → 右。";
  } else {
    $("readingOrderLabel").textContent = "一般版面：上 → 下、左 → 右";
    $("readingOrderHint").textContent =
      "招牌／海報等混合版面先依文字區塊位置排序。";
    $("fullTextOrder").textContent =
      "一般圖片：以文字區塊的視覺位置排序；必要時可手動切換模式重跑。";
  }
}

function showRouterAnalysis(analysis, forced = false) {
  $("routerCard").classList.remove("hidden");
  $("routerTitle").textContent =
    `${forced ? "手動模式" : "Auto 判定"}：${displayModeName(analysis.mode)}`;
  $("routerConfidence").textContent =
    forced ? "手動指定" : `信心 ${analysis.confidence}%`;

  const v = Math.round((analysis.verticalRatio || 0) * 100);
  const h = Math.round((analysis.horizontalRatio || 0) * 100);
  const m = Math.round((analysis.mixedRatio || 0) * 100);

  $("routerDetail").textContent =
    forced
      ? analysis.reason
      : `${analysis.reason} Detector：${analysis.boxCount} 區塊；直向 ${v}% / 橫向 ${h}% / 混合 ${m}%。`;
}

async function resolveRunMode() {
  const selected = $("ocrMode").value;

  if (selected !== "auto") {
    const aspect = state.fullInput.width / Math.max(1, state.fullInput.height);
    const forcedAnalysis = {
      mode: selected,
      confidence: 100,
      verticalRatio: selected === "traditional" ? 1 : 0,
      horizontalRatio: selected === "horizontal" ? 1 : 0,
      mixedRatio: selected === "general" ? 1 : 0,
      looksLikeSpread: aspect >= 1.16,
      boxCount: 0,
      reason: `使用者手動指定「${displayModeName(selected)}」。`,
    };
    state.routerAnalysis = forcedAnalysis;
    state.routedMode = selected;
    showRouterAnalysis(forcedAnalysis, true);
    updateModeUi(selected, forcedAnalysis);
    return selected;
  }

  setStatus("Auto：先分析文字方向…", "這一步只做輕量 Detector，不先假設是書。", 5);
  const boxes = await detectRawImage(state.fullInput, "原圖");
  const analysis = analyzeLayoutFromBoxes(boxes, state.fullInput);

  state.routerAnalysis = analysis;
  state.routedMode = analysis.mode;

  showRouterAnalysis(analysis, false);
  updateModeUi(analysis.mode, analysis);
  return analysis.mode;
}

function splitPixelsForBook() {
  const image = state.fullInput;
  const w = image.width;
  const h = image.height;

  const splitPct = Number($("splitRange").value) / 100;
  const gutterPct = Number($("gutterRange").value) / 100;
  const splitX = Math.round(w * splitPct);

  // Horizontal spread mode uses the same spine-overlap rule.
  const halfOverlap = Math.round((w * gutterPct) / 2);
  const leftEnd = Math.min(w, Math.max(1, splitX + halfOverlap));
  const rightStart = Math.max(0, Math.min(w - 1, splitX - halfOverlap));

  function cropPixels(src, x0, y0, cw, ch) {
    const channels =
      src.data.length === src.width * src.height * 4 ? 4 :
      src.data.length === src.width * src.height * 3 ? 3 : 1;
    const data = new Uint8Array(cw * ch * 4);
    let di = 0;

    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const sx = x0 + x;
        const sy = y0 + y;
        const si = (sy * src.width + sx) * channels;

        if (channels === 4) {
          data[di++] = src.data[si];
          data[di++] = src.data[si + 1];
          data[di++] = src.data[si + 2];
          data[di++] = src.data[si + 3];
        } else if (channels === 3) {
          data[di++] = src.data[si];
          data[di++] = src.data[si + 1];
          data[di++] = src.data[si + 2];
          data[di++] = 255;
        } else {
          const v = src.data[sy * src.width + sx];
          data[di++] = v;
          data[di++] = v;
          data[di++] = v;
          data[di++] = 255;
        }
      }
    }

    return { width: cw, height: ch, data };
  }

  const pagePadPercent = Number($("pagePadRange").value);

  return {
    left: addWhitePadding(
      cropPixels(image, 0, 0, leftEnd, h),
      pagePadPercent,
    ),
    right: addWhitePadding(
      cropPixels(image, rightStart, 0, w - rightStart, h),
      pagePadPercent,
    ),
  };
}

function sortBoxesHorizontal(boxes) {
  const items = boxes.map((box) => ({ box, b: boxBounds(box) }));
  if (!items.length) return [];

  const heights = items.map((x) => x.b.height).sort((a, b) => a - b);
  const medianH = heights[Math.floor(heights.length / 2)] || 20;
  const lineTolerance = Math.max(10, medianH * 0.72);

  items.sort((a, b) => a.b.cy - b.b.cy);

  const lines = [];
  for (const item of items) {
    let best = null;
    let bestDist = Infinity;

    for (const line of lines) {
      const dist = Math.abs(item.b.cy - line.cy);
      if (dist < lineTolerance && dist < bestDist) {
        best = line;
        bestDist = dist;
      }
    }

    if (!best) {
      best = { cy: item.b.cy, items: [] };
      lines.push(best);
    }

    best.items.push(item);
    best.cy = best.items.reduce((sum, x) => sum + x.b.cy, 0) / best.items.length;
  }

  lines.sort((a, b) => a.cy - b.cy);

  const out = [];
  for (const line of lines) {
    line.items.sort((a, b) => a.b.cx - b.b.cx);
    out.push(...line.items.map((x) => x.box));
  }
  return out;
}

function sortBoxesVertical(boxes) {
  return [...boxes].sort((a, b) => {
    const aa = boxBounds(a);
    const bb = boxBounds(b);
    const dx = bb.cx - aa.cx;
    if (Math.abs(dx) > Math.max(aa.width, bb.width) * 0.55) return dx;
    return aa.cy - bb.cy;
  });
}

function drawGeneralOverlay(image, boxes, canvas) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(220, 55, 65, .88)";
  ctx.fillStyle = "rgba(25, 105, 215, .96)";
  ctx.lineWidth = Math.max(2, canvas.width / 600);
  ctx.font = `bold ${Math.max(14, canvas.width / 44)}px system-ui`;

  boxes.forEach((box, index) => {
    const pts = boxPoints(box);
    if (!pts.length) return;

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();

    const b = boxBounds(box);
    ctx.fillText(
      String(index + 1),
      Math.max(3, Math.min(canvas.width - 40, b.minX + 3)),
      Math.max(17, b.minY + 16),
    );
  });
}

async function recognizeGeneralBoxes(image, boxes, recognizer, pageName) {
  const results = [];

  for (let i = 0; i < boxes.length; i++) {
    setStatus(
      `${pageName} OCR：區塊 ${i + 1}/${boxes.length}`,
      "PP-OCRv5 recognition",
      55 + 38 * ((i + 1) / Math.max(1, boxes.length)),
    );

    const rec = await recognizer.run(
      normalizeInputToRgb(image),
      [boxes[i]],
      { ordering: { sortByReadingOrder: false } },
    );

    results.push({
      pageName,
      index: i + 1,
      text: rec[0]?.text ?? "",
      confidence: Number(rec[0]?.confidence ?? 0),
    });
  }

  return results;
}

function renderGeneralPages(pages) {
  const root = $("generalPages");
  root.innerHTML = "";

  for (const page of pages) {
    const card = document.createElement("article");
    card.className = "card general-page-card";

    const meta = document.createElement("div");
    meta.className = "general-page-meta";

    const title = document.createElement("h3");
    title.textContent = page.name;

    const pill = document.createElement("span");
    pill.className = "route-pill";
    pill.textContent = `${page.boxes.length} 文字區塊`;

    const canvas = document.createElement("canvas");
    drawGeneralOverlay(page.image, page.boxes, canvas);

    meta.appendChild(title);
    meta.appendChild(pill);
    card.appendChild(meta);
    card.appendChild(canvas);
    root.appendChild(card);
  }
}

function renderGeneralRecognition(items) {
  const root = $("generalRecognition");
  root.innerHTML = "";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "recognition-item";

    const name = document.createElement("div");
    name.className = "col-name";
    name.textContent = `${item.pageName} #${item.index}`;

    const text = document.createElement("div");
    text.className = "rec-text";
    text.textContent = item.text || "（空白）";

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    confidence.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    if (item.rescueRan) {
      const badge = document.createElement("span");
      badge.className = "rescue-badge";
      badge.textContent =
        item.variant === "高解析4×"
          ? "4×救援"
          : item.variant === "高解析2×"
            ? "2×救援"
            : "高解析已比較";
      text.appendChild(badge);
    }

    if (item.cnnRan) {
      const badge = document.createElement("span");
      badge.className =
        item.variant === "CNN4×"
          ? "cnn-rescue-badge selected"
          : "cnn-rescue-badge";
      badge.textContent =
        item.variant === "CNN4×"
          ? "CNN採用"
          : "CNN已比較";
      text.appendChild(badge);
    }

    if (item.serverRan) {
      const badge = document.createElement("span");
      badge.className =
        item.variant === "Server原圖"
          ? "server-rescue-badge selected"
          : "server-rescue-badge";
      badge.textContent =
        item.variant === "Server原圖"
          ? "Server採用"
          : "Server已比較";
      text.appendChild(badge);

      const raw = document.createElement("div");
      raw.className = "server-raw-result";
      raw.textContent =
        `Mobile：${item.mobileText || "（空白）"} ${
          (Number(item.mobileConfidence || 0) * 100).toFixed(1)
        }% · Server：${item.serverText || "（空白）"} ${
          (Number(item.serverConfidence || 0) * 100).toFixed(1)
        }%`;
      text.appendChild(raw);
    }

    if (item.v6Ran) {
      const selectedV6 =
        item.variant === "PP-OCRv6 small" ||
        item.variant === "PP-OCRv6 medium";

      const badge = document.createElement("span");
      badge.className = selectedV6
        ? "v6-rescue-badge selected"
        : "v6-rescue-badge";
      badge.textContent = selectedV6
        ? "v6採用"
        : "v6已比較";
      text.appendChild(badge);

      const raw = document.createElement("div");
      raw.className = "v6-raw-result";
      raw.textContent =
        `v5 Mobile：${item.mobileText || "（空白）"} ${
          (Number(item.mobileConfidence || 0) * 100).toFixed(1)
        }% · ${
          item.v6RescueModeUsed === "medium6"
            ? "v6 Medium"
            : "v6 Small"
        }：${item.v6Text || "（空白）"} ${
          (Number(item.v6Confidence || 0) * 100).toFixed(1)
        }%`;
      text.appendChild(raw);
    }

    row.appendChild(name);
    row.appendChild(text);
    row.appendChild(confidence);

    if (item.columnId) {
      const del = document.createElement("button");
      del.className = "rec-delete";
      del.textContent = "× 刪除此欄";
      del.addEventListener("click", () => removeColumn(side, item.columnId));
      row.appendChild(del);
    }
    root.appendChild(row);
  });

  const nonEmpty = items.filter((x) => x.text.trim()).length;
  $("generalStats").textContent = `${nonEmpty}/${items.length} 區塊有文字`;
}

async function runGeneralPipeline(mode) {
  $("generalPipeline").classList.remove("hidden");
  $("verticalPipeline").classList.add("hidden");

  let pages = [];

  if (mode === "horizontal" && state.routerAnalysis?.looksLikeSpread) {
    const halves = splitPixelsForBook();
    const uv = await ensureUvDoc();

    setStatus("橫排書籍：展平左頁…", "", 30);
    const leftFlat = (await uv.run(halves.left)).doctrImage;

    setStatus("橫排書籍：展平右頁…", "", 38);
    const rightFlat = (await uv.run(halves.right)).doctrImage;

    pages = [
      { name: "左頁", image: leftFlat },
      { name: "右頁", image: rightFlat },
    ];
  } else if (mode === "horizontal") {
    // A single photographed horizontal page can also benefit from UVDoc.
    const uv = await ensureUvDoc();
    setStatus("橫排單頁：UVDoc 展平…", "", 30);
    const padded = addWhitePadding(
      state.fullInput,
      Number($("pagePadRange").value),
    );
    const flat = (await uv.run(padded)).doctrImage;
    pages = [{ name: "單頁", image: flat }];
  } else {
    pages = [{ name: mode === "vertical-single" ? "直排單頁" : "一般圖片", image: state.fullInput }];
  }

  const recognizer = await ensureRecognizer(false);
  const allResults = [];

  for (const page of pages) {
    setStatus(`${page.name}：偵測文字區塊…`, "", 42);
    let boxes = await detectRawImage(page.image, page.name);

    if (mode === "vertical-single") {
      boxes = sortBoxesVertical(boxes);
    } else {
      boxes = sortBoxesHorizontal(boxes);
    }

    page.boxes = boxes;

    const pageResults = await recognizeGeneralBoxes(
      page.image,
      boxes,
      recognizer,
      page.name,
    );

    allResults.push(...pageResults);
  }

  state.generalPages = pages;
  state.generalRecognition = allResults;

  renderGeneralPages(pages);
  renderGeneralRecognition(allResults);

  const textParts = [];
  for (const page of pages) {
    const texts = allResults
      .filter((x) => x.pageName === page.name)
      .map((x) => x.text.trim())
      .filter(Boolean);

    if (texts.length) textParts.push(texts.join("\n"));
  }

  const text = textParts.join("\n\n");
  $("fullText").value = text;
  $("copyTextBtn").disabled = !text;
  $("downloadTextBtn").disabled = !text;

  setStatus(
    "OCR 完成。",
    `使用 ${displayModeName(mode)} 流程；${allResults.length} 個文字區塊。`,
    100,
  );
}


function addWhitePadding(image, percent) {
  const p = Math.max(0, Number(percent) || 0) / 100;
  if (p <= 0) return image;

  const padX = Math.round(image.width * p);
  const padY = Math.round(image.height * p);
  const outW = image.width + padX * 2;
  const outH = image.height + padY * 2;
  const out = new Uint8Array(outW * outH * 4);

  for (let i = 0; i < out.length; i += 4) {
    out[i] = 246;
    out[i + 1] = 245;
    out[i + 2] = 239;
    out[i + 3] = 255;
  }

  const channels =
    image.data.length === image.width * image.height * 4 ? 4 :
    image.data.length === image.width * image.height * 3 ? 3 : 1;

  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const dst = ((y + padY) * outW + (x + padX)) * 4;

      if (channels === 4) {
        const src = (y * image.width + x) * 4;
        out[dst] = image.data[src];
        out[dst + 1] = image.data[src + 1];
        out[dst + 2] = image.data[src + 2];
        out[dst + 3] = image.data[src + 3];
      } else if (channels === 3) {
        const src = (y * image.width + x) * 3;
        out[dst] = image.data[src];
        out[dst + 1] = image.data[src + 1];
        out[dst + 2] = image.data[src + 2];
        out[dst + 3] = 255;
      } else {
        const v = image.data[y * image.width + x];
        out[dst] = v;
        out[dst + 1] = v;
        out[dst + 2] = v;
        out[dst + 3] = 255;
      }
    }
  }

  return { width: outW, height: outH, data: out };
}

function splitPages() {
  if (!state.bitmap) return;

  const w = state.bitmap.width;
  const h = state.bitmap.height;
  const splitPct = Number($("splitRange").value) / 100;
  const gutterPct = Number($("gutterRange").value) / 100;
  const splitX = w * splitPct;

  // Keep an overlapping band around the spine.
  // This avoids deleting characters near the binding before UVDoc runs.
  const halfOverlap = (w * gutterPct) / 2;
  const leftEnd = Math.min(w, Math.max(1, splitX + halfOverlap));
  const rightStart = Math.max(0, Math.min(w - 1, splitX - halfOverlap));

  const rc = document.createElement("canvas");
  const lc = document.createElement("canvas");

  drawBitmapCropToCanvas(state.bitmap, rightStart, 0, w - rightStart, h, rc);
  drawBitmapCropToCanvas(state.bitmap, 0, 0, leftEnd, h, lc);

  const pagePadPercent = Number($("pagePadRange").value);
  state.rightInput = addWhitePadding(canvasToPixels(rc), pagePadPercent);
  state.leftInput = addWhitePadding(canvasToPixels(lc), pagePadPercent);
  state.rightFlat = null;
  state.leftFlat = null;
  state.rightDetection = null;
  state.leftDetection = null;
  state.rightStrips = [];
  state.leftStrips = [];
  state.excludedColumns.right = new Set();
  state.excludedColumns.left = new Set();
  state.overlayHitboxes.right = [];
  state.overlayHitboxes.left = [];
  state.selectedColumn = null;
  state.columnOverrides.right = new Map();
  state.columnOverrides.left = new Map();
  state.overlayDrag = null;
  $("restoreColumnsBtn").disabled = true;
  closeColumnEditor();
  state.rightRecognition = [];
  state.leftRecognition = [];

  clearCanvas("rightFlat", "rightFlatEmpty", "等待 UVDoc");
  clearCanvas("leftFlat", "leftFlatEmpty", "等待 UVDoc");
  clearCanvas("rightOverlay", "rightOverlayEmpty", "等待 Detector");
  clearCanvas("leftOverlay", "leftOverlayEmpty", "等待 Detector");
  $("rightStats").textContent = "尚未偵測";
  $("leftStats").textContent = "尚未偵測";
  $("rightStripStats").textContent = "尚未抽欄";
  $("leftStripStats").textContent = "尚未抽欄";
  $("rightStrips").innerHTML = '<div class="empty">等待 V3 抽欄</div>';
  $("leftStrips").innerHTML = '<div class="empty">等待 V3 抽欄</div>';
  $("rightRecStats").textContent = "尚未辨識";
  $("leftRecStats").textContent = "尚未辨識";
  $("rightRecognition").innerHTML = '<div class="empty">等待 recognition</div>';
  $("leftRecognition").innerHTML = '<div class="empty">等待 recognition</div>';
  $("fullText").value = "";
  $("copyTextBtn").disabled = true;
  $("downloadTextBtn").disabled = true;

  setStatus("左右頁切割完成。", "中央書脊採重疊保留，不會先刪掉靠裝訂處的文字。", 0);
}

async function runUvDoc() {
  if (!state.rightInput || !state.leftInput) splitPages();

  const uv = await ensureUvDoc();

  setStatus("UVDoc：正在展平右頁…", "", 55);
  state.rightFlat = (await uv.run(state.rightInput)).doctrImage;
  pixelsToCanvas(state.rightFlat, $("rightFlat"));
  showCanvas("rightFlat", "rightFlatEmpty");

  setStatus("UVDoc：正在展平左頁…", "", 72);
  state.leftFlat = (await uv.run(state.leftInput)).doctrImage;
  pixelsToCanvas(state.leftFlat, $("leftFlat"));
  showCanvas("leftFlat", "leftFlatEmpty");

  setStatus("UVDoc 完成。", "現在可以跑 Detector + PCA 中心線。", 100);
}

function boxPoints(box) {
  if (Array.isArray(box.points) && box.points.length >= 4) {
    return box.points.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  if (Array.isArray(box.polygon) && box.polygon.length >= 4) {
    return box.polygon.map((p) => ({ x: Number(p.x), y: Number(p.y) }));
  }
  return [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x + box.width, y: box.y + box.height },
    { x: box.x, y: box.y + box.height },
  ];
}

function pcaGeometry(points) {
  const n = points.length;
  if (n < 2) return null;

  let cx = 0, cy = 0;
  for (const p of points) { cx += p.x; cy += p.y; }
  cx /= n; cy /= n;

  let a = 0, b = 0, d = 0;
  for (const p of points) {
    const x = p.x - cx;
    const y = p.y - cy;
    a += x * x;
    b += x * y;
    d += y * y;
  }
  a /= n;
  b /= n;
  d /= n;

  // Principal eigenvector for symmetric 2x2 covariance matrix.
  const trace = a + d;
  const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const lambda1 = (trace + disc) / 2;

  let vx, vy;
  if (Math.abs(b) > 1e-8) {
    vx = lambda1 - d;
    vy = b;
  } else if (a >= d) {
    vx = 1; vy = 0;
  } else {
    vx = 0; vy = 1;
  }

  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm; vy /= norm;
  if (vy < 0) { vx = -vx; vy = -vy; }

  const mx = -vy, my = vx;
  let majorMin = Infinity, majorMax = -Infinity;
  let minorMin = Infinity, minorMax = -Infinity;

  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const q1 = dx * vx + dy * vy;
    const q2 = dx * mx + dy * my;
    majorMin = Math.min(majorMin, q1);
    majorMax = Math.max(majorMax, q1);
    minorMin = Math.min(minorMin, q2);
    minorMax = Math.max(minorMax, q2);
  }

  const top = { x: cx + vx * majorMin, y: cy + vy * majorMin };
  const bottom = { x: cx + vx * majorMax, y: cy + vy * majorMax };
  const length = majorMax - majorMin;
  const width = minorMax - minorMin;
  const dy = bottom.y - top.y;
  if (Math.abs(dy) < 3) return null;

  const slope = (bottom.x - top.x) / dy;
  const intercept = top.x - slope * top.y;

  return {
    center: { x: cx, y: cy },
    top,
    bottom,
    length,
    width,
    slope,
    intercept,
    y0: Math.min(top.y, bottom.y),
    y1: Math.max(top.y, bottom.y),
  };
}

function median(values) {
  if (!values.length) return 0;
  const x = [...values].sort((a, b) => a - b);
  const m = Math.floor(x.length / 2);
  return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2;
}

function lineX(c, y) {
  return c.slope * y + c.intercept;
}

function makeColumns(boxes, imageHeight) {
  const cols = [];

  for (const box of boxes) {
    const points = boxPoints(box);
    const g = pcaGeometry(points);
    if (!g) continue;

    // Same intent as Colab: keep long vertical-ish text regions,
    // including shorter headings, but reject tiny/non-column shapes.
    if (g.length < g.width * 2) continue;
    if (g.length < imageHeight * 0.025) continue;

    cols.push({ ...g, points, rawBox: box });
  }

  if (!cols.length) return [];

  const stableSlopes = cols
    .map((c) => c.slope)
    .filter((s) => Math.abs(s) < 0.5);

  const pageMedianSlope = stableSlopes.length ? median(stableSlopes) : 0;

  for (const c of cols) {
    if (Math.abs(c.slope - pageMedianSlope) > 0.12) {
      c.slope = pageMedianSlope;
      c.intercept = c.center.x - pageMedianSlope * c.center.y;
    }
    c.xRef = lineX(c, imageHeight * 0.5);
  }

  // Traditional vertical reading order: right to left.
  cols.sort((a, b) => b.xRef - a.xRef);

  cols.forEach((c, index) => {
    c._columnId = `col-${index + 1}`;
  });

  return cols;
}


function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  };
}

function polygonCenter(poly) {
  return {
    x: poly.reduce((sum, p) => sum + p.x, 0) / poly.length,
    y: poly.reduce((sum, p) => sum + p.y, 0) / poly.length,
  };
}

function pointInPolygon(x, y, polygon) {
  let inside = false;

  for (
    let i = 0, j = polygon.length - 1;
    i < polygon.length;
    j = i++
  ) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersect =
      ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / Math.max(1e-9, yj - yi) + xi);

    if (intersect) inside = !inside;
  }

  return inside;
}

function canvasPointerPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  };
}

function circleContains(point, handle) {
  const dx = point.x - handle.x;
  const dy = point.y - handle.y;
  return dx * dx + dy * dy <= handle.hitRadius * handle.hitRadius;
}

function selectedHitbox(side) {
  if (!state.selectedColumn || state.selectedColumn.side !== side) {
    return null;
  }

  return state.overlayHitboxes[side].find(
    (h) => h.columnId === state.selectedColumn.columnId,
  ) ?? null;
}

function beginOverlayDrag(event, side, mode, columnId) {
  const canvas = event.currentTarget;
  const point = canvasPointerPoint(event, canvas);
  const col = getColumnById(side, columnId);
  const cols = activeColumns(side);
  const image = side === "right" ? state.rightFlat : state.leftFlat;

  if (!col || !image) return;

  const ov = getColumnOverride(side, columnId);
  const gap = typicalGap(cols, image.height);
  const charH = Math.max(8, col.width);

  state.overlayDrag = {
    pointerId: event.pointerId,
    side,
    columnId,
    mode,
    startX: point.x,
    startY: point.y,
    gap: Math.max(8, gap),
    charH,
    startOverride: {
      widthScale: ov.widthScale,
      shiftChars: ov.shiftChars,
      shiftYChars: ov.shiftYChars || 0,
      topChars: ov.topChars,
      bottomChars: ov.bottomChars,
    },
  };

  canvas.setPointerCapture?.(event.pointerId);
  canvas.classList.add("overlay-dragging");
  event.preventDefault();
}

function applyOverlayDrag(event) {
  const drag = state.overlayDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;

  const canvas = event.currentTarget;
  const point = canvasPointerPoint(event, canvas);
  const dx = point.x - drag.startX;
  const dy = point.y - drag.startY;
  const ov = getColumnOverride(drag.side, drag.columnId);
  const start = drag.startOverride;

  if (drag.mode === "move") {
    // True whole-box movement: moving no longer changes its height.
    ov.shiftChars = clampNumber(
      start.shiftChars + dx / drag.gap,
      -4,
      4,
    );
    ov.shiftYChars = clampNumber(
      start.shiftYChars + dy / drag.charH,
      -5,
      5,
    );
  } else if (drag.mode === "left") {
    ov.shiftChars = clampNumber(
      start.shiftChars + dx / (2 * drag.gap),
      -2.5,
      2.5,
    );
    ov.widthScale = clampNumber(
      start.widthScale - dx / drag.gap,
      0.35,
      2.2,
    );
  } else if (drag.mode === "right") {
    ov.shiftChars = clampNumber(
      start.shiftChars + dx / (2 * drag.gap),
      -2.5,
      2.5,
    );
    ov.widthScale = clampNumber(
      start.widthScale + dx / drag.gap,
      0.35,
      2.2,
    );
  } else if (drag.mode === "top") {
    ov.topChars = clampNumber(
      start.topChars - dy / drag.charH,
      -1.5,
      4,
    );
  } else if (drag.mode === "bottom") {
    ov.bottomChars = clampNumber(
      start.bottomChars + dy / drag.charH,
      -1.5,
      4,
    );
  }

  redrawOverlaySide(drag.side);
  updateSelectionToolbar("正在調整框線…放開後只更新裁切，不會自動 OCR。");
  event.preventDefault();
}

async function finishOverlayDrag(event) {
  const drag = state.overlayDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;

  const canvas = event.currentTarget;
  state.overlayDrag = null;

  try {
    canvas.releasePointerCapture?.(event.pointerId);
  } catch {}

  canvas.classList.remove("overlay-dragging");

  // IMPORTANT: do NOT run OCR here.
  // Only rebuild the Column preview so dragging stays responsive.
  commitSelectedColumnGeometry();
  redrawOverlaySide(drag.side);
  updateSelectionToolbar(
    "框線已更新；確認沒問題後再按「重新辨識選取欄」。",
  );

  setStatus(
    "欄位框線已調整。",
    "目前只更新裁切，不會自動跑 OCR。",
    0,
  );

  event.preventDefault();
}
function drawOverlay(image, boxes, cols, canvas, side = null, suggestions = []) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");

  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  ctx.strokeStyle = "rgba(220, 55, 65, .52)";
  ctx.lineWidth = Math.max(1.15, canvas.width / 780);

  for (const box of boxes) {
    const pts = boxPoints(box);
    if (!pts.length) continue;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();
  }

  const fontSize = Math.max(15, canvas.width / 42);
  const hitboxes = [];

  cols.forEach((c, i) => {
    const polygon = side
      ? columnCorridorPolygon(image, cols, i, side)
      : null;

    const isSelected =
      side &&
      state.selectedColumn?.side === side &&
      state.selectedColumn?.columnId === c._columnId;

    const topMid = polygon
      ? midpoint(polygon[0], polygon[1])
      : { x: lineX(c, c.y0), y: c.y0 };
    const bottomMid = polygon
      ? midpoint(polygon[2], polygon[3])
      : { x: lineX(c, c.y1), y: c.y1 };

    // Selected frame.
    let editHandles = null;
    if (isSelected && polygon) {
      ctx.beginPath();
      ctx.moveTo(polygon[0].x, polygon[0].y);
      for (let k = 1; k < polygon.length; k++) {
        ctx.lineTo(polygon[k].x, polygon[k].y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 196, 0, .20)";
      ctx.fill();
      ctx.strokeStyle = "rgba(210, 145, 0, .99)";
      ctx.lineWidth = Math.max(2.8, canvas.width / 360);
      ctx.stroke();

      const top = midpoint(polygon[0], polygon[1]);
      const right = midpoint(polygon[1], polygon[2]);
      const bottom = midpoint(polygon[2], polygon[3]);
      const left = midpoint(polygon[3], polygon[0]);
      const center = polygonCenter(polygon);

      const radius = Math.max(7, canvas.width / 105);
      const hitRadius = Math.max(14, radius * 2.05);

      editHandles = [
        { mode: "top", ...top, radius, hitRadius },
        { mode: "right", ...right, radius, hitRadius },
        { mode: "bottom", ...bottom, radius, hitRadius },
        { mode: "left", ...left, radius, hitRadius },
        {
          mode: "move",
          ...center,
          radius: radius * 1.15,
          hitRadius: hitRadius * 1.15,
        },
      ];

      for (const handle of editHandles) {
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, handle.radius, 0, Math.PI * 2);
        ctx.fillStyle =
          handle.mode === "move"
            ? "rgba(255, 196, 0, .99)"
            : "white";
        ctx.fill();
        ctx.strokeStyle = "rgba(194, 130, 0, 1)";
        ctx.lineWidth = Math.max(2, canvas.width / 650);
        ctx.stroke();

        if (handle.mode === "move") {
          ctx.fillStyle = "rgba(105, 70, 0, .98)";
          ctx.font = `bold ${Math.max(11, handle.radius * 1.25)}px system-ui`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("✥", handle.x, handle.y + 0.5);
          ctx.textAlign = "start";
          ctx.textBaseline = "alphabetic";
        }
      }
    }

    // Centerline follows the EDITED corridor, not the original PCA line.
    ctx.strokeStyle = isSelected
      ? "rgba(220, 145, 0, 1)"
      : c._rescued
        ? "rgba(24, 145, 105, .96)"
        : "rgba(25, 105, 215, .92)";
    ctx.lineWidth = isSelected
      ? Math.max(3, canvas.width / 350)
      : Math.max(2.2, canvas.width / 430);

    ctx.beginPath();
    ctx.moveTo(topMid.x, topMid.y);
    ctx.lineTo(bottomMid.x, bottomMid.y);
    ctx.stroke();

    const labelX = Math.max(
      4,
      Math.min(canvas.width - 70, topMid.x + 4),
    );
    const labelY = Math.max(22, topMid.y + 20);
    const badgeSize = Math.max(17, fontSize * 0.95);

    const bx = Math.max(2, labelX - badgeSize - 3);
    const by = Math.max(2, labelY - badgeSize + 2);

    ctx.fillStyle = "rgba(190, 42, 50, .92)";
    ctx.fillRect(bx, by, badgeSize, badgeSize);
    ctx.fillStyle = "white";
    ctx.font = `bold ${Math.max(13, badgeSize * 0.78)}px system-ui`;
    ctx.fillText("×", bx + badgeSize * 0.17, by + badgeSize * 0.82);

    ctx.fillStyle = isSelected
      ? "rgba(175, 110, 0, 1)"
      : c._rescued
        ? "rgba(20, 125, 92, .98)"
        : "rgba(25, 105, 215, .95)";
    ctx.font = `bold ${fontSize}px system-ui`;
    ctx.fillText(
      c._rescued
        ? `${i + 1}補`
        : String(i + 1),
      labelX,
      labelY,
    );

    // Every column gets a full corridor-shaped hit target.
    const xs = polygon ? polygon.map((p) => p.x) : [topMid.x, bottomMid.x];
    const ys = polygon ? polygon.map((p) => p.y) : [topMid.y, bottomMid.y];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const corridorWidth = Math.max(
      12,
      Math.abs(polygon?.[1]?.x - polygon?.[0]?.x || c.width),
    );

    hitboxes.push({
      x: minX - 8,
      y: minY - 8,
      width: Math.max(18, maxX - minX + 16),
      height: Math.max(18, maxY - minY + 16),
      columnId: c._columnId,
      polygon,
      editHandles,
      centerX: (minX + maxX) / 2,
      lineStart: topMid,
      lineEnd: bottomMid,
      selectionRadius: Math.max(13, Math.min(26, corridorWidth * 0.72)),
      deleteBox: {
        x: bx - 4,
        y: by - 4,
        width: badgeSize + 8,
        height: badgeSize + 8,
      },
    });
  });

  const suggestionHitboxes = [];

  if (side && Array.isArray(suggestions)) {
    ctx.save();
    ctx.setLineDash([
      Math.max(7, canvas.width / 115),
      Math.max(5, canvas.width / 150),
    ]);
    ctx.lineWidth = Math.max(2.2, canvas.width / 430);
    ctx.strokeStyle = "rgba(24, 145, 105, .92)";
    ctx.fillStyle = "rgba(20, 125, 92, .98)";
    ctx.font = `bold ${Math.max(13, fontSize * 0.82)}px system-ui`;

    suggestions.forEach((c, i) => {
      const y0 = Math.max(0, c.y0);
      const y1 = Math.min(canvas.height - 1, c.y1);
      const x0 = lineX(c, y0);
      const x1 = lineX(c, y1);

      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();

      const labelX = Math.max(
        4,
        Math.min(canvas.width - 70, x0 + 5),
      );
      const labelY = Math.max(20, y0 + 18);
      ctx.fillText(`補？${i + 1}`, labelX, labelY);

      suggestionHitboxes.push({
        suggestionId: c._suggestionId,
        lineStart: { x: x0, y: y0 },
        lineEnd: { x: x1, y: y1 },
        hitRadius: Math.max(11, c.width * 0.65),
      });
    });

    ctx.restore();
  }

  if (side) {
    state.overlayHitboxes[side] = hitboxes;
    state.rescueSuggestionHitboxes[side] = suggestionHitboxes;
    canvas.classList.add("clickable-overlay");
  }
}


function grayAt(image, x, y) {
  const rgba = rgbaAt(
    image,
    Math.round(x),
    Math.round(y),
  );

  return (
    rgba[0] * 0.299 +
    rgba[1] * 0.587 +
    rgba[2] * 0.114
  );
}

function pageMedianColumnSlope(cols) {
  const stable = cols
    .map((c) => Number(c.slope))
    .filter((s) => Number.isFinite(s) && Math.abs(s) < 0.5);

  return stable.length ? median(stable) : 0;
}

function medianColumnWidth(cols) {
  const widths = cols
    .map((c) => Number(c.width))
    .filter((w) => Number.isFinite(w) && w >= 4 && w <= 140);

  return widths.length ? median(widths) : 28;
}

function inkScoreAlongLine(
  image,
  xRef,
  slope,
  bandHalf,
  y0,
  y1,
) {
  let dark = 0;
  let samples = 0;

  const midY = image.height * 0.5;

  for (let y = y0; y <= y1; y += 3) {
    const centerX = xRef + slope * (y - midY);

    for (let dx = -bandHalf; dx <= bandHalf; dx += 2) {
      const x = centerX + dx;
      if (x < 1 || x >= image.width - 1) continue;

      const g = grayAt(image, x, y);

      // We count genuinely dark print, not mild page shadows / yellowing.
      if (g < 158) {
        dark += 1 + Math.max(0, 158 - g) / 110;
      }
      samples += 1;
    }
  }

  return samples > 0 ? dark / samples : 0;
}

function estimateInkSpan(
  image,
  xRef,
  slope,
  bandHalf,
) {
  const H = image.height;
  const midY = H * 0.5;
  const step = 3;
  const activity = [];

  for (let y = Math.round(H * 0.035); y <= H * 0.975; y += step) {
    const centerX = xRef + slope * (y - midY);
    let hits = 0;

    for (let dx = -bandHalf; dx <= bandHalf; dx += 2) {
      const x = centerX + dx;
      if (x < 1 || x >= image.width - 1) continue;
      if (grayAt(image, x, y) < 158) hits += 1;
    }

    activity.push({
      y,
      active: hits >= 1,
    });
  }

  // Smooth over ~27 px so blank spaces between characters do not split the line.
  const radius = 4;
  const smooth = activity.map((item, i) => {
    let count = 0;
    let total = 0;

    for (
      let j = Math.max(0, i - radius);
      j <= Math.min(activity.length - 1, i + radius);
      j++
    ) {
      total += 1;
      if (activity[j].active) count += 1;
    }

    return {
      y: item.y,
      active: total > 0 && count / total >= 0.22,
    };
  });

  const active = smooth.filter((x) => x.active);
  if (!active.length) return null;

  let y0 = active[0].y;
  let y1 = active[active.length - 1].y;

  const extra = Math.max(8, bandHalf * 2);
  y0 = Math.max(0, y0 - extra);
  y1 = Math.min(H - 1, y1 + extra);

  if (y1 - y0 < H * 0.24) return null;

  return { y0, y1 };
}

function makeInkRescuedColumn(
  image,
  xRef,
  slope,
  width,
  y0,
  y1,
) {
  const intercept = xRef - slope * image.height * 0.5;
  const topX = slope * y0 + intercept;
  const bottomX = slope * y1 + intercept;
  const half = Math.max(4, width * 0.5);

  const points = [
    { x: topX - half, y: y0 },
    { x: topX + half, y: y0 },
    { x: bottomX + half, y: y1 },
    { x: bottomX - half, y: y1 },
  ];

  return {
    center: {
      x: (topX + bottomX) / 2,
      y: (y0 + y1) / 2,
    },
    top: { x: topX, y: y0 },
    bottom: { x: bottomX, y: y1 },
    length: Math.hypot(bottomX - topX, y1 - y0),
    width,
    slope,
    intercept,
    y0,
    y1,
    xRef,
    points,
    rawBox: null,
    _rescued: true,
    _rescueSource: "ink-gap",
  };
}

function assignColumnIdsAndSort(cols, imageHeight) {
  for (const c of cols) {
    c.xRef = lineX(c, imageHeight * 0.5);
  }

  cols.sort((a, b) => b.xRef - a.xRef);

  cols.forEach((c, index) => {
    c._columnId = `col-${index + 1}`;
  });

  return cols;
}

function sortedColumnsNoIdMutation(cols, imageHeight) {
  return [...cols].sort(
    (a, b) =>
      lineX(b, imageHeight * 0.5) -
      lineX(a, imageHeight * 0.5),
  );
}


function rescueFromRejectedDetectorBoxes(
  boxes,
  cols,
  image,
) {
  const H = image.height;
  const gap = typicalGap(cols, H);
  const slope = pageMedianColumnSlope(cols);
  const added = [];

  for (const box of boxes) {
    const points = boxPoints(box);
    const g = pcaGeometry(points);
    if (!g) continue;

    // More permissive than makeColumns(), but still only column-like regions.
    if (g.length < H * 0.10) continue;
    if (g.length < g.width * 1.45) continue;
    if (g.width > gap * 1.7) continue;

    if (Math.abs(g.slope - slope) > 0.16) {
      g.slope = slope;
      g.intercept = g.center.x - slope * g.center.y;
    }

    g.xRef = lineX(g, H * 0.5);

    const represented = [...cols, ...added].some((c) => {
      const overlapY = Math.max(
        0,
        Math.min(c.y1, g.y1) - Math.max(c.y0, g.y0),
      );

      const overlapRatio =
        overlapY /
        Math.max(1, Math.min(c.y1 - c.y0, g.y1 - g.y0));

      if (overlapRatio < 0.20) return false;

      const y = Math.max(
        Math.max(c.y0, g.y0),
        Math.min(
          Math.min(c.y1, g.y1),
          (Math.max(c.y0, g.y0) + Math.min(c.y1, g.y1)) / 2,
        ),
      );

      const distance = Math.abs(
        lineX(c, y) - lineX(g, y),
      );

      return distance < Math.max(9, gap * 0.33);
    });

    if (represented) continue;

    added.push({
      ...g,
      points,
      rawBox: box,
      _rescued: true,
      _rescueSource: "detector-box",
    });
  }

  return added;
}

function rescueInkGapColumns(
  image,
  cols,
  maxAdds = 2,
) {
  if (cols.length < 3) return [];

  const H = image.height;
  const W = image.width;
  const slope = pageMedianColumnSlope(cols);
  const gap = typicalGap(cols, H);
  const width = medianColumnWidth(cols);
  const bandHalf = Math.max(
    3,
    Math.min(10, Math.round(width * 0.28)),
  );

  const y0 = Math.round(H * 0.07);
  const y1 = Math.round(H * 0.94);

  const longCols = cols.filter(
    (c) => c.y1 - c.y0 >= H * 0.25,
  );

  const referenceScores = longCols
    .slice(0, 20)
    .map((c) =>
      inkScoreAlongLine(
        image,
        lineX(c, H * 0.5),
        slope,
        bandHalf,
        y0,
        y1,
      ),
    )
    .filter((v) => Number.isFinite(v) && v > 0);

  if (!referenceScores.length) return [];

  const baseline = median(referenceScores);
  const threshold = Math.max(
    0.008,
    baseline * 0.30,
  );

  const rescued = [];

  function currentColumns() {
    return sortedColumnsNoIdMutation(
      [...cols, ...rescued],
      H,
    );
  }

  for (let iteration = 0; iteration < maxAdds; iteration++) {
    const active = currentColumns();
    const xs = active.map((c) => lineX(c, H * 0.5));
    const intervals = [];

    // Large internal gaps are the safest place to search.
    for (let i = 0; i < xs.length - 1; i++) {
      const rightX = xs[i];
      const leftX = xs[i + 1];
      const distance = rightX - leftX;

      if (distance > gap * 1.52) {
        intervals.push({
          minX: leftX + gap * 0.30,
          maxX: rightX - gap * 0.30,
          kind: "internal",
          gapSize: distance,
        });
      }
    }

    // Also inspect one expected-column band just outside the detected text block.
    const maxX = Math.max(...xs);
    const minX = Math.min(...xs);

    intervals.push({
      minX: maxX + gap * 0.32,
      maxX: maxX + gap * 1.32,
      kind: "outer-right",
      gapSize: gap,
    });
    intervals.push({
      minX: minX - gap * 1.32,
      maxX: minX - gap * 0.32,
      kind: "outer-left",
      gapSize: gap,
    });

    let best = null;

    for (const interval of intervals) {
      const lo = Math.max(
        W * 0.018,
        Math.min(interval.minX, interval.maxX),
      );
      const hi = Math.min(
        W * 0.982,
        Math.max(interval.minX, interval.maxX),
      );

      if (hi - lo < gap * 0.22) continue;

      for (let x = lo; x <= hi; x += 2) {
        const nearest = active.reduce(
          (bestDist, c) =>
            Math.min(
              bestDist,
              Math.abs(lineX(c, H * 0.5) - x),
            ),
          Infinity,
        );

        if (nearest < gap * 0.31) continue;

        const score = inkScoreAlongLine(
          image,
          x,
          slope,
          bandHalf,
          y0,
          y1,
        );

        if (
          score >= threshold &&
          (!best || score > best.score)
        ) {
          best = {
            x,
            score,
            interval,
          };
        }
      }
    }

    if (!best) break;

    const span = estimateInkSpan(
      image,
      best.x,
      slope,
      Math.max(bandHalf, Math.round(width * 0.36)),
    );

    if (!span) break;

    // A rescue line should still carry a meaningful amount of real print.
    const refinedScore = inkScoreAlongLine(
      image,
      best.x,
      slope,
      Math.max(bandHalf, Math.round(width * 0.32)),
      span.y0,
      span.y1,
    );

    if (refinedScore < threshold * 0.90) break;

    rescued.push(
      makeInkRescuedColumn(
        image,
        best.x,
        slope,
        width,
        span.y0,
        span.y1,
      ),
    );
  }

  return rescued;
}

function rescueMissingColumns(
  image,
  boxes,
  baseCols,
  side,
) {
  // Critical v6.5 rule:
  // rescue analysis may SUGGEST columns, but must never modify baseCols.
  // This guarantees an OCR result that was correct before cannot become
  // worse simply because rescue analysis found a false positive.
  const cols = [...baseCols];

  if (!$("missingColumnRescue")?.checked) {
    return {
      cols,
      suggestions: [],
      detectorBoxSuggestions: 0,
      inkSuggestions: 0,
    };
  }

  const detectorSuggestions = rescueFromRejectedDetectorBoxes(
    boxes,
    cols,
    image,
  );

  // Do not feed detector suggestions into ink-gap geometry.
  // Candidate algorithms are independent and cannot change the real column list.
  const inkSuggestions = rescueInkGapColumns(
    image,
    cols,
    2,
  );

  const rawSuggestions = [
    ...detectorSuggestions,
    ...inkSuggestions,
  ];

  // De-duplicate candidate lines by x at page midpoint.
  const H = image.height;
  const gap = typicalGap(cols, H);
  const suggestions = [];

  for (const candidate of rawSuggestions) {
    const x = lineX(candidate, H * 0.5);

    const tooCloseToReal = cols.some(
      (c) =>
        Math.abs(lineX(c, H * 0.5) - x) <
        Math.max(10, gap * 0.36),
    );

    if (tooCloseToReal) continue;

    const duplicate = suggestions.some(
      (c) =>
        Math.abs(lineX(c, H * 0.5) - x) <
        Math.max(10, gap * 0.42),
    );

    if (duplicate) continue;

    suggestions.push({
      ...candidate,
      _rescued: false,
      _rescueSuggestion: true,
      _suggestionId:
        `suggest-${side}-${suggestions.length + 1}`,
    });

    if (suggestions.length >= 3) break;
  }

  return {
    cols,
    suggestions,
    detectorBoxSuggestions: detectorSuggestions.length,
    inkSuggestions: inkSuggestions.length,
  };
}


function addDetectorEdgeGuard(image, percent = DETECTOR_EDGE_GUARD_PERCENT) {
  const p = Math.max(0, Number(percent) || 0) / 100;
  if (p <= 0) {
    return {
      image,
      padX: 0,
      padY: 0,
    };
  }

  const padX = Math.max(8, Math.round(image.width * p));
  const padY = Math.max(8, Math.round(image.height * p));
  const outW = image.width + padX * 2;
  const outH = image.height + padY * 2;
  const out = new Uint8Array(outW * outH * 4);

  // Warm paper-like white, close to existing page padding.
  for (let i = 0; i < out.length; i += 4) {
    out[i] = 246;
    out[i + 1] = 245;
    out[i + 2] = 239;
    out[i + 3] = 255;
  }

  const channels =
    image.data.length === image.width * image.height * 4 ? 4 :
    image.data.length === image.width * image.height * 3 ? 3 : 1;

  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const dst = ((y + padY) * outW + (x + padX)) * 4;

      if (channels === 4) {
        const src = (y * image.width + x) * 4;
        out[dst] = image.data[src];
        out[dst + 1] = image.data[src + 1];
        out[dst + 2] = image.data[src + 2];
        out[dst + 3] = image.data[src + 3];
      } else if (channels === 3) {
        const src = (y * image.width + x) * 3;
        out[dst] = image.data[src];
        out[dst + 1] = image.data[src + 1];
        out[dst + 2] = image.data[src + 2];
        out[dst + 3] = 255;
      } else {
        const v = image.data[y * image.width + x];
        out[dst] = v;
        out[dst + 1] = v;
        out[dst + 2] = v;
        out[dst + 3] = 255;
      }
    }
  }

  return {
    image: {
      width: outW,
      height: outH,
      data: out,
    },
    padX,
    padY,
  };
}

function shiftDetectorBoxToOriginal(box, padX, padY, width, height) {
  const clampX = (x) => Math.max(0, Math.min(width - 1, Number(x) - padX));
  const clampY = (y) => Math.max(0, Math.min(height - 1, Number(y) - padY));

  const shifted = { ...box };

  if (Array.isArray(box.points)) {
    shifted.points = box.points.map((p) => ({
      ...p,
      x: clampX(p.x),
      y: clampY(p.y),
    }));
  }

  if (Array.isArray(box.polygon)) {
    shifted.polygon = box.polygon.map((p) => ({
      ...p,
      x: clampX(p.x),
      y: clampY(p.y),
    }));
  }

  // Keep rectangle-style fallbacks correct too.
  if (Number.isFinite(Number(box.x))) {
    shifted.x = clampX(box.x);
  }
  if (Number.isFinite(Number(box.y))) {
    shifted.y = clampY(box.y);
  }

  if (
    Number.isFinite(Number(box.width)) &&
    Number.isFinite(Number(box.x))
  ) {
    const right = clampX(Number(box.x) + Number(box.width));
    shifted.width = Math.max(1, right - shifted.x);
  }

  if (
    Number.isFinite(Number(box.height)) &&
    Number.isFinite(Number(box.y))
  ) {
    const bottom = clampY(Number(box.y) + Number(box.height));
    shifted.height = Math.max(1, bottom - shifted.y);
  }

  return shifted;
}

async function detectPage(image, side) {
  const detector = await ensureDetector();
  const options = detectorRuntimeOptions();

  // v6.5 restores the v6.2 detector geometry that successfully produced
  // the "臺灣史" crop. No hidden padding is inserted here.
  const detectorImage = normalizeInputToRgb(image);

  const boxes = await detector.run(detectorImage, {
    ...options,
    onProgress(event) {
      const stageName = {
        preprocess: "前處理",
        infer: "模型推理",
        postprocess: "DB 後處理",
      }[event.stage] || event.stage;

      setStatus(
        `Detector：${side === "right" ? "右頁" : "左頁"} ${stageName}…`,
        event.detectedCount != null
          ? `偵測到 ${event.detectedCount} 個區域`
          : "",
        side === "right" ? 45 : 78,
      );
    },
  });

  const baseCols = makeColumns(boxes, image.height);
  const rescue = rescueMissingColumns(
    image,
    boxes,
    baseCols,
    side,
  );
  const cols = rescue.cols;

  const canvas = $(side === "right" ? "rightOverlay" : "leftOverlay");
  drawOverlay(
    image,
    boxes,
    cols,
    canvas,
    side,
    rescue.suggestions,
  );

  showCanvas(
    side === "right" ? "rightOverlay" : "leftOverlay",
    side === "right" ? "rightOverlayEmpty" : "leftOverlayEmpty",
  );

  $(side === "right" ? "rightStats" : "leftStats").textContent =
    rescue.suggestions.length > 0
      ? `紅框 ${boxes.length} · 中心線 ${cols.length} · 缺欄候選 ${rescue.suggestions.length}`
      : `紅框 ${boxes.length} · 中心線 ${cols.length}`;

  return {
    boxes,
    cols,
    rescue,
  };
}

async function runDetector() {
  if (!state.rightFlat || !state.leftFlat) {
    await runUvDoc();
  }

  setStatus("Detector：右頁…", "使用 PP-OCRv5_mobile_det", 35);
  state.rightDetection = await detectPage(state.rightFlat, "right");

  setStatus("Detector：左頁…", "使用 PP-OCRv5_mobile_det", 68);
  state.leftDetection = await detectPage(state.leftFlat, "left");

  const candidates =
    (state.rightDetection?.rescue?.suggestions?.length || 0) +
    (state.leftDetection?.rescue?.suggestions?.length || 0);

  setStatus(
    "中心線完成。",
    candidates > 0
      ? `找到 ${candidates} 個綠色虛線缺欄候選。它們目前不影響 V3 / OCR；真的缺欄時直接點綠線採用。`
      : "沒有發現高可信度缺欄候選；既有中心線維持 Detector/PCA 原結果。",
    100,
  );
}


function typicalGap(cols, imageHeight) {
  if (!cols || cols.length < 2) return 45;

  const yRef = imageHeight * 0.5;
  const xs = cols.map((c) => lineX(c, yRef));
  let gaps = [];

  for (let i = 0; i < xs.length - 1; i++) {
    const g = Math.abs(xs[i] - xs[i + 1]);
    if (g > 8) gaps.push(g);
  }

  if (!gaps.length) return 45;

  const med = median(gaps);
  const good = gaps.filter((g) => g > med * 0.45 && g < med * 1.8);
  return good.length ? median(good) : med;
}

function rgbaAt(image, x, y) {
  const { width, height, data } = image;
  x = Math.max(0, Math.min(width - 1, x));
  y = Math.max(0, Math.min(height - 1, y));

  const channels =
    data.length === width * height * 4 ? 4 :
    data.length === width * height * 3 ? 3 : 1;

  const i = (y * width + x) * channels;

  if (channels === 4) {
    return [data[i], data[i + 1], data[i + 2], data[i + 3]];
  }
  if (channels === 3) {
    return [data[i], data[i + 1], data[i + 2], 255];
  }

  const v = data[y * width + x];
  return [v, v, v, 255];
}

function bilinearSample(image, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const p00 = rgbaAt(image, x0, y0);
  const p10 = rgbaAt(image, x1, y0);
  const p01 = rgbaAt(image, x0, y1);
  const p11 = rgbaAt(image, x1, y1);

  const out = [0, 0, 0, 255];
  for (let c = 0; c < 4; c++) {
    const a = p00[c] * (1 - tx) + p10[c] * tx;
    const b = p01[c] * (1 - tx) + p11[c] * tx;
    out[c] = Math.max(0, Math.min(255, Math.round(a * (1 - ty) + b * ty)));
  }
  return out;
}



function defaultColumnOverride() {
  return {
    widthScale: 1,
    shiftChars: 0,
    shiftYChars: 0,
    topChars: 0,
    bottomChars: 0,
  };
}

function getColumnOverride(side, columnId) {
  if (!state.columnOverrides[side].has(columnId)) {
    state.columnOverrides[side].set(columnId, defaultColumnOverride());
  }
  return state.columnOverrides[side].get(columnId);
}

function getColumnById(side, columnId) {
  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;
  return detection?.cols?.find((c) => c._columnId === columnId) ?? null;
}

function sideLabel(side) {
  return side === "right" ? "右頁" : "左頁";
}

function updateSelectedCardStyles() {
  document.querySelectorAll(".strip-card").forEach((card) => {
    const selected =
      state.selectedColumn &&
      card.dataset.side === state.selectedColumn.side &&
      card.dataset.columnId === state.selectedColumn.columnId;
    card.classList.toggle("selected-strip-card", !!selected);
  });
}

function closeColumnEditor() {
  state.selectedColumn = null;
  state.overlayDrag = null;

  if ($("overlayEditBar")) $("overlayEditBar").classList.add("hidden");
  if ($("rerunSelectedBtn")) $("rerunSelectedBtn").disabled = true;
  if ($("resetSelectedBtn")) $("resetSelectedBtn").disabled = true;
  if ($("deleteSelectedBtn")) $("deleteSelectedBtn").disabled = true;
  if ($("clearSelectedBtn")) $("clearSelectedBtn").disabled = true;

  updateSelectedCardStyles();
}

function updateSelectionToolbar(message = null) {
  if (!state.selectedColumn) {
    closeColumnEditor();
    return;
  }

  const { side, columnId } = state.selectedColumn;
  const active = activeColumns(side);
  const position = active.findIndex((c) => c._columnId === columnId);

  if (position < 0) {
    closeColumnEditor();
    return;
  }

  $("overlayEditBar").classList.remove("hidden");
  $("overlayEditTitle").textContent =
    `${sideLabel(side)} · Column ${String(position + 1).padStart(2, "0")}`;
  $("overlayEditStatus").textContent =
    message || "黃色框內任意處都能拖；也可直接抓其他藍色欄位。調好後再按「重新辨識選取欄」。";

  $("rerunSelectedBtn").disabled = false;
  $("resetSelectedBtn").disabled = false;
  $("deleteSelectedBtn").disabled = false;
  $("clearSelectedBtn").disabled = false;

  updateSelectedCardStyles();
}

function scrollToSelectedOverlay(side) {
  const canvas = $(side === "right" ? "rightOverlay" : "leftOverlay");
  if (!canvas) return;

  requestAnimationFrame(() => {
    canvas.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
  });
}

function selectColumn(side, columnId, options = {}) {
  if (state.excludedColumns[side].has(columnId)) return;

  state.selectedColumn = { side, columnId };
  updateSelectionToolbar();
  redrawOverlaySide(side);

  if (options.scrollToOverlay) {
    scrollToSelectedOverlay(side);
  }
}

function resetSelectedColumnOverride() {
  if (!state.selectedColumn) return;
  const { side, columnId } = state.selectedColumn;

  state.columnOverrides[side].set(columnId, defaultColumnOverride());
  redrawOverlaySide(side);

  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;
  const image = side === "right"
    ? state.rightFlat
    : state.leftFlat;

  if (detection && image) {
    const strips = extractPageStrips(image, detection, side);
    if (side === "right") state.rightStrips = strips;
    else state.leftStrips = strips;
  }

  updateSelectionToolbar("已還原此欄框線；尚未重新辨識。");
}

function selectedColumnContext() {
  if (!state.selectedColumn) return null;

  const { side, columnId } = state.selectedColumn;
  const cols = activeColumns(side);
  const index = cols.findIndex((c) => c._columnId === columnId);
  if (index < 0) return null;

  const image = side === "right" ? state.rightFlat : state.leftFlat;
  if (!image) return null;

  return { side, columnId, cols, index, image };
}

function commitSelectedColumnGeometry() {
  const context = selectedColumnContext();
  if (!context) return null;

  const { side, columnId, image } = context;
  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;

  const strips = extractPageStrips(image, detection, side);

  if (side === "right") state.rightStrips = strips;
  else state.leftStrips = strips;

  const strip = strips.find((item) => item.columnId === columnId) ?? null;
  updateSelectedCardStyles();
  return strip;
}

async function rerunSelectedColumnOcr() {
  const context = selectedColumnContext();
  if (!context) return;

  if (!state.recPerf) {
    state.recPerf = freshRecPerf();
  }

  const { side, columnId } = context;
  const button = $("rerunSelectedBtn");
  button.disabled = true;

  try {
    updateSelectionToolbar("正在重新辨識這一欄…");
    await nextPaint();

    const strip = commitSelectedColumnGeometry();
    if (!strip) {
      throw new Error("找不到選取欄的最新裁切結果。");
    }

    let recognizer = state.recognizer;
    if (!recognizer) {
      recognizer = await ensureRecognizer(false);
    }

    let result;
    try {
      result = await recognizeStripWithRescue(strip, recognizer);
    } catch (error) {
      const currentBackend = state.recSpec
        ? state.backends[state.recSpec.label]
        : undefined;

      if (currentBackend !== "webgpu") throw error;

      console.warn(
        "Selected-column WebGPU recognition failed; retrying WASM.",
        error,
      );

      state.recognizer = null;
      state.recSession = null;
      state.recPreset = null;
      recognizer = await ensureRecognizer(true);
      result = await recognizeStripWithRescue(strip, recognizer);
    }

    result.columnId = columnId;

    const target = side === "right"
      ? state.rightRecognition
      : state.leftRecognition;
    const idx = target.findIndex((x) => x.columnId === columnId);

    if (idx >= 0) target[idx] = result;
    else target.push(result);

    const order = new Map(
      activeColumns(side).map((c, orderIndex) => [c._columnId, orderIndex]),
    );
    target.sort(
      (a, b) =>
        (order.get(a.columnId) ?? 9999) -
        (order.get(b.columnId) ?? 9999),
    );

    renderRecognition(target, side);

    const statsId = side === "right" ? "rightRecStats" : "leftRecStats";
    $(statsId).textContent =
      `${target.filter((x) => x.text.trim()).length}/${target.length} 欄有文字`;

    assembleFullText();

    updateSelectionToolbar(
      `已更新：${result.text || "（空白）"} · ${(result.confidence * 100).toFixed(1)}%`,
    );
    setStatus(
      "選取欄已重新辨識。",
      "只跑這一欄；全文已同步更新。",
      100,
    );
  } catch (error) {
    console.error("Selected column recognition failed:", error);
    updateSelectionToolbar(
      `重新辨識失敗：${error instanceof Error ? error.message : String(error)}`,
    );
  } finally {
    button.disabled = false;
  }
}


function columnCorridorPolygon(image, cols, index, side) {
  const c = cols[index];
  const H = image.height;
  const gap = typicalGap(cols, H);
  const ov = getColumnOverride(side, c._columnId);

  const baseExtend = Number($("columnExtendRange").value);
  const charH = Math.max(8, c.width);

  let y0 = Math.max(
    0,
    Math.round(c.y0 - charH * Math.max(0, baseExtend + ov.topChars)),
  );
  let y1 = Math.min(
    H - 1,
    Math.round(c.y1 + charH * Math.max(0, baseExtend + ov.bottomChars)),
  );

  if (ov.topChars < -baseExtend) {
    y0 = Math.min(
      y1 - 2,
      Math.round(c.y0 + charH * Math.abs(baseExtend + ov.topChars)),
    );
  }
  if (ov.bottomChars < -baseExtend) {
    y1 = Math.max(
      y0 + 2,
      Math.round(c.y1 - charH * Math.abs(baseExtend + ov.bottomChars)),
    );
  }

  const shiftY = Math.round((ov.shiftYChars || 0) * charH);
  const height = Math.max(3, y1 - y0);
  y0 += shiftY;
  y1 += shiftY;

  if (y0 < 0) {
    y1 -= y0;
    y0 = 0;
  }
  if (y1 > H - 1) {
    const overflow = y1 - (H - 1);
    y0 -= overflow;
    y1 = H - 1;
  }

  y0 = Math.max(0, y0);
  y1 = Math.min(H - 1, Math.max(y0 + Math.min(3, height), y1));

  function boundariesAt(y) {
    const shiftPx = ov.shiftChars * gap;
    const xc = lineX(c, y) + shiftPx;

    let rightBoundary;
    if (index > 0) {
      rightBoundary = (lineX(c, y) + lineX(cols[index - 1], y)) / 2 + shiftPx;
    } else {
      rightBoundary = lineX(c, y) + gap * 0.48 + shiftPx;
    }

    let leftBoundary;
    if (index < cols.length - 1) {
      leftBoundary = (lineX(c, y) + lineX(cols[index + 1], y)) / 2 + shiftPx;
    } else {
      leftBoundary = lineX(c, y) - gap * 0.48 + shiftPx;
    }

    let xl = Math.min(leftBoundary, rightBoundary);
    let xr = Math.max(leftBoundary, rightBoundary);

    const center = (xl + xr) / 2;
    const half = ((xr - xl) / 2) * ov.widthScale;
    xl = center - half;
    xr = center + half;

    return { xl, xr, xc };
  }

  const top = boundariesAt(y0);
  const bottom = boundariesAt(y1);

  return [
    { x: top.xl, y: y0 },
    { x: top.xr, y: y0 },
    { x: bottom.xr, y: y1 },
    { x: bottom.xl, y: y1 },
  ];
}

function activeColumns(side) {
  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;

  if (!detection?.cols) return [];

  const excluded = state.excludedColumns[side];
  return detection.cols.filter((c) => !excluded.has(c._columnId));
}

function refreshRestoreButton() {
  $("restoreColumnsBtn").disabled =
    state.excludedColumns.right.size === 0 &&
    state.excludedColumns.left.size === 0;
}

function redrawOverlaySide(side) {
  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;
  const image = side === "right"
    ? state.rightFlat
    : state.leftFlat;

  if (!detection || !image) return;

  const canvas = $(side === "right" ? "rightOverlay" : "leftOverlay");
  const cols = activeColumns(side);
  const suggestions = detection.rescue?.suggestions || [];

  drawOverlay(
    image,
    detection.boxes,
    cols,
    canvas,
    side,
    suggestions,
  );

  $(side === "right" ? "rightStats" : "leftStats").textContent =
    suggestions.length > 0
      ? `紅框 ${detection.boxes.length} · 使用 ${cols.length}/${detection.cols.length} 欄 · 候選 ${suggestions.length}`
      : `紅框 ${detection.boxes.length} · 使用 ${cols.length}/${detection.cols.length} 欄`;
}

function clearRecognitionForSide(side) {
  if (side === "right") {
    state.rightRecognition = state.rightRecognition.filter(
      (item) => !state.excludedColumns.right.has(item.columnId),
    );
    renderRecognition(state.rightRecognition, "right");
    $("rightRecStats").textContent =
      `${state.rightRecognition.filter((x) => x.text.trim()).length}/${state.rightRecognition.length} 欄有文字`;
  } else {
    state.leftRecognition = state.leftRecognition.filter(
      (item) => !state.excludedColumns.left.has(item.columnId),
    );
    renderRecognition(state.leftRecognition, "left");
    $("leftRecStats").textContent =
      `${state.leftRecognition.filter((x) => x.text.trim()).length}/${state.leftRecognition.length} 欄有文字`;
  }

  assembleFullText();
}

function removeColumn(side, columnId) {
  if (!columnId) return;

  state.excludedColumns[side].add(columnId);

  if (
    state.selectedColumn?.side === side &&
    state.selectedColumn?.columnId === columnId
  ) {
    state.selectedColumn = null;
    closeColumnEditor();
  }
  refreshRestoreButton();
  redrawOverlaySide(side);

  const detection = side === "right"
    ? state.rightDetection
    : state.leftDetection;
  const image = side === "right"
    ? state.rightFlat
    : state.leftFlat;

  if (detection && image) {
    const strips = extractPageStrips(image, detection, side);
    if (side === "right") state.rightStrips = strips;
    else state.leftStrips = strips;
  }

  clearRecognitionForSide(side);

  setStatus(
    "已刪除誤抓欄位。",
    "圖上、Column 列表與全文已同步排除；可按「恢復全部刪除」重來。",
    100,
  );
}

function restoreAllColumns() {
  state.excludedColumns.right = new Set();
  state.excludedColumns.left = new Set();
  state.selectedColumn = null;
  closeColumnEditor();
  refreshRestoreButton();

  redrawOverlaySide("right");
  redrawOverlaySide("left");

  if (state.rightDetection && state.rightFlat) {
    state.rightStrips = extractPageStrips(
      state.rightFlat,
      state.rightDetection,
      "right",
    );
  }

  if (state.leftDetection && state.leftFlat) {
    state.leftStrips = extractPageStrips(
      state.leftFlat,
      state.leftDetection,
      "left",
    );
  }

  // Previously deleted recognition results are not cached separately.
  // Keep existing results and ask the user to re-run recognition for restored columns.
  setStatus(
    "已恢復全部欄位。",
    "藍線與 Column 已恢復；若要補回文字，重新按「5. PP-OCRv5 辨識」。",
    0,
  );
}


function pointToSegmentDistance(point, a, b) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = point.x - a.x;
  const wy = point.y - a.y;

  const vv = vx * vx + vy * vy;
  if (vv <= 1e-9) {
    return Math.hypot(point.x - a.x, point.y - a.y);
  }

  const t = clampNumber(
    (wx * vx + wy * vy) / vv,
    0,
    1,
  );

  const px = a.x + t * vx;
  const py = a.y + t * vy;
  return Math.hypot(point.x - px, point.y - py);
}

function columnAtPoint(side, point) {
  const candidates = state.overlayHitboxes[side]
    .map((h) => {
      const insidePolygon =
        h.polygon &&
        pointInPolygon(point.x, point.y, h.polygon);

      const insideBox =
        point.x >= h.x &&
        point.x <= h.x + h.width &&
        point.y >= h.y &&
        point.y <= h.y + h.height;

      const lineDistance =
        h.lineStart && h.lineEnd
          ? pointToSegmentDistance(point, h.lineStart, h.lineEnd)
          : Math.abs(point.x - h.centerX);

      const closeToLine =
        lineDistance <= (h.selectionRadius || 16);

      if (!insidePolygon && !insideBox && !closeToLine) {
        return null;
      }

      return {
        ...h,
        lineDistance,
        insidePolygon,
      };
    })
    .filter(Boolean);

  if (!candidates.length) return null;

  // Nearest actual edited centerline wins.
  // This is important when the yellow corridor overlaps a neighboring column.
  candidates.sort((a, b) => {
    if (Math.abs(a.lineDistance - b.lineDistance) > 0.5) {
      return a.lineDistance - b.lineDistance;
    }

    // If distances are nearly equal, prefer the smaller / more specific corridor.
    return a.width - b.width;
  });

  return candidates[0];
}

function updateOverlayHoverCursor(event, side) {
  const canvas = event.currentTarget;
  if (state.overlayDrag) return;

  canvas.classList.remove("overlay-ew-resize", "overlay-ns-resize");
  canvas.style.cursor = "default";

  const point = canvasPointerPoint(event, canvas);

  if (rescueSuggestionAtPoint(side, point)) {
    canvas.style.cursor = "pointer";
    return;
  }

  const selected = selectedHitbox(side);

  if (selected?.editHandles) {
    const handle = selected.editHandles.find(
      (candidate) => circleContains(point, candidate),
    );

    if (handle) {
      if (handle.mode === "left" || handle.mode === "right") {
        canvas.classList.add("overlay-ew-resize");
      } else if (handle.mode === "top" || handle.mode === "bottom") {
        canvas.classList.add("overlay-ns-resize");
      } else {
        canvas.style.cursor = "grab";
      }
      return;
    }
  }

  const nearest = columnAtPoint(side, point);

  if (nearest) {
    canvas.style.cursor = "grab";
    return;
  }

  if (
    selected?.polygon &&
    pointInPolygon(point.x, point.y, selected.polygon)
  ) {
    canvas.style.cursor = "grab";
  }
}


function rescueSuggestionAtPoint(side, point) {
  const candidates = state.rescueSuggestionHitboxes[side] || [];

  let best = null;

  for (const candidate of candidates) {
    const distance = pointToSegmentDistance(
      point,
      candidate.lineStart,
      candidate.lineEnd,
    );

    if (distance > candidate.hitRadius) continue;

    if (!best || distance < best.distance) {
      best = {
        ...candidate,
        distance,
      };
    }
  }

  return best;
}

function acceptRescueSuggestion(side, suggestionId) {
  const detection =
    side === "right"
      ? state.rightDetection
      : state.leftDetection;

  if (!detection?.rescue?.suggestions?.length) return;

  const index = detection.rescue.suggestions.findIndex(
    (s) => s._suggestionId === suggestionId,
  );

  if (index < 0) return;

  const [candidate] = detection.rescue.suggestions.splice(index, 1);

  // Preserve every existing column ID. The accepted column gets its own
  // unique stable ID; only array reading order is re-sorted.
  const accepted = {
    ...candidate,
    _rescued: true,
    _rescueSuggestion: false,
    _columnId:
      `accepted-${side}-${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 7)}`,
  };

  detection.cols.push(accepted);
  detection.cols.sort(
    (a, b) =>
      lineX(b, (side === "right" ? state.rightFlat : state.leftFlat).height * 0.5) -
      lineX(a, (side === "right" ? state.rightFlat : state.leftFlat).height * 0.5),
  );

  redrawOverlaySide(side);

  const image =
    side === "right"
      ? state.rightFlat
      : state.leftFlat;

  if (image) {
    const strips = extractPageStrips(
      image,
      detection,
      side,
    );

    if (side === "right") {
      state.rightStrips = strips;
      state.rightRecognition = [];
      renderRecognition([], "right");
      $("rightRecStats").textContent = "0 欄有文字";
    } else {
      state.leftStrips = strips;
      state.leftRecognition = [];
      renderRecognition([], "left");
      $("leftRecStats").textContent = "0 欄有文字";
    }

    assembleFullText();
  }

  setStatus(
    "已採用缺欄候選。",
    "這條綠色虛線現在才正式加入 V3。因鄰欄 corridor 可能改變，請重新跑 Recognition。",
    100,
  );
}

function handleOverlayPointerDown(event, side) {
  const canvas = event.currentTarget;
  const point = canvasPointerPoint(event, canvas);


  // Safe rescue proposal: clicking a green dashed line is explicit consent
  // to add it. Until this moment it has not affected any OCR geometry.
  const rescueSuggestion = rescueSuggestionAtPoint(side, point);
  if (rescueSuggestion) {
    acceptRescueSuggestion(
      side,
      rescueSuggestion.suggestionId,
    );
    event.preventDefault();
    return;
  }

  // 1) Delete X always wins.
  const deleteHit = state.overlayHitboxes[side].find((h) => {
    const d = h.deleteBox;
    return (
      point.x >= d.x &&
      point.x <= d.x + d.width &&
      point.y >= d.y &&
      point.y <= d.y + d.height
    );
  });

  if (deleteHit) {
    removeColumn(side, deleteHit.columnId);
    event.preventDefault();
    return;
  }

  const selected = selectedHitbox(side);

  // 2) Resize handles of the currently selected box have priority.
  if (selected?.editHandles) {
    const handle = selected.editHandles.find(
      (candidate) => circleContains(point, candidate),
    );

    if (handle) {
      beginOverlayDrag(
        event,
        side,
        handle.mode,
        selected.columnId,
      );
      return;
    }
  }

  // 3) Find the nearest ACTUAL column centerline before allowing the large
  // selected yellow corridor to consume the click.
  const nearest = columnAtPoint(side, point);

  if (
    nearest &&
    (!selected || nearest.columnId !== selected.columnId)
  ) {
    // Switch immediately on pointer-down. One click is enough.
    selectColumn(side, nearest.columnId);
    redrawOverlaySide(side);

    // And keep the free-direct behavior: the same gesture can start moving it.
    beginOverlayDrag(
      event,
      side,
      "move",
      nearest.columnId,
    );
    return;
  }

  // 4) If the nearest column is the selected one, or there is no other
  // specific column under the pointer, anywhere inside the yellow polygon moves it.
  if (
    selected?.polygon &&
    pointInPolygon(point.x, point.y, selected.polygon)
  ) {
    beginOverlayDrag(
      event,
      side,
      "move",
      selected.columnId,
    );
    return;
  }

  // 5) Fallback: grab whichever column is nearest.
  if (!nearest) return;

  if (
    !state.selectedColumn ||
    state.selectedColumn.side !== side ||
    state.selectedColumn.columnId !== nearest.columnId
  ) {
    selectColumn(side, nearest.columnId);
    redrawOverlaySide(side);
  }

  beginOverlayDrag(
    event,
    side,
    "move",
    nearest.columnId,
  );
}

function handleOverlayPointerMove(event, side) {
  if (!state.overlayDrag) {
    updateOverlayHoverCursor(event, side);
    return;
  }

  if (state.overlayDrag.side !== side) return;
  applyOverlayDrag(event);
}

async function handleOverlayPointerUp(event, side) {
  if (!state.overlayDrag || state.overlayDrag.side !== side) return;
  await finishOverlayDrag(event);
}



function extractV3Strip(
  image,
  cols,
  index,
  side = null,
  outputScale = 1,
) {
  const H = image.height;
  const W = image.width;
  const c = cols[index];

  let y0 = Math.round(c.y0);
  let y1 = Math.round(c.y1);

  const baseExtendChars = Number($("columnExtendRange").value);
  const ov = side
    ? getColumnOverride(side, c._columnId)
    : defaultColumnOverride();

  const charH = Math.max(8, c.width);
  const topTotal = baseExtendChars + ov.topChars;
  const bottomTotal = baseExtendChars + ov.bottomChars;

  if (topTotal >= 0) {
    y0 = Math.max(0, Math.round(y0 - charH * topTotal));
  } else {
    y0 = Math.min(y1 - 2, Math.round(y0 + charH * Math.abs(topTotal)));
  }

  if (bottomTotal >= 0) {
    y1 = Math.min(H - 1, Math.round(y1 + charH * bottomTotal));
  } else {
    y1 = Math.max(y0 + 2, Math.round(y1 - charH * Math.abs(bottomTotal)));
  }

  const shiftY = Math.round((ov.shiftYChars || 0) * charH);
  y0 += shiftY;
  y1 += shiftY;

  if (y0 < 0) {
    y1 -= y0;
    y0 = 0;
  }
  if (y1 > H - 1) {
    const overflow = y1 - (H - 1);
    y0 -= overflow;
    y1 = H - 1;
  }

  y0 = Math.max(0, y0);
  y1 = Math.min(H - 1, y1);

  if (y1 - y0 < 3) return null;

  const gap = typicalGap(cols, H);
  const minAllowed = gap * 0.50;
  const maxAllowed = gap * 1.35;
  const scale = clampNumber(Number(outputScale) || 1, 1, 4);

  function geometryAt(y) {
    const baseXc = lineX(c, y);
    const shiftPx = ov.shiftChars * gap;
    const xc = baseXc + shiftPx;

    let rightBoundary;
    if (index > 0) {
      const rightNeighbor = cols[index - 1];
      rightBoundary =
        (baseXc + lineX(rightNeighbor, y)) / 2 + shiftPx;
    } else {
      rightBoundary = baseXc + gap * 0.48 + shiftPx;
    }

    let leftBoundary;
    if (index < cols.length - 1) {
      const leftNeighbor = cols[index + 1];
      leftBoundary =
        (baseXc + lineX(leftNeighbor, y)) / 2 + shiftPx;
    } else {
      leftBoundary = baseXc - gap * 0.48 + shiftPx;
    }

    let xl = Math.min(leftBoundary, rightBoundary);
    let xr = Math.max(leftBoundary, rightBoundary);

    const center = (xl + xr) / 2;
    const half = ((xr - xl) / 2) * ov.widthScale;
    xl = center - half;
    xr = center + half;

    return {
      y,
      xc,
      xl,
      xr,
      wr: xr - xl,
    };
  }

  const rowWidths = [];
  for (let y = y0; y <= y1; y++) {
    const g = geometryAt(y);
    if (Number.isFinite(g.wr) && g.wr > 1) {
      rowWidths.push(g.wr);
    }
  }

  let medianWidth = rowWidths.length ? median(rowWidths) : gap;
  const manualWidth = side && Math.abs(ov.widthScale - 1) > 0.001;

  if (!manualWidth) {
    medianWidth = Math.max(minAllowed, Math.min(maxAllowed, medianWidth));
  } else {
    medianWidth = Math.max(4, medianWidth);
  }

  // Key v5.9 change:
  // 2x/4x rescue is sampled DIRECTLY from the flattened page.
  // We do not first create a 41px strip and then enlarge that bitmap.
  const baseOutW = Math.max(28, Math.round(medianWidth));
  const baseOutH = Math.max(4, y1 - y0 + 1);
  const outW = Math.max(28, Math.round(baseOutW * scale));
  const outH = Math.max(4, Math.round(baseOutH * scale));

  const out = new Uint8Array(outW * outH * 4);
  let di = 0;

  for (let row = 0; row < outH; row++) {
    const ry =
      outH === 1
        ? 0.5
        : row / (outH - 1);

    const sy = y0 + ry * (y1 - y0);
    let { xc, xl, xr, wr } = geometryAt(sy);

    if (
      !Number.isFinite(wr) ||
      (!manualWidth && (wr < minAllowed || wr > maxAllowed))
    ) {
      xl = xc - medianWidth / 2;
      xr = xc + medianWidth / 2;
    }

    for (let x = 0; x < outW; x++) {
      const t = outW === 1 ? 0.5 : x / (outW - 1);
      const sx = xl + t * (xr - xl);

      let rgba;
      if (sx < 0 || sx > W - 1 || sy < 0 || sy > H - 1) {
        rgba = [245, 245, 240, 255];
      } else {
        rgba = bilinearSample(image, sx, sy);
      }

      out[di++] = rgba[0];
      out[di++] = rgba[1];
      out[di++] = rgba[2];
      out[di++] = rgba[3];
    }
  }

  return {
    width: outW,
    height: outH,
    data: out,
    sourceIndex: index,
    y0,
    y1,
    typicalGap: gap,
    outputScale: scale,
  };
}

function renderStrips(strips, side) {
  const root = $(side === "right" ? "rightStrips" : "leftStrips");
  root.innerHTML = "";

  if (!strips.length) {
    root.innerHTML = '<div class="empty">沒有可抽出的直欄</div>';
    return;
  }

  strips.forEach((strip, i) => {
    const card = document.createElement("div");
    card.className = "strip-card";

    const title = document.createElement("h4");
    title.textContent = `Column ${String(i + 1).padStart(2, "0")}`;

    const wrap = document.createElement("div");
    wrap.className = "strip-canvas-wrap";

    const canvas = document.createElement("canvas");
    pixelsToCanvas(strip, canvas);

    const meta = document.createElement("div");
    meta.className = "strip-meta";
    meta.textContent = `${strip.width} × ${strip.height}`;

    card.dataset.side = side;
    card.dataset.columnId = strip.columnId;

    card.title = "點這張 Column 小圖回上方修改框線";
    card.addEventListener("click", () => {
      selectColumn(
        side,
        strip.columnId,
        { scrollToOverlay: true },
      );
    });

    const editTip = document.createElement("div");
    editTip.className = "strip-edit-tip";
    editTip.textContent = "點此欄 → 回上方拖框修正";

    const del = document.createElement("button");
    del.className = "strip-delete";
    del.textContent = "× 刪除此欄";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      removeColumn(side, strip.columnId);
    });

    wrap.appendChild(canvas);
    card.appendChild(title);
    card.appendChild(wrap);
    card.appendChild(meta);
    card.appendChild(editTip);
    card.appendChild(del);
    root.appendChild(card);
  });

  updateSelectedCardStyles();
}

function extractPageStrips(image, detection, side) {
  if (!detection?.cols?.length) return [];

  const cols = activeColumns(side);
  const strips = [];

  cols.forEach((column, index) => {
    const strip = extractV3Strip(image, cols, index, side);
    if (strip) {
      strip.columnId = column._columnId;
      strip.sourceSide = side;
      strips.push(strip);
    }
  });

  renderStrips(strips, side);

  $(side === "right" ? "rightStripStats" : "leftStripStats").textContent =
    `${strips.length}/${detection.cols.length} 欄`;

  return strips;
}

async function runExtraction() {
  if (!state.rightDetection || !state.leftDetection) {
    await runDetector();
  }

  setStatus("V3：抽出右頁直欄…", "scanline corridor + bilinear remap", 35);
  state.rightStrips = extractPageStrips(
    state.rightFlat,
    state.rightDetection,
    "right",
  );

  setStatus("V3：抽出左頁直欄…", "scanline corridor + bilinear remap", 70);
  state.leftStrips = extractPageStrips(
    state.leftFlat,
    state.leftDetection,
    "left",
  );

  setStatus(
    "V3 抽欄完成。",
    "Column 01 現在會顯示在最右邊；請檢查每欄是否完整。",
    100,
  );
}


async function fetchTextCached(url, resourceName) {
  const cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;

  if (cache) {
    const cached = await cache.match(url);
    if (cached) return cached.text();
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`${resourceName} 下載失敗：HTTP ${response.status}`);
  }

  const text = await response.text();

  if (cache) {
    await cache.put(
      url,
      new Response(text, {
        headers: { "content-type": "text/plain;charset=utf-8" },
      }),
    );
  }

  return text;
}

function parsePpOcrV5Dictionary(text) {
  // Do NOT trim the beginning of this file: PP-OCRv5's dictionary may
  // contain whitespace-like characters as real dictionary entries.
  const dictionary = text.split(/\r?\n/);

  // A trailing newline creates one synthetic empty record. Remove only that.
  if (dictionary.length && dictionary[dictionary.length - 1] === "") {
    dictionary.pop();
  }

  // Current official ppocrv5_dict.txt is observed as 18,383 entries.
  // Paddle's PP-OCRv5 config has use_space_char=true, so the normal ASCII
  // space belongs at the end of the recognition character list.
  if (dictionary[dictionary.length - 1] !== " ") {
    dictionary.push(" ");
  }

  // Do not hard-code 18,384 as a minimum. RecognitionService validates
  // against the actual ONNX output class count at inference time.
  if (dictionary.length < 1000) {
    throw new Error(
      `PP-OCRv5 字典讀取異常：只讀到 ${dictionary.length} 個項目。`,
    );
  }

  console.info(`PP-OCRv5 dictionary entries: ${dictionary.length}`);
  return dictionary;
}

function parsePpOcrV6Dictionary(text) {
  const dictionary = text.split(/\r?\n/);

  if (dictionary.length && dictionary[dictionary.length - 1] === "") {
    dictionary.pop();
  }

  // Official ppocrv6_dict.txt has 18,708 entries.
  // PP-OCRv6 uses use_space_char=true, so add normal ASCII space.
  if (dictionary[dictionary.length - 1] !== " ") {
    dictionary.push(" ");
  }

  if (dictionary.length < 18000) {
    throw new Error(
      `PP-OCRv6 字典讀取異常：只讀到 ${dictionary.length} 個項目。`,
    );
  }

  console.info(`PP-OCRv6 dictionary entries: ${dictionary.length}`);
  return dictionary;
}


async function ensureRecognizer(forceWasm = false) {
  const key = $("recModel").value;
  const spec = REC_MODELS[key];

  if (!spec) {
    throw new Error(`未知 recognition 模型：${key}`);
  }

  if (
    state.recognizer &&
    state.recPreset === spec.preset &&
    state.recForcedWasm === forceWasm
  ) {
    return state.recognizer;
  }

  setStatus(
    `準備 ${spec.label}…`,
    forceWasm
      ? "Recognition 已切換為 CPU / WASM fallback。"
      : "Recognition 先嘗試 WebGPU；若實際推理失敗會自動重試 CPU。",
    5,
  );
  await nextPaint();

  const [modelBuffer, dictText] = await Promise.all([
    state.recModelBuffer && state.recSpec?.preset === spec.preset
      ? Promise.resolve(state.recModelBuffer)
      : fetchArrayBufferCached(spec.url, spec.label, 5, 68),
    state.recDictionary
      ? Promise.resolve(null)
      : fetchTextCached(DICT_URL, "PP-OCRv5 字典"),
  ]);

  if (!state.recDictionary) {
    state.recDictionary = parsePpOcrV5Dictionary(dictText);
  }

  let session;
  if (forceWasm) {
    session = await createWasmSession(modelBuffer, spec.label);
  } else {
    session = await createSessionAuto(modelBuffer, spec.label);
  }

  state.recModelBuffer = modelBuffer;
  state.recSpec = spec;
  state.recSession = session;
  state.recPreset = spec.preset;
  state.recForcedWasm = forceWasm;
  state.recognizer = new RecognitionService(ort, session, {
    ...getTextRecognitionPresetOptions(spec.preset),
    charactersDictionary: state.recDictionary,
  });

  setStatus(
    `${spec.label} 已準備完成。`,
    forceWasm ? "Recognition = CPU / WASM fallback" : "",
    90,
  );
  return state.recognizer;
}

function fullVerticalBox(image) {
  // Supplying polygon points is intentional.
  // RecognitionService.cropRotated() automatically rotates a crop
  // counter-clockwise when height/width >= 1.5. For a traditional
  // vertical column this maps original top->bottom to horizontal
  // left->right before PP-OCRv5 recognition.
  return {
    x: 0,
    y: 0,
    width: image.width,
    height: image.height,
    points: [
      { x: 0, y: 0 },
      { x: image.width, y: 0 },
      { x: image.width, y: image.height },
      { x: 0, y: image.height },
    ],
  };
}

function stripChannels(image) {
  const px = image.width * image.height;
  if (image.data.length === px * 4) return 4;
  if (image.data.length === px * 3) return 3;
  return 1;
}

function stripGray(image) {
  const channels = stripChannels(image);
  const out = new Uint8Array(image.width * image.height);

  for (let i = 0; i < out.length; i++) {
    if (channels === 1) {
      out[i] = image.data[i];
    } else {
      const si = i * channels;
      const r = image.data[si];
      const g = image.data[si + 1];
      const b = image.data[si + 2];
      out[i] = Math.max(
        0,
        Math.min(255, Math.round(0.299 * r + 0.587 * g + 0.114 * b)),
      );
    }
  }
  return out;
}

function histogramPercentile(hist, total, fraction) {
  const target = total * fraction;
  let seen = 0;
  for (let i = 0; i < 256; i++) {
    seen += hist[i];
    if (seen >= target) return i;
  }
  return 255;
}

function grayBoxBlur3(gray, width, height) {
  const out = new Float32Array(gray.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;

      for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
        for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
          sum += gray[yy * width + xx];
          count++;
        }
      }

      out[y * width + x] = sum / count;
    }
  }
  return out;
}

function resizePixelsBilinear(image, scale) {
  if (scale === 1) return image;

  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const data = new Uint8Array(width * height * 4);

  let di = 0;
  for (let y = 0; y < height; y++) {
    const sy = height === 1 ? 0 : y * (image.height - 1) / (height - 1);

    for (let x = 0; x < width; x++) {
      const sx = width === 1 ? 0 : x * (image.width - 1) / (width - 1);
      const rgba = bilinearSample(image, sx, sy);

      data[di++] = rgba[0];
      data[di++] = rgba[1];
      data[di++] = rgba[2];
      data[di++] = 255;
    }
  }

  return { width, height, data };
}

function enhanceStripForOcr(strip, mode = "normal") {
  const scale = mode === "strong" ? 3 : 2;
  const enlarged = resizePixelsBilinear(strip, scale);

  const width = enlarged.width;
  const height = enlarged.height;
  const gray = stripGray(enlarged);

  const hist = new Uint32Array(256);
  for (const v of gray) hist[v]++;

  const lowFrac = mode === "strong" ? 0.035 : 0.02;
  const highFrac = mode === "strong" ? 0.965 : 0.985;

  let low = histogramPercentile(hist, gray.length, lowFrac);
  let high = histogramPercentile(hist, gray.length, highFrac);

  if (high - low < 45) {
    low = Math.max(0, low - 20);
    high = Math.min(255, high + 20);
  }

  const stretched = new Uint8Array(gray.length);
  const gamma = mode === "strong" ? 1.10 : 1.04;

  for (let i = 0; i < gray.length; i++) {
    let t = (gray[i] - low) / Math.max(1, high - low);
    t = Math.max(0, Math.min(1, t));
    t = Math.pow(t, gamma);
    stretched[i] = Math.round(t * 255);
  }

  const blur = grayBoxBlur3(stretched, width, height);
  const amount = mode === "strong" ? 1.05 : 0.72;
  const output = new Uint8Array(width * height * 4);

  let di = 0;
  for (let i = 0; i < stretched.length; i++) {
    let v = stretched[i] + amount * (stretched[i] - blur[i]);

    if (v > 205) {
      v = 205 + (v - 205) * 1.35;
    }

    v = Math.max(0, Math.min(255, Math.round(v)));
    output[di++] = v;
    output[di++] = v;
    output[di++] = v;
    output[di++] = 255;
  }

  return { width, height, data: output };
}

function coreTextLength(text) {
  return [...String(text || "")
    .replace(/[\s，。；：、,.!?！？「」『』（）()《》〈〉【】〔〕—…·・：；]/g, "")]
    .length;
}

function estimateShortColumnChars(strip) {
  const ratio = strip.height / Math.max(1, strip.width);
  const estimate = Math.max(1, Math.round(ratio * 0.96));

  if (estimate > 12 || strip.height > 360) return null;
  return estimate;
}

function chooseRecognitionCandidate(candidates, strip) {
  const expectedChars = estimateShortColumnChars(strip);
  const normalizedCounts = new Map();

  for (const candidate of candidates) {
    const key = candidate.text.trim();
    if (key) normalizedCounts.set(key, (normalizedCounts.get(key) || 0) + 1);
  }

  const scored = candidates.map((candidate) => {
    const text = candidate.text.trim();
    const charCount = coreTextLength(text);
    let adjustedScore = Number(candidate.confidence || 0);

    const consensus = normalizedCounts.get(text) || 0;
    if (text && consensus >= 2) {
      adjustedScore += 0.035 * (consensus - 1);
    }

    if (expectedChars !== null && text) {
      const diff = Math.abs(charCount - expectedChars);
      adjustedScore -= Math.min(0.22, diff * 0.06);
      if (diff === 0) adjustedScore += 0.025;
    }

    if (!text) adjustedScore -= 0.30;

    return {
      ...candidate,
      charCount,
      adjustedScore,
    };
  });

  scored.sort((a, b) => b.adjustedScore - a.adjustedScore);

  return {
    ...scored[0],
    expectedChars,
    candidates: scored,
  };
}





function v6RescueMode() {
  return $("v6TextRescue")?.value || "off";
}

function v6RescueTarget(strip) {
  const mode = v6RescueMode();
  if (mode === "off") return false;

  const expected = estimateShortColumnChars(strip);
  if (expected === null || expected > 6) return false;
  if (strip.height > 430 || strip.width > 105) return false;

  return true;
}

async function ensureV6RescueRecognizer() {
  const key = v6RescueMode();
  const spec = V6_RESCUE_MODELS[key];

  if (!spec) {
    throw new Error(`未知 PP-OCRv6 rescue 模式：${key}`);
  }

  if (
    state.v6RescueRecognizer &&
    state.v6RescueKey === key
  ) {
    return state.v6RescueRecognizer;
  }

  const started = performance.now();

  setStatus(
    `準備 ${spec.label}…`,
    key === "small6"
      ? "首次約 21 MB；只用於 1～6 字短欄。"
      : "首次約 77 MB；精度優先，只用於 1～6 字短欄。",
    8,
  );
  await nextPaint();

  // Switching small <-> medium requires a different ONNX graph,
  // but both share the same official ppocrv6 dictionary.
  const [modelBuffer, dictText] = await Promise.all([
    state.v6RescueModelBuffer && state.v6RescueKey === key
      ? Promise.resolve(state.v6RescueModelBuffer)
      : fetchArrayBufferCached(
          spec.url,
          spec.label,
          8,
          key === "small6" ? 42 : 62,
        ),
    state.v6RescueDictionary
      ? Promise.resolve(null)
      : fetchTextCached(V6_DICT_URL, "PP-OCRv6 字典"),
  ]);

  if (!state.v6RescueDictionary) {
    state.v6RescueDictionary = parsePpOcrV6Dictionary(dictText);
  }

  if (state.v6RescueSession) {
    try {
      state.v6RescueSession.release?.();
    } catch {}
  }

  // v6 small is modest enough to try WebGPU.
  // v6 medium also tries WebGPU and createSessionAuto falls back to WASM
  // if session creation fails.
  const session = await createSessionAuto(
    modelBuffer,
    spec.label,
  );

  state.v6RescueModelBuffer = modelBuffer;
  state.v6RescueSession = session;
  state.v6RescueKey = key;
  state.v6RescueRecognizer = new RecognitionService(
    ort,
    session,
    {
      ...getTextRecognitionPresetOptions(spec.preset),
      charactersDictionary: state.v6RescueDictionary,
    },
  );

  if (state.recPerf) {
    state.recPerf.v6SetupMs += performance.now() - started;
  }

  return state.v6RescueRecognizer;
}

async function recognizeWithV6Once(strip) {
  const recognizer = await ensureV6RescueRecognizer();
  await nextPaint();

  const started = performance.now();
  const variant =
    v6RescueMode() === "medium6"
      ? "PP-OCRv6 medium"
      : "PP-OCRv6 small";

  const result = await recognizeStripOnce(
    strip,
    recognizer,
    variant,
  );

  if (state.recPerf) {
    state.recPerf.v6Runs += 1;
    state.recPerf.v6InferenceMs += performance.now() - started;
  }

  return result;
}

function chooseCurrentVsV6(currentBest, v6Candidate, strip) {
  const expected = estimateShortColumnChars(strip);
  const aText = String(currentBest?.text || "").trim();
  const bText = String(v6Candidate?.text || "").trim();

  if (!bText) return currentBest;
  if (!aText) return v6Candidate;

  const aConf = Number(currentBest?.confidence || 0);
  const bConf = Number(v6Candidate?.confidence || 0);

  if (aText === bText) {
    return bConf > aConf ? v6Candidate : currentBest;
  }

  if (expected !== null) {
    const aDiff = Math.abs(coreTextLength(aText) - expected);
    const bDiff = Math.abs(coreTextLength(bText) - expected);

    if (bDiff < aDiff && bConf >= 0.45) {
      return v6Candidate;
    }

    if (
      bDiff === 0 &&
      aDiff === 0 &&
      bConf >= aConf + 0.01
    ) {
      return v6Candidate;
    }
  }

  if (bConf >= 0.88 && bConf >= aConf - 0.03) {
    return v6Candidate;
  }

  if (bConf >= aConf + 0.04) {
    return v6Candidate;
  }

  return currentBest;
}

function serverRescueMode() {
  return $("serverTextRescue")?.value || "off";
}

function serverRescueTarget(strip, currentBest, candidates) {
  const mode = serverRescueMode();

  // If the user already chose server as the main recognizer,
  // another server pass would be redundant.
  if ($("recModel")?.value === "server") return false;
  if (mode === "off") return false;

  const expected = estimateShortColumnChars(strip);
  if (expected === null) return false;
  if (strip.height > 430 || strip.width > 100) return false;

  if (mode === "short6") {
    return expected <= 6;
  }

  if (mode === "smart10") {
    if (expected <= 6) return true;
    if (expected > 10) return false;

    const text = String(currentBest?.text || "").trim();
    const chars = coreTextLength(text);
    const confidence = Number(currentBest?.confidence || 0);

    return (
      confidence < 0.90 ||
      candidateDisagreement(candidates) ||
      Math.abs(chars - expected) >= 1
    );
  }

  return false;
}

async function ensureServerRescueRecognizer() {
  if (state.serverRescueRecognizer) {
    return state.serverRescueRecognizer;
  }

  const spec = REC_MODELS.server;
  const started = performance.now();

  setStatus(
    "準備 PP-OCRv5 server hard-case recognizer…",
    "首次約 84.5 MB。為避免大型 WebGPU session 初始化卡住，救援模型固定使用 CPU / WASM；只跑少量短欄。",
    8,
  );
  await nextPaint();

  const [modelBuffer, dictText] = await Promise.all([
    state.serverRescueModelBuffer
      ? Promise.resolve(state.serverRescueModelBuffer)
      : fetchArrayBufferCached(
          spec.url,
          `${spec.label} · hard-case`,
          8,
          60,
        ),
    state.recDictionary
      ? Promise.resolve(null)
      : fetchTextCached(DICT_URL, "PP-OCRv5 字典"),
  ]);

  if (!state.recDictionary) {
    state.recDictionary = parsePpOcrV5Dictionary(dictText);
  }

  // We intentionally use WASM here. The server model is large and earlier
  // browser tests could stall during WebGPU session compilation.
  const session = await createWasmSession(
    modelBuffer,
    `${spec.label} · hard-case`,
  );

  state.serverRescueModelBuffer = modelBuffer;
  state.serverRescueSession = session;
  state.serverRescueRecognizer = new RecognitionService(ort, session, {
    ...getTextRecognitionPresetOptions(spec.preset),
    charactersDictionary: state.recDictionary,
  });

  if (state.recPerf) {
    state.recPerf.serverSetupMs += performance.now() - started;
  }

  return state.serverRescueRecognizer;
}

async function recognizeWithServerOnce(strip) {
  const recognizer = await ensureServerRescueRecognizer();
  await nextPaint();

  const started = performance.now();
  const result = await recognizeStripOnce(
    strip,
    recognizer,
    "Server原圖",
  );
  const elapsed = performance.now() - started;

  if (state.recPerf) {
    state.recPerf.serverRuns += 1;
    state.recPerf.serverInferenceMs += elapsed;
  }

  return result;
}

function chooseMobileVsServer(mobileBest, serverCandidate, strip) {
  const expected = estimateShortColumnChars(strip);
  const mobileText = String(mobileBest?.text || "").trim();
  const serverText = String(serverCandidate?.text || "").trim();

  if (!serverText) return mobileBest;
  if (!mobileText) return serverCandidate;
  if (serverText === mobileText) {
    return Number(serverCandidate.confidence || 0) >
      Number(mobileBest.confidence || 0)
      ? serverCandidate
      : mobileBest;
  }

  const mobileConf = Number(mobileBest?.confidence || 0);
  const serverConf = Number(serverCandidate?.confidence || 0);
  const mobileChars = coreTextLength(mobileText);
  const serverChars = coreTextLength(serverText);

  // The server result is not thrown into the mobile A/B majority vote.
  // We compare recognizers separately, otherwise 3 similar mobile variants
  // can outvote one correct server answer.
  if (expected !== null) {
    const mobileDiff = Math.abs(mobileChars - expected);
    const serverDiff = Math.abs(serverChars - expected);

    if (serverDiff < mobileDiff && serverConf >= 0.50) {
      return serverCandidate;
    }

    if (
      serverDiff === 0 &&
      mobileDiff === 0 &&
      serverConf >= mobileConf + 0.015
    ) {
      return serverCandidate;
    }

    if (
      serverDiff === 0 &&
      mobileConf < 0.80 &&
      serverConf >= mobileConf - 0.05
    ) {
      return serverCandidate;
    }
  }

  if (serverConf >= 0.90 && serverConf >= mobileConf - 0.02) {
    return serverCandidate;
  }

  if (serverConf >= mobileConf + 0.05) {
    return serverCandidate;
  }

  return mobileBest;
}

function cnnRescueMode() {
  return $("cnnTextRescue")?.value || "off";
}

function cnnRescueTarget(strip) {
  const expected = estimateShortColumnChars(strip);

  // CNN is deliberately limited to SMALL, SHORT strips.
  // This is the main latency guard: never run Real-ESRGAN on a long page column.
  if (expected === null) return false;
  if (expected > 10) return false;
  if (strip.height > 420) return false;
  if (strip.width > 90) return false;

  return true;
}

function cnnResultLooksSuspicious(strip, chosen, candidates) {
  const expected = estimateShortColumnChars(strip);
  const text = String(chosen?.text || "").trim();
  const chars = coreTextLength(text);
  const confidence = Number(chosen?.confidence || 0);
  const disagreement = candidateDisagreement(candidates);

  // Very short labels/titles are exactly the failure mode we are targeting
  // ("臺灣史" etc.). They always get ONE learned-restoration attempt.
  if (expected !== null && expected <= 6) return true;

  // For longer short columns, only pay CNN cost if there is evidence of trouble.
  if (confidence < 0.90) return true;
  if (disagreement) return true;

  if (expected !== null && Math.abs(chars - expected) >= 1) return true;
  if (expected !== null && expected >= 2 && chars <= 1) return true;

  return false;
}

function stripToRgbTensor(strip) {
  const channels = stripChannels(strip);
  const plane = strip.width * strip.height;
  const data = new Float32Array(plane * 3);

  for (let i = 0; i < plane; i++) {
    let r;
    let g;
    let b;

    if (channels === 1) {
      r = g = b = strip.data[i];
    } else {
      const si = i * channels;
      r = strip.data[si];
      g = strip.data[si + 1];
      b = strip.data[si + 2];
    }

    data[i] = r / 255;
    data[plane + i] = g / 255;
    data[plane * 2 + i] = b / 255;
  }

  return new ort.Tensor(
    "float32",
    data,
    [1, 3, strip.height, strip.width],
  );
}

function srTensorToPixels(tensor) {
  const dims = tensor.dims || [];
  if (dims.length !== 4 || Number(dims[0]) !== 1 || Number(dims[1]) !== 3) {
    throw new Error(`CNN SR 輸出 shape 異常：${dims.join("×")}`);
  }

  const height = Number(dims[2]);
  const width = Number(dims[3]);
  const plane = width * height;
  const src = tensor.data;
  const out = new Uint8Array(plane * 4);

  let di = 0;
  for (let i = 0; i < plane; i++) {
    const r = Math.max(0, Math.min(1, Number(src[i])));
    const g = Math.max(0, Math.min(1, Number(src[plane + i])));
    const b = Math.max(0, Math.min(1, Number(src[plane * 2 + i])));

    out[di++] = Math.round(r * 255);
    out[di++] = Math.round(g * 255);
    out[di++] = Math.round(b * 255);
    out[di++] = 255;
  }

  return { width, height, data: out };
}

async function ensureSrSession(forceWasm = false) {
  if (
    state.srSession &&
    state.srForcedWasm === forceWasm
  ) {
    return state.srSession;
  }

  const setupStart = performance.now();

  if (!state.srModelBuffer) {
    state.srModelBuffer = await fetchArrayBufferCached(
      CNN_SR_MODEL.url,
      CNN_SR_MODEL.label,
      5,
      22,
    );
  }

  if (state.srSession) {
    try {
      state.srSession.release?.();
    } catch {}
    state.srSession = null;
  }

  state.srSession = forceWasm
    ? await createWasmSession(
        state.srModelBuffer,
        CNN_SR_MODEL.label,
      )
    : await createSessionAuto(
        state.srModelBuffer,
        CNN_SR_MODEL.label,
      );

  state.srForcedWasm = forceWasm;

  if (state.recPerf) {
    state.recPerf.cnnSetupMs += performance.now() - setupStart;
  }

  return state.srSession;
}

async function runCnnSuperResolutionOnce(strip, forceWasm = false) {
  const session = await ensureSrSession(forceWasm);
  const inputName = session.inputNames?.[0] || "input";
  const outputName = session.outputNames?.[0] || "output";
  const inputTensor = stripToRgbTensor(strip);

  await nextPaint();
  const inferStart = performance.now();

  const outputs = await session.run({
    [inputName]: inputTensor,
  });

  const elapsed = performance.now() - inferStart;

  if (state.recPerf) {
    state.recPerf.cnnRuns += 1;
    state.recPerf.cnnInferenceMs += elapsed;
  }

  const outputTensor =
    outputs[outputName] ||
    outputs[Object.keys(outputs)[0]];

  if (!outputTensor) {
    throw new Error("CNN SR 沒有回傳 output tensor。");
  }

  return srTensorToPixels(outputTensor);
}

async function runCnnSuperResolution(strip) {
  try {
    return await runCnnSuperResolutionOnce(strip, false);
  } catch (gpuError) {
    const backend = state.backends[CNN_SR_MODEL.label];

    if (backend !== "webgpu") {
      throw gpuError;
    }

    console.warn(
      "CNN SR WebGPU inference failed; retrying WASM.",
      gpuError,
    );

    if (state.recPerf) {
      state.recPerf.cnnFailures += 1;
    }

    try {
      state.srSession?.release?.();
    } catch {}
    state.srSession = null;

    return runCnnSuperResolutionOnce(strip, true);
  }
}

function rescueMode() {
  return $("smallTextRescue")?.value || "off";
}

function rescueEligible(strip) {
  const expected = estimateShortColumnChars(strip);

  return (
    strip.width < 70 ||
    (expected !== null && expected <= 8)
  );
}

function rebuildStripAtScale(strip, scale) {
  const side = strip.sourceSide;
  const columnId = strip.columnId;

  if (!side || !columnId) return null;

  const cols = activeColumns(side);
  const index = cols.findIndex((c) => c._columnId === columnId);
  if (index < 0) return null;

  const image = side === "right"
    ? state.rightFlat
    : state.leftFlat;

  if (!image) return null;

  const hi = extractV3Strip(
    image,
    cols,
    index,
    side,
    scale,
  );

  if (!hi) return null;

  hi.columnId = columnId;
  hi.sourceSide = side;
  return hi;
}

function candidateDisagreement(candidates) {
  const texts = new Set(
    (candidates || [])
      .map((c) => String(c.text || "").trim())
      .filter(Boolean),
  );
  return texts.size >= 2;
}

async function recognizeStripWithRescue(strip, recognizer) {
  const base = await recognizeStrip(strip, recognizer);
  const scaleMode = rescueMode();
  const cnnMode = cnnRescueMode();
  const small = rescueEligible(strip);

  const candidates = [
    ...(Array.isArray(base.candidates) ? base.candidates : [base]),
  ];

  let scaleRescueRan = false;

  const shouldRunScale =
    scaleMode === "all2" ||
    (
      small &&
      (
        scaleMode === "auto2" ||
        scaleMode === "strong"
      )
    );

  if (shouldRunScale) {
    const twoX = rebuildStripAtScale(strip, 2);
    if (twoX) {
      candidates.push(
        await recognizeStripOnce(
          twoX,
          recognizer,
          "高解析2×",
        ),
      );
      scaleRescueRan = true;
    }

    if (scaleMode === "strong" && small) {
      const fourX = rebuildStripAtScale(strip, 4);
      if (fourX) {
        candidates.push(
          await recognizeStripOnce(
            fourX,
            recognizer,
            "高解析4×",
          ),
        );
        scaleRescueRan = true;
      }
    }
  }

  let currentBest = chooseRecognitionCandidate(
    candidates,
    strip,
  );

  let cnnRan = false;
  let cnnError = null;

  const cnnTarget = cnnRescueTarget(strip);
  const shouldRunCnn =
    cnnMode !== "off" &&
    cnnTarget &&
    (
      cnnMode === "short" ||
      cnnResultLooksSuspicious(strip, currentBest, candidates)
    );

  if (shouldRunCnn) {
    try {
      const cnnStrip = await runCnnSuperResolution(strip);
      const cnnCandidate = await recognizeStripOnce(
        cnnStrip,
        recognizer,
        "CNN4×",
      );
      candidates.push(cnnCandidate);
      cnnRan = true;
      currentBest = chooseRecognitionCandidate(candidates, strip);
    } catch (error) {
      cnnError =
        error instanceof Error ? error.message : String(error);
      console.warn("Selective CNN rescue failed.", error);
      if (state.recPerf) state.recPerf.cnnFailures += 1;
    }
  }

  let serverRan = false;
  let serverError = null;
  let serverCandidate = null;

  if (serverRescueTarget(strip, currentBest, candidates)) {
    try {
      setStatus(
        "高精度 hard-case：PP-OCRv5 server 比較中…",
        `只重認目前短欄 ${strip.width}×${strip.height}。`,
        null,
      );
      serverCandidate = await recognizeWithServerOnce(strip);
      candidates.push(serverCandidate);
      serverRan = true;
    } catch (error) {
      serverError =
        error instanceof Error ? error.message : String(error);
      console.warn("Server hard-case rescue failed.", error);
      if (state.recPerf) state.recPerf.serverFailures += 1;
    }
  }

  // Important: heavy recognizers do not participate in mobile A/B majority voting.
  // Compare recognizers independently.
  let recognizerBest = serverCandidate
    ? chooseMobileVsServer(currentBest, serverCandidate, strip)
    : currentBest;

  let v6Ran = false;
  let v6Error = null;
  let v6Candidate = null;

  if (v6RescueTarget(strip)) {
    try {
      setStatus(
        "最新模型 hard-case：PP-OCRv6 比較中…",
        `只重認目前 ${estimateShortColumnChars(strip) ?? "短"} 字欄。`,
        null,
      );

      v6Candidate = await recognizeWithV6Once(strip);
      candidates.push(v6Candidate);
      v6Ran = true;

      recognizerBest = chooseCurrentVsV6(
        recognizerBest,
        v6Candidate,
        strip,
      );
    } catch (error) {
      v6Error =
        error instanceof Error ? error.message : String(error);
      console.warn("PP-OCRv6 hard-case rescue failed.", error);
      if (state.recPerf) state.recPerf.v6Failures += 1;
    }
  }

  const chosen = recognizerBest;

  return {
    ...chosen,
    expectedChars:
      chosen.expectedChars ??
      estimateShortColumnChars(strip),
    candidates,
    rescueModeUsed: scaleMode,
    rescueRan: scaleRescueRan,
    cnnRescueModeUsed: cnnMode,
    cnnRan,
    cnnError,
    serverRescueModeUsed: serverRescueMode(),
    serverRan,
    serverError,
    serverText: serverCandidate?.text ?? null,
    serverConfidence:
      serverCandidate
        ? Number(serverCandidate.confidence || 0)
        : null,
    mobileText: currentBest?.text ?? "",
    mobileConfidence: Number(currentBest?.confidence || 0),
    v6RescueModeUsed: v6RescueMode(),
    v6Ran,
    v6Error,
    v6Text: v6Candidate?.text ?? null,
    v6Confidence:
      v6Candidate
        ? Number(v6Candidate.confidence || 0)
        : null,
    baseText: base.text,
    baseConfidence: base.confidence,
    rescueChangedText:
      String(chosen.text || "").trim() !==
      String(base.text || "").trim(),
    rescueDisagreement: candidateDisagreement(candidates),
  };
}

async function recognizeStripOnce(strip, recognizer, variantName) {
  const image = normalizeInputToRgb(strip);
  const box = fullVerticalBox(image);

  const results = await recognizer.run(image, [box], {
    ordering: { sortByReadingOrder: false },
  });

  const result = results[0];

  return {
    variant: variantName,
    text: result?.text ?? "",
    confidence: Number(result?.confidence ?? 0),
  };
}

async function recognizeStrip(strip, recognizer) {
  const useEnhancement = $("enhanceOcr")?.checked ?? true;

  if (!useEnhancement) {
    const single = await recognizeStripOnce(strip, recognizer, "原圖");
    return {
      ...single,
      expectedChars: estimateShortColumnChars(strip),
      candidates: [single],
    };
  }

  const candidates = [];

  candidates.push(
    await recognizeStripOnce(strip, recognizer, "原圖"),
  );

  const enhanced = enhanceStripForOcr(strip, "normal");
  candidates.push(
    await recognizeStripOnce(enhanced, recognizer, "增強"),
  );

  if (estimateShortColumnChars(strip) !== null) {
    const strong = enhanceStripForOcr(strip, "strong");
    candidates.push(
      await recognizeStripOnce(strong, recognizer, "增強+"),
    );
  }

  return chooseRecognitionCandidate(candidates, strip);
}

function renderRecognition(items, side) {
  const root = $(side === "right" ? "rightRecognition" : "leftRecognition");
  root.innerHTML = "";

  if (!items.length) {
    root.innerHTML = '<div class="empty">沒有辨識結果</div>';
    return;
  }

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "recognition-item";

    const name = document.createElement("div");
    name.className = "col-name";
    name.textContent = `Column ${String(index + 1).padStart(2, "0")}`;
    if (item.columnId) {
      name.style.cursor = "pointer";
      name.title = "點一下編輯這一欄";
      name.addEventListener("click", () =>
        selectColumn(side, item.columnId, { scrollToOverlay: true })
      );
    }

    const text = document.createElement("div");
    text.className = "rec-text";
    text.textContent = item.text || "（空白）";

    const confidence = document.createElement("div");
    confidence.className = "confidence";
    confidence.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    row.appendChild(name);
    row.appendChild(text);
    row.appendChild(confidence);

    if (Array.isArray(item.candidates) && item.candidates.length > 1) {
      const info = document.createElement("div");
      info.className = "variant-info";

      const parts = item.candidates.map((candidate) => {
        const picked = candidate.variant === item.variant ? "✓ " : "";
        return `${picked}${candidate.variant} ${(candidate.confidence * 100).toFixed(1)}%`;
      });

      const expected =
        item.expectedChars !== null && item.expectedChars !== undefined
          ? ` · 短欄估計約 ${item.expectedChars} 字`
          : "";

      info.innerHTML =
        `<strong>選 ${item.variant}</strong> · ${parts.join(" / ")}${expected}`;

      info.title = item.candidates
        .map(
          (candidate) =>
            `${candidate.variant}: ${candidate.text} (${(candidate.confidence * 100).toFixed(1)}%)`,
        )
        .join("\n");

      row.appendChild(info);
    }

    root.appendChild(row);
  });
}
async function recognizePage(strips, side, recognizer, startProgress, span) {
  const results = [];

  for (let i = 0; i < strips.length; i++) {
    setStatus(
      `Recognition：${side === "right" ? "右頁" : "左頁"} Column ${i + 1}/${strips.length}`,
      `${
        $("enhanceOcr")?.checked ? "A/B" : "單次"
      } · 重採樣 ${
        {
          off: "關",
          auto2: "Auto 2×",
          strong: "2×+4×",
          all2: "全部 2×",
        }[rescueMode()] || rescueMode()
      } · CNN ${
        {
          off: "關",
          smart: "智慧",
          short: "短欄全開",
        }[cnnRescueMode()] || cnnRescueMode()
      } · Server ${
        {
          off: "關",
          short6: "≤6字",
          smart10: "智慧≤10字",
        }[serverRescueMode()] || serverRescueMode()
      } · v6 ${
        {
          off: "關",
          small6: "Small≤6字",
          medium6: "Medium≤6字",
        }[v6RescueMode()] || v6RescueMode()
      }`,
      startProgress + span * ((i + 1) / Math.max(1, strips.length)),
    );

    const result = await recognizeStripWithRescue(strips[i], recognizer);
    results.push({
      ...result,
      columnId: strips[i].columnId,
    });
  }

  renderRecognition(results, side);

  const nonEmpty = results.filter((x) => x.text.trim()).length;
  $(side === "right" ? "rightRecStats" : "leftRecStats").textContent =
    `${nonEmpty}/${results.length} 欄有文字`;

  return results;
}

function assembleFullText() {
  // Traditional book reading order:
  // right page first, then left page;
  // strips are already sorted right -> left within each page.
  const right = state.rightRecognition
    .map((x) => x.text.trim())
    .filter(Boolean);

  const left = state.leftRecognition
    .map((x) => x.text.trim())
    .filter(Boolean);

  const pageParts = [];
  if (right.length) pageParts.push(right.join("\n"));
  if (left.length) pageParts.push(left.join("\n"));

  const text = pageParts.join("\n\n");
  $("fullText").value = text;
  $("copyTextBtn").disabled = !text;
  $("downloadTextBtn").disabled = !text;

  return text;
}

async function runRecognition() {
  state.recPerf = freshRecPerf();
  updateRecTiming();

  if (!state.rightStrips.length || !state.leftStrips.length) {
    await runExtraction();
  }

  let recognizer = await ensureRecognizer(false);

  try {
    state.rightRecognition = await recognizePage(
      state.rightStrips,
      "right",
      recognizer,
      10,
      40,
    );

    state.leftRecognition = await recognizePage(
      state.leftStrips,
      "left",
      recognizer,
      52,
      40,
    );
  } catch (gpuError) {
    const currentBackend = state.recSpec
      ? state.backends[state.recSpec.label]
      : undefined;

    if (currentBackend !== "webgpu") {
      throw gpuError;
    }

    console.warn(
      "Recognition WebGPU runtime failed. Rebuilding recognizer on WASM and retrying.",
      gpuError,
    );

    setStatus(
      "Recognition GPU 推理失敗，正在自動改用 CPU / WASM…",
      gpuError instanceof Error
        ? `${gpuError.name}: ${gpuError.message}`
        : String(gpuError),
      5,
    );
    await nextPaint();

    try {
      state.recSession?.release?.();
    } catch {}
    state.recognizer = null;
    state.recSession = null;
    state.recPreset = null;

    recognizer = await ensureRecognizer(true);

    // Retry both pages from the beginning so output stays deterministic.
    state.rightRecognition = await recognizePage(
      state.rightStrips,
      "right",
      recognizer,
      10,
      40,
    );

    state.leftRecognition = await recognizePage(
      state.leftStrips,
      "left",
      recognizer,
      52,
      40,
    );
  }

  const text = assembleFullText();

  const totalRecognitionMs =
    state.recPerf
      ? performance.now() - state.recPerf.startedAt
      : 0;
  updateRecTiming(totalRecognitionMs);

  setStatus(
    "OCR 辨識完成。",
    text
      ? (
          state.recForcedWasm
            ? "Recognition 使用 CPU / WASM fallback；全文已組合。"
            : "Recognition 使用 WebGPU；全文已組合。"
        )
      : "模型完成推理，但沒有得到文字。",
    100,
  );
}


function directPassMode() {
  return $("directPassMode")?.value || "two";
}

function resizePixelImage(image, scale) {
  scale = Math.max(0.25, Number(scale) || 1);

  if (Math.abs(scale - 1) < 0.001) {
    return { image, scale: 1 };
  }

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = image.width;
  srcCanvas.height = image.height;
  pixelsToCanvas(image, srcCanvas);

  const outCanvas = document.createElement("canvas");
  outCanvas.width = Math.max(1, Math.round(image.width * scale));
  outCanvas.height = Math.max(1, Math.round(image.height * scale));

  const ctx = outCanvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(srcCanvas, 0, 0, outCanvas.width, outCanvas.height);

  return {
    image: canvasToPixels(outCanvas),
    scale,
  };
}

function scaleDetectorBox(box, scale) {
  const s = Number(scale) || 1;
  if (Math.abs(s - 1) < 0.001) return { ...box };

  const out = { ...box };

  if (Array.isArray(box.points)) {
    out.points = box.points.map((p) => ({
      ...p,
      x: Number(p.x) / s,
      y: Number(p.y) / s,
    }));
  }

  if (Array.isArray(box.polygon)) {
    out.polygon = box.polygon.map((p) => ({
      ...p,
      x: Number(p.x) / s,
      y: Number(p.y) / s,
    }));
  }

  if (Number.isFinite(Number(box.x))) out.x = Number(box.x) / s;
  if (Number.isFinite(Number(box.y))) out.y = Number(box.y) / s;
  if (Number.isFinite(Number(box.width))) out.width = Number(box.width) / s;
  if (Number.isFinite(Number(box.height))) out.height = Number(box.height) / s;

  return out;
}

function directRectIoU(a, b) {
  const ax0 = a.minX;
  const ax1 = a.maxX;
  const ay0 = a.minY;
  const ay1 = a.maxY;
  const bx0 = b.minX;
  const bx1 = b.maxX;
  const by0 = b.minY;
  const by1 = b.maxY;

  const ix0 = Math.max(ax0, bx0);
  const ix1 = Math.min(ax1, bx1);
  const iy0 = Math.max(ay0, by0);
  const iy1 = Math.min(ay1, by1);

  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  if (inter <= 0) return 0;

  const areaA = Math.max(1, a.width * a.height);
  const areaB = Math.max(1, b.width * b.height);
  return inter / Math.max(1, areaA + areaB - inter);
}

function directContainment(inner, outer) {
  const ix0 = Math.max(inner.minX, outer.minX);
  const ix1 = Math.min(inner.maxX, outer.maxX);
  const iy0 = Math.max(inner.minY, outer.minY);
  const iy1 = Math.min(inner.maxY, outer.maxY);

  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;

  return inter / Math.max(1, inner.width * inner.height);
}


function directInkGuardMode() {
  return $("directInkGuard")?.value || "balanced";
}

function percentileSorted(values, q) {
  if (!values.length) return 0;

  const index =
    Math.max(
      0,
      Math.min(
        values.length - 1,
        Math.round((values.length - 1) * q),
      ),
    );

  return values[index];
}

function measureDirectInkEvidence(image, box) {
  const pts = boxPoints(box);
  const b = directBoxInfo(box);

  if (pts.length < 3 || b.width < 1 || b.height < 1) {
    return {
      valid: false,
      contrast: 0,
      strongRatio: 0,
      mediumRatio: 0,
      weightedDarkness: 0,
      activeBandRatio: 0,
      samples: 0,
    };
  }

  // Keep the measurement cheap even for full-page-height boxes.
  const targetSamples = 9000;
  const area = Math.max(1, b.width * b.height);
  const step = Math.max(
    1,
    Math.floor(Math.sqrt(area / targetSamples)),
  );

  const grayscale = [];
  const bandCount = 12;
  const bandStrong = new Array(bandCount).fill(0);
  const bandSamples = new Array(bandCount).fill(0);

  for (
    let y = Math.max(0, Math.floor(b.minY));
    y <= Math.min(image.height - 1, Math.ceil(b.maxY));
    y += step
  ) {
    for (
      let x = Math.max(0, Math.floor(b.minX));
      x <= Math.min(image.width - 1, Math.ceil(b.maxX));
      x += step
    ) {
      if (!pointInPolygon(x, y, pts)) continue;

      const g = grayAt(image, x, y);
      grayscale.push(g);

      const relativeY =
        (y - b.minY) /
        Math.max(1, b.height);
      const band = Math.max(
        0,
        Math.min(
          bandCount - 1,
          Math.floor(relativeY * bandCount),
        ),
      );

      bandSamples[band] += 1;
    }
  }

  if (grayscale.length < 20) {
    return {
      valid: false,
      contrast: 0,
      strongRatio: 0,
      mediumRatio: 0,
      weightedDarkness: 0,
      activeBandRatio: 0,
      samples: grayscale.length,
    };
  }

  grayscale.sort((a, b) => a - b);

  const p10 = percentileSorted(grayscale, 0.10);
  const p20 = percentileSorted(grayscale, 0.20);
  const p50 = percentileSorted(grayscale, 0.50);
  const p85 = percentileSorted(grayscale, 0.85);
  const p92 = percentileSorted(grayscale, 0.92);

  const background = Math.max(p85, p92 - 4);
  const contrast = Math.max(0, p92 - p10);

  // Relative-to-paper thresholds make this work on yellow/gray old pages.
  const strongThreshold = Math.max(
    65,
    Math.min(178, background - 43),
  );
  const mediumThreshold = Math.max(
    85,
    Math.min(195, background - 27),
  );

  let strong = 0;
  let medium = 0;
  let weighted = 0;

  // Second pass through sampled area for foreground ratios.
  for (
    let y = Math.max(0, Math.floor(b.minY));
    y <= Math.min(image.height - 1, Math.ceil(b.maxY));
    y += step
  ) {
    for (
      let x = Math.max(0, Math.floor(b.minX));
      x <= Math.min(image.width - 1, Math.ceil(b.maxX));
      x += step
    ) {
      if (!pointInPolygon(x, y, pts)) continue;

      const g = grayAt(image, x, y);
      const relativeY =
        (y - b.minY) /
        Math.max(1, b.height);
      const band = Math.max(
        0,
        Math.min(
          bandCount - 1,
          Math.floor(relativeY * bandCount),
        ),
      );

      if (g <= strongThreshold) {
        strong += 1;
        bandStrong[band] += 1;
      }

      if (g <= mediumThreshold) {
        medium += 1;
      }

      weighted += Math.max(0, background - g);
    }
  }

  const samples = grayscale.length;
  const strongRatio = strong / samples;
  const mediumRatio = medium / samples;
  const weightedDarkness =
    weighted /
    Math.max(1, samples * 255);

  let activeBands = 0;

  for (let i = 0; i < bandCount; i++) {
    if (!bandSamples[i]) continue;

    const ratio =
      bandStrong[i] /
      bandSamples[i];

    if (ratio >= 0.0045) {
      activeBands += 1;
    }
  }

  const activeBandRatio =
    activeBands / bandCount;

  return {
    valid: true,
    p10,
    p20,
    p50,
    p85,
    p92,
    background,
    contrast,
    strongRatio,
    mediumRatio,
    weightedDarkness,
    activeBandRatio,
    samples,
  };
}

function directBoxHasEnoughInk(image, box) {
  const mode = directInkGuardMode();

  if (mode === "off") {
    return {
      keep: true,
      evidence: null,
      reason: "guard-off",
    };
  }

  const evidence =
    measureDirectInkEvidence(
      image,
      box,
    );

  if (!evidence.valid) {
    return {
      keep: false,
      evidence,
      reason: "too-few-pixels",
    };
  }

  const source =
    box._directSource ||
    "primary";

  const isRecovery =
    source === "recovery";

  const b = directBoxInfo(box);
  const estimatedChars =
    estimateVerticalChars(box);

  // Important short labels such as "臺灣史" are allowed to be small,
  // but real black print should still have clear local contrast.
  const shortImportant =
    estimatedChars <= 7 &&
    b.height >= b.width * 1.2;

  let minContrast;
  let minStrong;
  let minMedium;
  let minDarkness;
  let minActiveBands;

  if (mode === "strict") {
    minContrast = isRecovery ? 34 : 28;
    minStrong = isRecovery ? 0.010 : 0.0065;
    minMedium = isRecovery ? 0.026 : 0.018;
    minDarkness = isRecovery ? 0.017 : 0.012;
    minActiveBands = isRecovery ? 0.25 : 0.16;
  } else {
    // Balanced: primary boxes are treated gently; sensitive second-pass
    // additions must show noticeably stronger evidence.
    minContrast = isRecovery ? 29 : 21;
    minStrong = isRecovery ? 0.0075 : 0.0035;
    minMedium = isRecovery ? 0.020 : 0.011;
    minDarkness = isRecovery ? 0.013 : 0.0075;
    minActiveBands = isRecovery ? 0.20 : 0.08;
  }

  if (shortImportant && !isRecovery) {
    minActiveBands = 0;
    minStrong *= 0.75;
    minMedium *= 0.75;
  }

  const inkVotes = [
    evidence.strongRatio >= minStrong,
    evidence.mediumRatio >= minMedium,
    evidence.weightedDarkness >= minDarkness,
    evidence.activeBandRatio >= minActiveBands,
  ].filter(Boolean).length;

  // Require clear contrast plus at least two independent foreground signals.
  const keep =
    evidence.contrast >= minContrast &&
    inkVotes >= 2;

  let reason = "ok";

  if (!keep) {
    reason =
      `contrast=${evidence.contrast.toFixed(1)},` +
      ` strong=${(evidence.strongRatio * 100).toFixed(2)}%,` +
      ` medium=${(evidence.mediumRatio * 100).toFixed(2)}%,` +
      ` bands=${(evidence.activeBandRatio * 100).toFixed(0)}%`;
  }

  return {
    keep,
    evidence,
    reason,
  };
}

function filterDirectBoxesByInk(image, boxes) {
  const kept = [];
  const rejected = [];

  for (const box of boxes) {
    const verdict =
      directBoxHasEnoughInk(
        image,
        box,
      );

    if (verdict.keep) {
      kept.push({
        ...box,
        _directInkEvidence:
          verdict.evidence,
      });
    } else {
      rejected.push({
        box,
        verdict,
      });
    }
  }

  return {
    boxes: kept,
    rejected,
  };
}

function plausibleRecoveryBox(box, image) {
  const b = directBoxInfo(box);

  if (b.width < 4 || b.height < 12) return false;
  if (b.width * b.height < 70) return false;

  // Traditional book path: only let the sensitive pass add vertical-ish regions.
  if (b.height < b.width * 1.10) return false;

  if (
    b.centerX < -2 ||
    b.centerX > image.width + 2 ||
    b.centerY < -2 ||
    b.centerY > image.height + 2
  ) {
    return false;
  }

  return true;
}

function mergeRecoveryBoxesPrimarySafe(primaryBoxes, recoveryBoxes, image) {
  const merged = primaryBoxes.map((box, index) => ({
    ...box,
    _directSource: "primary",
    _directPrimaryIndex: index,
  }));

  let added = 0;

  for (const raw of recoveryBoxes) {
    if (!plausibleRecoveryBox(raw, image)) continue;

    const rb = directBoxInfo(raw);
    let duplicate = false;

    for (const existing of merged) {
      const eb = directBoxInfo(existing);
      const iou = directRectIoU(rb, eb);
      const c1 = directContainment(rb, eb);
      const c2 = directContainment(eb, rb);

      if (
        iou >= 0.20 ||
        c1 >= 0.62 ||
        c2 >= 0.62
      ) {
        duplicate = true;
        break;
      }

      // Same vertical line but chopped differently: avoid adding a near-duplicate
      // if x is almost identical and y ranges overlap substantially.
      const xClose =
        Math.abs(rb.centerX - eb.centerX) <=
        Math.max(8, Math.min(rb.width, eb.width) * 0.55);

      const yOverlap =
        Math.max(
          0,
          Math.min(rb.maxY, eb.maxY) -
          Math.max(rb.minY, eb.minY),
        );

      const yOverlapRatio =
        yOverlap /
        Math.max(1, Math.min(rb.height, eb.height));

      if (xClose && yOverlapRatio >= 0.45) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      merged.push({
        ...raw,
        _directSource: "recovery",
      });
      added += 1;
    }
  }

  return {
    boxes: merged,
    added,
  };
}

function orderQuadVertical(box) {
  const pts = boxPoints(box).map((p) => ({
    x: Number(p.x),
    y: Number(p.y),
  }));

  if (pts.length < 4) return null;

  // The book columns are only mildly skewed. Sorting into top/bottom pairs
  // is more stable here than relying on detector polygon winding.
  const sortedY = [...pts].sort(
    (a, b) => a.y - b.y || a.x - b.x,
  );

  const top = sortedY.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sortedY.slice(-2).sort((a, b) => a.x - b.x);

  const tl = top[0];
  const tr = top[1];
  const bl = bottom[0];
  const br = bottom[1];

  return [tl, tr, br, bl];
}

function lerpPoint(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function makeSubQuad(box, t0, t1, meta = {}) {
  const q = orderQuadVertical(box);
  if (!q) return { ...box, ...meta };

  const [tl, tr, br, bl] = q;

  const topL = lerpPoint(tl, bl, t0);
  const topR = lerpPoint(tr, br, t0);
  const botL = lerpPoint(tl, bl, t1);
  const botR = lerpPoint(tr, br, t1);

  const points = [topL, topR, botR, botL];

  return {
    ...box,
    points,
    polygon: points,
    ...meta,
  };
}

function estimateVerticalChars(box) {
  const b = directBoxInfo(box);
  const width = Math.max(5, b.width);

  // Chinese vertical print is approximately square. 0.92 compensates for
  // punctuation/line spacing so we do not over-split normal boxes.
  return Math.max(
    1,
    (b.height / width) * 0.92,
  );
}

function splitDirectBoxForRecognition(box, parentIndex) {
  const estimatedChars = estimateVerticalChars(box);

  // Short and medium lines stay whole. This preserves the successful
  // "臺灣史" behavior.
  if (estimatedChars <= 13) {
    return [{
      ...box,
      _directParentIndex: parentIndex,
      _directChunkIndex: 0,
      _directChunkCount: 1,
      _directExpectedChars: estimatedChars,
      _directSource: box._directSource || "primary",
    }];
  }

  const targetChars = 11.0;
  const overlapChars = 1.4;
  const stepChars = targetChars - overlapChars;

  const chunks = [];
  let start = 0;

  while (start < estimatedChars - 0.5) {
    const end = Math.min(
      estimatedChars,
      start + targetChars,
    );

    const t0 = Math.max(0, start / estimatedChars);
    const t1 = Math.min(1, end / estimatedChars);

    chunks.push(
      makeSubQuad(
        box,
        t0,
        t1,
        {
          _directParentIndex: parentIndex,
          _directChunkIndex: chunks.length,
          _directExpectedChars: end - start,
          _directSource: box._directSource || "primary",
        },
      ),
    );

    if (end >= estimatedChars - 0.01) break;
    start += stepChars;
  }

  chunks.forEach((chunk) => {
    chunk._directChunkCount = chunks.length;
  });

  return chunks;
}

function coreDirectText(text) {
  return String(text || "")
    .replace(/\s+/g, "")
    .trim();
}

function mergeChunkTexts(texts) {
  const clean = texts
    .map(coreDirectText)
    .filter(Boolean);

  if (!clean.length) return "";
  if (clean.length === 1) return clean[0];

  let out = clean[0];

  for (let i = 1; i < clean.length; i++) {
    const next = clean[i];
    let bestOverlap = 0;
    const maxOverlap = Math.min(6, out.length, next.length);

    for (let k = maxOverlap; k >= 1; k--) {
      if (out.slice(-k) === next.slice(0, k)) {
        bestOverlap = k;
        break;
      }
    }

    out += next.slice(bestOverlap);
  }

  return out;
}

function expandQuad(box, image, widthScale = 1.14, heightScale = 1.08) {
  const pts = boxPoints(box);
  if (pts.length < 4) return { ...box };

  const cx = pts.reduce((s, p) => s + Number(p.x), 0) / pts.length;
  const cy = pts.reduce((s, p) => s + Number(p.y), 0) / pts.length;

  const points = pts.map((p) => ({
    x: Math.max(
      0,
      Math.min(
        image.width - 1,
        cx + (Number(p.x) - cx) * widthScale,
      ),
    ),
    y: Math.max(
      0,
      Math.min(
        image.height - 1,
        cy + (Number(p.y) - cy) * heightScale,
      ),
    ),
  }));

  return {
    ...box,
    points,
    polygon: points,
    _directExpandedChunk: true,
  };
}

function chunkNeedsRepair(result, chunk) {
  const conf = Number(result?.confidence || 0);
  const text = coreDirectText(result?.text);
  const expected = Number(chunk?._directExpectedChars || 0);

  if (!text) return true;
  if (conf < 0.66) return true;

  if (
    expected >= 3 &&
    text.length < expected * 0.58
  ) {
    return true;
  }

  // Tiny/short labels are important to user trust, so low-ish confidence
  // gets one expanded attempt.
  if (
    expected <= 6 &&
    conf < 0.76
  ) {
    return true;
  }

  return false;
}

function chooseChunkRecognition(base, alt, chunk) {
  const baseText = coreDirectText(base?.text);
  const altText = coreDirectText(alt?.text);
  const baseConf = Number(base?.confidence || 0);
  const altConf = Number(alt?.confidence || 0);
  const expected = Number(chunk?._directExpectedChars || 0);

  if (!altText) return base;
  if (!baseText) return {
    ...alt,
    _directChunkRefined: true,
  };

  const baseDiff = expected
    ? Math.abs(baseText.length - expected)
    : 0;
  const altDiff = expected
    ? Math.abs(altText.length - expected)
    : 0;

  if (
    altDiff + 0.6 < baseDiff &&
    altConf >= baseConf - 0.10
  ) {
    return {
      ...alt,
      _directChunkRefined: true,
    };
  }

  if (
    altText.length > baseText.length &&
    altConf >= baseConf - 0.06
  ) {
    return {
      ...alt,
      _directChunkRefined: true,
    };
  }

  if (altConf >= baseConf + 0.055) {
    return {
      ...alt,
      _directChunkRefined: true,
    };
  }

  return base;
}

async function recognizeDirectChunked(
  image,
  normalized,
  parentBoxes,
  recognizer,
) {
  const chunks = [];

  parentBoxes.forEach((box, parentIndex) => {
    chunks.push(
      ...splitDirectBoxForRecognition(
        box,
        parentIndex,
      ),
    );
  });

  setStatus(
    "Hybrid Recognition：PP-OCRv6 分段辨識…",
    `${parentBoxes.length} 個文字框 → ${chunks.length} 個 recognition 小段`,
    null,
  );
  await nextPaint();

  let chunkResults = await recognizer.run(
    normalized,
    chunks,
    {
      ordering: {
        sortByReadingOrder: false,
      },
    },
  );

  // Preserve our metadata even if the library returns fresh result objects.
  chunkResults = chunkResults.map((result, i) => ({
    ...result,
    _directParentIndex: chunks[i]?._directParentIndex,
    _directChunkIndex: chunks[i]?._directChunkIndex,
    _directChunkCount: chunks[i]?._directChunkCount,
    _directExpectedChars: chunks[i]?._directExpectedChars,
    _directSource: chunks[i]?._directSource,
  }));

  let refinedCount = 0;

  if (directPassMode() === "two") {
    const repairIndexes = chunkResults
      .map((result, i) => ({
        i,
        need: chunkNeedsRepair(result, chunks[i]),
        confidence: Number(result?.confidence || 0),
      }))
      .filter((x) => x.need)
      .sort((a, b) => a.confidence - b.confidence)
      .slice(0, 12)
      .map((x) => x.i);

    if (repairIndexes.length) {
      const expandedChunks = repairIndexes.map((i) =>
        expandQuad(
          chunks[i],
          image,
          1.14,
          1.09,
        ),
      );

      setStatus(
        "Hybrid Recognition：可疑小段擴框重認…",
        `只重認 ${expandedChunks.length} 個小段`,
        null,
      );
      await nextPaint();

      const alternatives = await recognizer.run(
        normalized,
        expandedChunks,
        {
          ordering: {
            sortByReadingOrder: false,
          },
        },
      );

      repairIndexes.forEach((chunkIndex, j) => {
        const alt = alternatives[j];
        if (!alt) return;

        const chosen = chooseChunkRecognition(
          chunkResults[chunkIndex],
          alt,
          chunks[chunkIndex],
        );

        if (chosen !== chunkResults[chunkIndex]) {
          chunkResults[chunkIndex] = {
            ...chosen,
            _directParentIndex: chunks[chunkIndex]._directParentIndex,
            _directChunkIndex: chunks[chunkIndex]._directChunkIndex,
            _directChunkCount: chunks[chunkIndex]._directChunkCount,
            _directExpectedChars: chunks[chunkIndex]._directExpectedChars,
            _directSource: chunks[chunkIndex]._directSource,
            _directChunkRefined: true,
          };
          refinedCount += 1;
        }
      });
    }
  }

  const byParent = new Map();

  chunkResults.forEach((result) => {
    const parentIndex = Number(result._directParentIndex);

    if (!byParent.has(parentIndex)) {
      byParent.set(parentIndex, []);
    }
    byParent.get(parentIndex).push(result);
  });

  const parentResults = parentBoxes.map((box, parentIndex) => {
    const parts = (byParent.get(parentIndex) || []).sort(
      (a, b) =>
        Number(a._directChunkIndex || 0) -
        Number(b._directChunkIndex || 0),
    );

    const text = mergeChunkTexts(
      parts.map((part) => part.text),
    );

    const confidences = parts
      .map((part) => Number(part.confidence || 0))
      .filter(Number.isFinite);

    const confidence = confidences.length
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length
      : 0;

    return {
      box,
      text,
      confidence,
      _directSource: box._directSource || "primary",
      _directChunkCount: parts.length || 1,
      _directChunkTexts: parts.map((part) => coreDirectText(part.text)),
      _directChunkRefined: parts.some((part) => part._directChunkRefined),
    };
  });

  return {
    results: parentResults,
    chunkCount: chunks.length,
    refinedCount,
  };
}

function directResultBox(result) {
  return result?.box || result?.boundingBox || result;
}

function directBoxInfo(box) {
  const points = boxPoints(box);

  if (!points.length) {
    return {
      centerX: 0,
      centerY: 0,
      minX: 0,
      maxX: 0,
      minY: 0,
      maxY: 0,
      width: 0,
      height: 0,
    };
  }

  const xs = points.map((p) => Number(p.x));
  const ys = points.map((p) => Number(p.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    minX,
    maxX,
    minY,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
  };
}

function horizontalOverlapRatio(a, b) {
  const overlap = Math.max(
    0,
    Math.min(a.maxX, b.maxX) -
      Math.max(a.minX, b.minX),
  );

  return overlap /
    Math.max(
      1,
      Math.min(a.width, b.width),
    );
}

function clusterTraditionalDirectResults(results) {
  const items = (results || [])
    .filter((result) =>
      String(result?.text || "").trim(),
    )
    .map((result, rawIndex) => ({
      result,
      rawIndex,
      info: directBoxInfo(
        directResultBox(result),
      ),
    }));

  if (!items.length) return [];

  const verticalWidths = items
    .filter(
      (item) =>
        item.info.height >
        item.info.width * 1.08,
    )
    .map((item) => item.info.width)
    .filter((v) => v > 2 && v < 250);

  const typicalWidth =
    verticalWidths.length
      ? median(verticalWidths)
      : median(
          items
            .map((item) =>
              Math.min(
                item.info.width,
                item.info.height,
              ),
            )
            .filter((v) => v > 2),
        );

  const baseThreshold = Math.max(
    6,
    Math.min(
      16,
      (typicalWidth || 20) * 0.48,
    ),
  );

  function verticalOverlapRatio(a, b) {
    const overlap = Math.max(
      0,
      Math.min(a.maxY, b.maxY) -
        Math.max(a.minY, b.minY),
    );

    return overlap /
      Math.max(
        1,
        Math.min(a.height, b.height),
      );
  }

  function verticalGap(a, b) {
    if (a.maxY < b.minY) {
      return b.minY - a.maxY;
    }

    if (b.maxY < a.minY) {
      return a.minY - b.maxY;
    }

    return 0;
  }

  // Build columns from right to left, but ONLY stack boxes that behave like
  // upper/lower fragments. Two boxes that coexist over the same y range are
  // adjacent columns even if x centers are close.
  const sorted = [...items].sort(
    (a, b) =>
      b.info.centerX - a.info.centerX ||
      a.info.minY - b.info.minY,
  );

  const clusters = [];

  for (const item of sorted) {
    let best = null;

    for (const cluster of clusters) {
      const centerDistance =
        Math.abs(
          item.info.centerX -
          cluster.centerX,
        );

      const overlapX =
        horizontalOverlapRatio(
          item.info,
          {
            minX: cluster.minX,
            maxX: cluster.maxX,
            width: Math.max(
              1,
              cluster.maxX - cluster.minX,
            ),
          },
        );

      // Compare against the most vertically-near existing fragment.
      let nearestItem = null;
      let nearestGap = Infinity;
      let maxYOverlap = 0;

      for (const existing of cluster.items) {
        const gap =
          verticalGap(
            item.info,
            existing.info,
          );

        const yOverlap =
          verticalOverlapRatio(
            item.info,
            existing.info,
          );

        maxYOverlap = Math.max(
          maxYOverlap,
          yOverlap,
        );

        if (gap < nearestGap) {
          nearestGap = gap;
          nearestItem = existing;
        }
      }

      // Strong rule learned from the 9 / 9.2 failure:
      // if two tall boxes overlap substantially in Y, they are side-by-side
      // columns, NOT fragments of one vertical line.
      if (maxYOverlap >= 0.28) {
        continue;
      }

      const xCompatible =
        centerDistance <=
          baseThreshold * 1.25 ||
        overlapX >= 0.36;

      if (!xCompatible) continue;

      // Same-column split fragments should be reasonably near each other
      // vertically. Allow a generous gap for punctuation/blank line, but not
      // half a page.
      const pageSpan =
        Math.max(
          item.info.maxY,
          ...cluster.items.map(
            (x) => x.info.maxY,
          ),
        ) -
        Math.min(
          item.info.minY,
          ...cluster.items.map(
            (x) => x.info.minY,
          ),
        );

      const maxAllowedGap =
        Math.max(
          (typicalWidth || 20) * 4.5,
          pageSpan * 0.22,
        );

      if (
        Number.isFinite(nearestGap) &&
        nearestGap > maxAllowedGap
      ) {
        continue;
      }

      const score =
        centerDistance +
        Math.max(0, nearestGap) * 0.03 -
        overlapX * baseThreshold;

      if (!best || score < best.score) {
        best = {
          cluster,
          score,
        };
      }
    }

    if (!best) {
      clusters.push({
        items: [item],
        minX: item.info.minX,
        maxX: item.info.maxX,
        centerX: item.info.centerX,
      });
      continue;
    }

    const cluster = best.cluster;
    cluster.items.push(item);
    cluster.minX = Math.min(
      cluster.minX,
      item.info.minX,
    );
    cluster.maxX = Math.max(
      cluster.maxX,
      item.info.maxX,
    );

    // Median center is more robust to one curved / wide fragment.
    cluster.centerX = median(
      cluster.items.map(
        (x) => x.info.centerX,
      ),
    );
  }

  // Traditional page order: rightmost column first.
  clusters.sort(
    (a, b) =>
      b.centerX - a.centerX,
  );

  return clusters.map(
    (cluster, columnIndex) => {
      // Within a TRUE split column: strictly top edge -> bottom edge.
      // This same ordering drives labels AND output text.
      cluster.items.sort(
        (a, b) =>
          a.info.minY - b.info.minY ||
          a.info.centerY - b.info.centerY,
      );

      const texts =
        cluster.items
          .map((item) =>
            String(
              item.result.text || "",
            ).trim(),
          )
          .filter(Boolean);

      const confidences =
        cluster.items
          .map((item) =>
            Number(
              item.result.confidence || 0,
            ),
          )
          .filter(Number.isFinite);

      const confidence =
        confidences.length
          ? confidences.reduce(
              (a, b) => a + b,
              0,
            ) / confidences.length
          : 0;

      return {
        columnIndex,
        text: texts.join(""),
        confidence,
        items: cluster.items,
        centerX: cluster.centerX,
      };
    },
  );
}

function drawDirectV6Overlay(
  image,
  rawResults,
  columns,
  canvas,
) {
  pixelsToCanvas(image, canvas);
  const ctx = canvas.getContext("2d");
  const resultToColumn = new Map();

  columns.forEach((column) => {
    column.items.forEach(
      (item, fragmentIndex) => {
        resultToColumn.set(
          item.result,
          {
            columnIndex:
              column.columnIndex,
            fragmentIndex,
          },
        );
      },
    );
  });

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = Math.max(
    1.6,
    canvas.width / 650,
  );
  ctx.font =
    `bold ${Math.max(
      13,
      canvas.width / 58,
    )}px system-ui`;

  for (const result of rawResults) {
    const box = directResultBox(result);
    const points = boxPoints(box);
    if (!points.length) continue;

    const label =
      resultToColumn.get(result);

    ctx.beginPath();
    ctx.moveTo(
      points[0].x,
      points[0].y,
    );

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(
        points[i].x,
        points[i].y,
      );
    }

    ctx.closePath();
    const source =
      result._directSource ||
      box?._directSource ||
      "primary";

    const refined =
      !!result._directChunkRefined;

    ctx.fillStyle = refined
      ? "rgba(39, 142, 81, .10)"
      : source === "recovery"
        ? "rgba(214, 132, 33, .10)"
        : "rgba(35, 132, 115, .08)";
    ctx.fill();

    ctx.strokeStyle = refined
      ? "rgba(31, 135, 72, .95)"
      : source === "recovery"
        ? "rgba(199, 116, 20, .95)"
        : "rgba(30, 120, 105, .90)";
    ctx.stroke();

    if (label) {
      const info =
        directBoxInfo(box);

      const text =
        label.fragmentIndex > 0
          ? `${label.columnIndex + 1}.${label.fragmentIndex + 1}`
          : `${label.columnIndex + 1}`;

      const tx =
        Math.max(
          3,
          Math.min(
            canvas.width - 60,
            info.minX + 3,
          ),
        );
      const ty =
        Math.max(
          18,
          info.minY + 17,
        );

      ctx.fillStyle =
        "rgba(18, 104, 91, .98)";
      ctx.fillText(
        text,
        tx,
        ty,
      );
    }
  }
}

function renderDirectRecognition(
  columns,
  side,
) {
  const root =
    $(
      side === "right"
        ? "rightDirectRecognition"
        : "leftDirectRecognition",
    );

  root.innerHTML = "";

  if (!columns.length) {
    root.innerHTML =
      '<div class="empty">沒有辨識到文字</div>';
    return;
  }

  columns.forEach((column) => {
    const row =
      document.createElement("div");
    row.className =
      "recognition-row direct-rec-row";

    const name =
      document.createElement("div");
    name.className =
      "recognition-name";
    name.textContent =
      `Direct Column ${String(
        column.columnIndex + 1,
      ).padStart(2, "0")}`;

    const text =
      document.createElement("div");
    text.className =
      "recognition-text";
    text.textContent =
      column.text;

    const confidence =
      document.createElement("div");
    confidence.className =
      "recognition-confidence";
    confidence.textContent =
      `${(
        column.confidence * 100
      ).toFixed(1)}%`;

    const meta =
      document.createElement("div");
    meta.className =
      "direct-box-meta";
    const chunkTotal = column.items.reduce(
      (sum, item) =>
        sum +
        Math.max(
          1,
          Number(item.result?._directChunkCount || 1),
        ),
      0,
    );

    meta.textContent =
      column.items.length > 1
        ? `同欄共 ${column.items.length} 個 Detector box；合計 ${chunkTotal} 個 v6 recognition 小段。`
        : chunkTotal > 1
          ? `單一 Detector box 太長，已切成 ${chunkTotal} 個 v6 recognition 小段後合併。`
          : "短欄：Detector 四邊形直接交給 v6 recognition。";

    const hasRecovery = column.items.some(
      (item) =>
        (
          item.result?._directSource ||
          directResultBox(item.result)?._directSource
        ) === "recovery",
    );

    const hasChunking = column.items.some(
      (item) =>
        Number(item.result?._directChunkCount || 1) > 1,
    );

    const hasRefine = column.items.some(
      (item) => !!item.result?._directChunkRefined,
    );

    if (hasRecovery) {
      const badge = document.createElement("span");
      badge.className = "direct-recovery-badge";
      badge.textContent = "第二遍補回";
      meta.appendChild(badge);
    }

    if (hasChunking) {
      const badge = document.createElement("span");
      badge.className = "direct-chunk-badge";
      badge.textContent = "長欄分段";
      meta.appendChild(badge);
    }

    if (hasRefine) {
      const badge = document.createElement("span");
      badge.className = "direct-refine-badge";
      badge.textContent = "擴框重認";
      meta.appendChild(badge);
    }

    text.appendChild(meta);

    row.appendChild(name);
    row.appendChild(text);
    row.appendChild(confidence);
    root.appendChild(row);
  });
}

function assembleDirectFullText() {
  const right =
    state.directRightColumns
      .map((column) =>
        column.text.trim(),
      )
      .filter(Boolean);

  const left =
    state.directLeftColumns
      .map((column) =>
        column.text.trim(),
      )
      .filter(Boolean);

  const pages = [];
  if (right.length) {
    pages.push(right.join("\n"));
  }
  if (left.length) {
    pages.push(left.join("\n"));
  }

  const text = pages.join("\n\n");

  $("fullText").value = text;
  $("copyTextBtn").disabled = !text;
  $("downloadTextBtn").disabled = !text;
  $("fullTextOrder").textContent =
    "v7.1 Hybrid：右頁→左頁；v5 Detector 找框，v6 長欄分段辨識；同欄結果依 y 串接。";

  return text;
}

async function ensureDirectV6Services(
  forceWasm = false,
) {
  // Hybrid v7.1:
  // Detection uses the already tuned PP-OCRv5 mobile detector.
  const detector = await ensureDetector();

  if (
    state.directV6Recognizer &&
    state.directV6ForcedWasm === forceWasm
  ) {
    return {
      detector,
      recognizer: state.directV6Recognizer,
    };
  }

  setStatus(
    "準備 Hybrid Recognition…",
    forceWasm
      ? "PP-OCRv6 small recognizer · CPU/WASM"
      : "PP-OCRv5 detector + PP-OCRv6 small recognizer",
    8,
  );
  await nextPaint();

  const [recBuffer, dictText] = await Promise.all([
    state.directV6RecBuffer
      ? Promise.resolve(state.directV6RecBuffer)
      : fetchArrayBufferCached(
          DIRECT_V6_REC.url,
          DIRECT_V6_REC.label,
          24,
          52,
        ),
    state.directV6Dictionary
      ? Promise.resolve(null)
      : fetchTextCached(
          V6_DICT_URL,
          "PP-OCRv6 字典",
        ),
  ]);

  if (!state.directV6Dictionary) {
    state.directV6Dictionary =
      parsePpOcrV6Dictionary(dictText);
  }

  try {
    state.directV6RecSession?.release?.();
  } catch {}

  const recSession = forceWasm
    ? await createWasmSession(
        recBuffer,
        DIRECT_V6_REC.label,
      )
    : await createSessionAuto(
        recBuffer,
        DIRECT_V6_REC.label,
      );

  state.directV6RecBuffer = recBuffer;
  state.directV6RecSession = recSession;
  state.directV6ForcedWasm = forceWasm;

  state.directV6Recognizer =
    new RecognitionService(
      ort,
      recSession,
      {
        ...getTextRecognitionPresetOptions(
          DIRECT_V6_REC.preset,
        ),
        charactersDictionary:
          state.directV6Dictionary,
      },
    );

  // Alias only for existing status/fallback code; this is v5 detector.
  state.directV6Detector = detector;

  return {
    detector,
    recognizer: state.directV6Recognizer,
  };
}

async function runDirectV6Page(
  image,
  side,
  detector,
  recognizer,
  baseProgress,
) {
  const normalized =
    normalizeInputToRgb(image);

  const detectorOptions =
    detectorRuntimeOptions();

  setStatus(
    `Hybrid：${
      side === "right"
        ? "右頁"
        : "左頁"
    } v5 Detector 第一次…`,
    "使用已經在老書頁面較穩定的 v5 mobile detection",
    baseProgress,
  );
  await nextPaint();

  const primaryBoxes =
    await detector.run(
      normalized,
      {
        ...detectorOptions,
      },
    );

  let merged = {
    boxes: primaryBoxes.map((box, index) => ({
      ...box,
      _directSource: "primary",
      _directPrimaryIndex: index,
    })),
    added: 0,
  };

  let recoveryMs = 0;

  if (directPassMode() === "two") {
    const recoveryStarted =
      performance.now();

    setStatus(
      `Hybrid：${
        side === "right"
          ? "右頁"
          : "左頁"
      } 第二次漏框檢查…`,
      "同一 v5 Detector · 1.2× · 較敏感；只新增第一遍完全沒有的框",
      baseProgress + 8,
    );
    await nextPaint();

    const scaled =
      resizePixelImage(
        image,
        1.20,
      );

    const recoveryScaled =
      await detector.run(
        normalizeInputToRgb(
          scaled.image,
        ),
        {
          ...detectorOptions,
          textPixelThreshold: Math.min(
            Number(detectorOptions.textPixelThreshold || 0.20),
            0.15,
          ),
          boxScoreThreshold: Math.min(
            Number(detectorOptions.boxScoreThreshold || 0.35),
            0.27,
          ),
          unclipRatio: Math.max(
            Number(detectorOptions.unclipRatio || 1.15),
            1.22,
          ),
          maxSideLength: Math.min(
            3000,
            Math.round(
              Number(detectorOptions.maxSideLength || 2000) * 1.25,
            ),
          ),
          minimumAreaThreshold: 12,
          maxCandidates: 1400,
          boxType: "quad",
        },
      );

    const recoveryBoxes =
      recoveryScaled.map(
        (box) =>
          scaleDetectorBox(
            box,
            scaled.scale,
          ),
      );

    merged =
      mergeRecoveryBoxesPrimarySafe(
        primaryBoxes,
        recoveryBoxes,
        image,
      );

    recoveryMs =
      performance.now() -
      recoveryStarted;
  }

  const inkFiltered =
    filterDirectBoxesByInk(
      image,
      merged.boxes,
    );

  const boxes =
    inkFiltered.boxes;

  if (inkFiltered.rejected.length > 0) {
    console.info(
      `Ink Guard rejected ${inkFiltered.rejected.length} ${
        side === "right" ? "right" : "left"
      } page boxes`,
      inkFiltered.rejected.map(
        (item) => ({
          source:
            item.box._directSource ||
            "primary",
          reason:
            item.verdict.reason,
          box:
            directBoxInfo(
              item.box,
            ),
        }),
      ),
    );
  }

  const recognition =
    await recognizeDirectChunked(
      image,
      normalized,
      boxes,
      recognizer,
    );

  const results =
    recognition.results;

  const columns =
    clusterTraditionalDirectResults(
      results,
    );

  const canvas =
    $(
      side === "right"
        ? "rightDirectOverlay"
        : "leftDirectOverlay",
    );

  drawDirectV6Overlay(
    image,
    results,
    columns,
    canvas,
  );

  showCanvas(
    side === "right"
      ? "rightDirectOverlay"
      : "leftDirectOverlay",
    side === "right"
      ? "rightDirectOverlayEmpty"
      : "leftDirectOverlayEmpty",
  );

  renderDirectRecognition(
    columns,
    side,
  );

  const statParts = [
    `v5框 ${primaryBoxes.length}`,
  ];

  if (merged.added > 0) {
    statParts.push(
      `第二遍補 ${merged.added}`,
    );
  }

  if (inkFiltered.rejected.length > 0) {
    statParts.push(
      `空白擋掉 ${inkFiltered.rejected.length}`,
    );
  }

  statParts.push(
    `${recognition.chunkCount} rec段`,
  );

  if (recognition.refinedCount > 0) {
    statParts.push(
      `擴框採用 ${recognition.refinedCount}`,
    );
  }

  if (recoveryMs > 0) {
    statParts.push(
      `補漏 +${(recoveryMs / 1000).toFixed(2)}s`,
    );
  }

  $(
    side === "right"
      ? "rightDirectStats"
      : "leftDirectStats",
  ).textContent =
    statParts.join(" · ");

  $(
    side === "right"
      ? "rightDirectRecStats"
      : "leftDirectRecStats",
  ).textContent =
    `${columns.filter(
      (column) =>
        column.text.trim(),
    ).length}/${columns.length} 欄有文字`;

  return {
    results,
    columns,
    boxes,
    primaryCount: primaryBoxes.length,
    recoveryAdded: merged.added,
    inkRejected: inkFiltered.rejected.length,
    chunkCount: recognition.chunkCount,
    refinedCount: recognition.refinedCount,
    recoveryMs,
  };
}

async function resetDirectV6Runtime() {
  try {
    state.directV6RecSession?.release?.();
  } catch {}

  state.directV6Recognizer = null;
  state.directV6RecSession = null;
  state.directV6ForcedWasm = false;

  // v5 detector is shared with the legacy pipeline; keep it alive.
  state.directV6Detector = state.detector;
}

async function runDirectV6Ocr() {
  if (!state.rightFlat || !state.leftFlat) {
    splitPages();
    await runUvDoc();
  }

  $("directV6Pipeline").classList.remove(
    "hidden",
  );

  const started =
    performance.now();

  let services =
    await ensureDirectV6Services(false);

  let right;
  let left;

  try {
    right =
      await runDirectV6Page(
        state.rightFlat,
        "right",
        services.detector,
        services.recognizer,
        16,
      );

    left =
      await runDirectV6Page(
        state.leftFlat,
        "left",
        services.detector,
        services.recognizer,
        58,
      );
  } catch (gpuError) {
    const recBackend =
      state.backends[
        DIRECT_V6_REC.label
      ];

    if (recBackend !== "webgpu") {
      throw gpuError;
    }

    console.warn(
      "Direct PP-OCRv6 WebGPU inference failed; retrying both detector and recognizer on WASM.",
      gpuError,
    );

    setStatus(
      "Direct PP-OCRv6 GPU 推理失敗，切換 CPU / WASM…",
      gpuError instanceof Error
        ? gpuError.message
        : String(gpuError),
      5,
    );

    await resetDirectV6Runtime();
    services =
      await ensureDirectV6Services(true);

    right =
      await runDirectV6Page(
        state.rightFlat,
        "right",
        services.detector,
        services.recognizer,
        16,
      );

    left =
      await runDirectV6Page(
        state.leftFlat,
        "left",
        services.detector,
        services.recognizer,
        58,
      );
  }

  state.directRightResults =
    right.results;
  state.directLeftResults =
    left.results;
  state.directRightColumns =
    right.columns;
  state.directLeftColumns =
    left.columns;

  const text =
    assembleDirectFullText();

  const elapsed =
    performance.now() - started;

  $("directTiming").textContent =
    `${(elapsed / 1000).toFixed(2)} 秒`;

  $("directSummary").textContent =
    `右頁：v5 第一遍 ${right.primaryCount} 框，補漏 ${right.recoveryAdded}，Ink Guard 擋掉 ${right.inkRejected}，${right.chunkCount} 個 v6 recognition 小段；` +
    `左頁：v5 第一遍 ${left.primaryCount} 框，補漏 ${left.recoveryAdded}，Ink Guard 擋掉 ${left.inkRejected}，${left.chunkCount} 個 v6 recognition 小段。` +
    `相鄰欄若在 y 方向大量重疊，現在不再被合併成 9 / 9.2；只有真正上下接續的 box 才會串成同欄。`;

  setStatus(
    "PP-OCRv6 Direct OCR 完成。",
    text
      ? "全文已用 Direct Box 閱讀順序組合。下方舊 V3 仍可另外跑來 A/B。"
      : "模型完成，但沒有取得文字。",
    100,
  );
}

async function withBusy(fn) {
  const buttons = ["splitBtn", "uvBtn", "detBtn", "extractBtn", "recBtn", "directV6Btn", "allBtn"];
  buttons.forEach((id) => $(id).disabled = true);

  try {
    await fn();
  } catch (error) {
    console.error("BookOCR processing error:", error);

    const name = error instanceof Error ? error.name : "Error";
    const message = error instanceof Error ? error.message : String(error);
    const stackLine =
      error instanceof Error && error.stack
        ? error.stack.split("\n").slice(1, 3).join(" | ")
        : "";

    setStatus(
      "處理失敗。",
      `${name}: ${message}${stackLine ? ` ｜ ${stackLine}` : ""}`,
      0,
    );
  } finally {
    if (state.bitmap) {
      buttons.forEach((id) => $(id).disabled = false);
    }
  }
}
async function loadFile(file) {
  state.file = file;
  state.bitmap?.close?.();
  state.bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  state.fullInput = bitmapToPixels(state.bitmap, 2600);
  state.routedMode = null;
  state.routerAnalysis = null;
  state.generalPages = [];
  state.generalRecognition = [];
  state.directRightResults = [];
  state.directLeftResults = [];
  state.directRightColumns = [];
  state.directLeftColumns = [];

  $("directV6Pipeline").classList.add("hidden");
  $("rightDirectRecognition").innerHTML =
    '<div class="empty">等待 PP-OCRv6 Direct OCR</div>';
  $("leftDirectRecognition").innerHTML =
    '<div class="empty">等待 PP-OCRv6 Direct OCR</div>';
  $("rightDirectStats").textContent = "尚未執行";
  $("leftDirectStats").textContent = "尚未執行";
  $("rightDirectRecStats").textContent = "尚未辨識";
  $("leftDirectRecStats").textContent = "尚未辨識";
  $("directTiming").textContent = "尚未執行";

  $("routerCard").classList.add("hidden");
  $("generalPages").innerHTML = "";
  $("generalRecognition").innerHTML = '<div class="empty">等待 Auto OCR</div>';
  $("generalStats").textContent = "尚未辨識";

  $("photoPreview").src = URL.createObjectURL(file);
  $("photoPreview").style.display = "block";
  $("dropHint").style.display = "none";

  ["splitBtn", "uvBtn", "detBtn", "extractBtn", "recBtn", "directV6Btn", "allBtn"].forEach((id) => $(id).disabled = false);

  const selected = $("ocrMode").value;
  if (selected === "traditional") {
    updateModeUi("traditional", null);
    splitPages();
  } else {
    updateModeUi(selected === "auto" ? "general" : selected, null);
    setStatus(
      selected === "auto" ? "圖片已載入，等待 Auto OCR。" : `圖片已載入：${displayModeName(selected)}。`,
      selected === "auto" ? "按「Auto OCR 到文字」即可自動判斷。" : "",
      0,
    );
  }
}

$("fileInput").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) loadFile(file);
});

const dropZone = $("dropZone");
["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file?.type.startsWith("image/")) loadFile(file);
});

$("splitRange").addEventListener("input", () => {
  $("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
  if (state.bitmap && $("ocrMode").value === "traditional") splitPages();
});

$("gutterRange").addEventListener("input", () => {
  $("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;
  if (state.bitmap && $("ocrMode").value === "traditional") splitPages();
});

$("pagePadRange").addEventListener("input", () => {
  $("pagePadValue").textContent =
    `${Number($("pagePadRange").value).toFixed(1)}%`;
  if (state.bitmap && $("ocrMode").value === "traditional") splitPages();
});

$("columnExtendRange").addEventListener("input", () => {
  $("columnExtendValue").textContent =
    `${Number($("columnExtendRange").value).toFixed(1)} 字`;

  if (state.rightDetection || state.leftDetection) {
    setStatus(
      "直欄上下延伸已變更。",
      "重新按「4. V3 抽出直欄」即可，不必重跑 UVDoc / Detector。",
      0,
    );
  }
});

$("rightOverlay").addEventListener(
  "pointerdown",
  (event) => handleOverlayPointerDown(event, "right"),
);
$("rightOverlay").addEventListener(
  "pointermove",
  (event) => handleOverlayPointerMove(event, "right"),
);
$("rightOverlay").addEventListener(
  "pointerup",
  (event) => handleOverlayPointerUp(event, "right"),
);
$("rightOverlay").addEventListener(
  "pointercancel",
  (event) => handleOverlayPointerUp(event, "right"),
);

$("leftOverlay").addEventListener(
  "pointerdown",
  (event) => handleOverlayPointerDown(event, "left"),
);
$("leftOverlay").addEventListener(
  "pointermove",
  (event) => handleOverlayPointerMove(event, "left"),
);
$("leftOverlay").addEventListener(
  "pointerup",
  (event) => handleOverlayPointerUp(event, "left"),
);
$("leftOverlay").addEventListener(
  "pointercancel",
  (event) => handleOverlayPointerUp(event, "left"),
);
$("restoreColumnsBtn").addEventListener("click", restoreAllColumns);

$("rerunSelectedBtn").addEventListener(
  "click",
  rerunSelectedColumnOcr,
);

$("resetSelectedBtn").addEventListener(
  "click",
  resetSelectedColumnOverride,
);

$("deleteSelectedBtn").addEventListener("click", () => {
  if (!state.selectedColumn) return;
  removeColumn(
    state.selectedColumn.side,
    state.selectedColumn.columnId,
  );
});

$("clearSelectedBtn").addEventListener("click", () => {
  const previousSide = state.selectedColumn?.side;
  closeColumnEditor();
  if (previousSide) redrawOverlaySide(previousSide);
});

$("directInkGuard").addEventListener("change", () => {
  const mode = directInkGuardMode();

  $("directInkGuardHint").textContent =
    mode === "off"
      ? "Ink Guard 已關閉：Detector 找到的框全部送 OCR，透字／空白 hallucination 風險最高。"
      : mode === "strict"
        ? "嚴格模式：要求更高對比、深色墨跡與垂直覆蓋；透字很多時可測，但可能漏掉非常淡的真字。"
        : "平衡模式：第一遍 Detector 框保守過濾；第二遍敏感補框使用較嚴格墨跡證據，避免空白區被 recognizer 硬猜。";

  setStatus(
    "Ink Guard 模式已切換。",
    "下一次 Hybrid OCR 套用；不用重跑 UVDoc。",
    0,
  );
});

$("directPassMode").addEventListener("change", () => {
  const mode = directPassMode();

  $("directPassHint").textContent =
    mode === "two"
      ? "平衡模式：v5 Detector 第二遍只新增漏框；長欄分段；最多 12 個可疑小段再擴框重認。"
      : "快速模式：只跑一次 v5 Detector；長欄仍會分段辨識，但不做第二遍補漏與擴框重認。";

  setStatus(
    "Hybrid Direct 模式已切換。",
    "下一次按新主路徑 OCR 套用；不用改 UVDoc。",
    0,
  );
});

$("directV6Btn").addEventListener("click", () => withBusy(runDirectV6Ocr));
$("splitBtn").addEventListener("click", () => splitPages());
$("uvBtn").addEventListener("click", () => withBusy(runUvDoc));
$("detBtn").addEventListener("click", () => withBusy(runDetector));
$("extractBtn").addEventListener("click", () => withBusy(runExtraction));
$("recBtn").addEventListener("click", () => withBusy(runRecognition));
$("allBtn").addEventListener("click", () => withBusy(async () => {
  if (!state.fullInput) {
    throw new Error("請先上傳圖片。");
  }

  const mode = await resolveRunMode();

  if (mode === "traditional") {
    splitPages();
    await runUvDoc();
    await runDirectV6Ocr();
  } else {
    await runGeneralPipeline(mode);
  }
}));

$("ocrMode").addEventListener("change", () => {
  const selected = $("ocrMode").value;
  state.routedMode = null;
  state.routerAnalysis = null;
  $("routerCard").classList.add("hidden");

  if (selected === "auto") {
    $("readingOrderLabel").textContent = "Auto 判斷中";
    $("readingOrderHint").textContent = "上傳後按 Auto OCR，先分析版面再選流程。";
    $("verticalPipeline").classList.add("hidden");
    $("generalPipeline").classList.add("hidden");
    $("verticalSteps").classList.add("hidden");
    $("directV6Steps").classList.add("hidden");
  } else {
    updateModeUi(selected, null);
  }

  if (state.bitmap && selected === "traditional") {
    splitPages();
  }

  setStatus(
    selected === "auto"
      ? "已切換 Auto。"
      : `已切換：${displayModeName(selected)}。`,
    "重新按 Auto OCR 到文字即可重跑。",
    0,
  );
});

$("smallTextRescue").addEventListener("change", () => {
  const mode = rescueMode();
  const text = {
    off: "小字救援已關閉：最快。",
    auto2: "小字救援：自動 2×。只對小字／短欄多跑一次。",
    strong: "小字救援：強力 2×＋4×。精度優先，會更慢。",
    all2: "小字救援：全部欄 2×。建議只拿來做速度／精度比較。",
  }[mode];

  $("smallRescueHint").textContent = text;
  setStatus(
    "小字高解析救援模式已切換。",
    "下一次 Recognition 會套用新設定；不需要重跑 UVDoc / Detector。",
    0,
  );
});

$("cnnTextRescue").addEventListener("change", () => {
  const mode = cnnRescueMode();
  const text = {
    off: "CNN 已關閉：不下載、不推理，速度最快。",
    smart: "智慧 CNN：6 字內短欄一定試 1 次；7–10 字只在低信心／候選不一致時才試。",
    short: "短欄 CNN 全開：所有符合小型短欄條件者都試 1 次，較慢。",
  }[mode];

  $("cnnRescueHint").textContent =
    `${text} 第一次實際需要 CNN 時才載入約 4.9 MB 模型。`;

  setStatus(
    "AI 模糊字 CNN 救援模式已切換。",
    "下一次 Recognition 套用；UVDoc / Detector / V3 都不用重跑。",
    0,
  );
});

$("v6TextRescue").addEventListener("change", () => {
  const mode = v6RescueMode();

  const text = {
    off: "PP-OCRv6 救援已關閉。",
    small6: "v6 small：只比較 1～6 字短欄；模型約 21 MB，先用它測速度與精度。",
    medium6: "v6 medium：只比較 1～6 字短欄；模型約 77 MB，精度優先。",
  }[mode];

  $("v6RescueHint").textContent =
    `${text} 切換後只需重跑 Recognition，不用重跑 UVDoc / Detector / V3。`;

  setStatus(
    "PP-OCRv6 hard-case 模式已切換。",
    "下一次 Recognition 套用。",
    0,
  );
});

$("serverTextRescue").addEventListener("change", () => {
  const mode = serverRescueMode();
  const text = {
    off: "Server 救援已關閉，不會下載 84.5 MB 模型。",
    short6: "1～6 字短欄會用 Server 再辨識一次；這是目前最直接的 hard-case 測試。",
    smart10: "1～6 字一定比較；7～10 字只有低信心／候選不一致／字數怪時才比較。",
  }[mode];

  $("serverRescueHint").textContent =
    `${text} 第一次使用會比較久，之後模型走瀏覽器快取。`;

  setStatus(
    "PP-OCRv5 Server hard-case 模式已切換。",
    "下一次 Recognition 套用；不用重跑 UVDoc / Detector / V3。",
    0,
  );
});

$("missingColumnRescue").addEventListener("change", () => {
  setStatus(
    $("missingColumnRescue").checked
      ? "缺欄候選提示已開啟（安全模式）。"
      : "缺欄候選提示已關閉。",
    "候選只畫綠色虛線，不會改 V3/OCR。重新按一次「找直欄中心線」即可更新候選。",
    0,
  );
});

$("recModel").addEventListener("change", () => {
  state.recognizer = null;
  state.recSession = null;
  state.recPreset = null;
  state.recModelBuffer = null;
  state.recSpec = null;
  state.recForcedWasm = false;

  const key = $("recModel").value;
  setStatus(
    key === "mobile"
      ? "已切換到 mobile recognizer。"
      : "已切換到重型 server recognizer。",
    key === "mobile"
      ? "下一次辨識會使用較輕的 mobile 模型。"
      : "server 模型較大，建立 WebGPU session 可能需要更久。",
    0,
  );
});

$("copyTextBtn").addEventListener("click", async () => {
  const text = $("fullText").value;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  setStatus("文字已複製。", "", 100);
});

$("downloadTextBtn").addEventListener("click", () => {
  const text = $("fullText").value;
  if (!text) return;

  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "bookocr.txt";
  a.click();
  URL.revokeObjectURL(url);
});

$("clearCacheBtn").addEventListener("click", async () => {
  if ("caches" in window) await caches.delete(MODEL_CACHE);
  state.uvdoc = null;
  state.uvSession = null;
  state.detector = null;
  state.detSession = null;
  state.recognizer = null;
  state.recSession = null;
  state.recPreset = null;
  state.recDictionary = null;
  state.recModelBuffer = null;
  state.recSpec = null;
  state.recForcedWasm = false;
  state.backends = {};
  updateRuntimeBadge();
  setStatus(
    "模型快取已清除。",
    "下一次執行會重新下載 UVDoc、Detector，以及你選的 Recognition 模型。",
    0,
  );
});

$("splitValue").textContent = `${Number($("splitRange").value).toFixed(1)}%`;
$("gutterValue").textContent = `${Number($("gutterRange").value).toFixed(1)}%`;
$("pagePadValue").textContent = `${Number($("pagePadRange").value).toFixed(1)}%`;
$("columnExtendValue").textContent =
  `${Number($("columnExtendRange").value).toFixed(1)} 字`;


probeWebGpu().then((available) => {
  updateRuntimeBadge();
  console.info(
    available
      ? "WebGPU adapter detected. GPU will be preferred."
      : "No WebGPU adapter. WASM/CPU fallback will be used.",
  );
});


$("verticalPipeline").classList.add("hidden");
$("generalPipeline").classList.add("hidden");
$("verticalSteps").classList.add("hidden");
    $("directV6Steps").classList.add("hidden");
$("readingOrderLabel").textContent = "Auto 判斷中";
