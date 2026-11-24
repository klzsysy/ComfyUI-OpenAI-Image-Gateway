/**
 * ComfyUI Settings UI extension for ComfyUI-OpenAI-Image-Gateway.
 *
 * Registers under Settings -> OpenAI Image Gateway so users can select one or
 * more local ComfyUI workflows, assign each an OpenAI model name exposed via
 * /v1/models, and manage gateway settings without hand-editing YAML files.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ADMIN_BASE = "/openai-gateway/admin";
const STYLE_ID = "oig-settings-style";

function isChineseLocale() {
  try {
    const comfyLocale = app?.ui?.settings?.getSettingValue?.("Comfy.Locale");
    if (typeof comfyLocale === "string" && comfyLocale) {
      return comfyLocale.toLowerCase().startsWith("zh");
    }
  } catch {
    // ignore
  }
  return (navigator.language || "").toLowerCase().startsWith("zh");
}

function getStrings() {
  if (isChineseLocale()) {
    return {
      extCategory: "OpenAI Image Gateway",
      secModels: "工作流与模型映射",
      secGlobal: "全局网关设置",
      baseUrl: "OpenAI Base URL",
      copy: "复制",
      copied: "已复制",
      viewModels: "查看 /v1/models",
      refresh: "刷新",
      addTitle: "添加本地工作流到 /v1/models",
      selectWorkflow: "本地工作流",
      loadingWorkflows: "正在读取本地工作流...",
      noWorkflows: "未找到已保存的本地工作流",
      useCurrentCanvas: "使用当前画布工作流",
      modelName: "模型名称（将出现在 /v1/models）",
      modelNamePlaceholder: "例如：qwen-edit 或 gpt-image-1",
      aliasesLabel: "别名（可选，逗号分隔）",
      aliasesPlaceholder: "留空即可；需要兼容多个名字时再填",
      setDefault: "设为默认",
      addBtn: "添加并启用",
      adding: "正在转换并分析工作流...",
      configuredTitle: "已暴露的模型列表",
      noModelsYet: "尚未配置任何工作流。请从上方选择一个本地工作流或使用当前画布工作流。",
      defaultBadge: "默认",
      capMultiEdit: (n) => `多图编辑 (最多 ${n} 张)`,
      capImg2Img: (n) => `图生图 (${n} 槽)`,
      capT2I: "文生图",
      sourcePrefix: "工作流：",
      canvasSource: "当前画布快照",
      syncBtn: "重新同步",
      editBtn: "高级配置",
      closeBtn: "收起",
      deleteBtn: "删除",
      confirmDelete: (id) => `确认从 /v1/models 移除工作流模型 "${id}"？`,
      saveModelBtn: "保存修改",
      fallbackLabel: "未匹配模型名时回退到默认工作流",
      fallbackTip: "当外部客户端写死 gpt-image-1 等名字且未单独配置时，自动路由到默认工作流。",
      exposeAliasesLabel: "在 /v1/models 中同时列出别名",
      exposeAliasesTip: "关闭时 /v1/models 仅展示你定义的主模型名称，更简洁。",
      respFormatLabel: "默认返回格式",
      apiKeysLabel: "API Keys（可选，逗号分隔）",
      apiKeysPlaceholder: "留空表示免鉴权",
      timeoutLabel: "单次请求超时（秒）",
      maxNLabel: "最大单次生成张数 (n)",
      saveGlobalBtn: "保存全局设置",
      savedOk: "已保存并立即生效",
      addedOk: (id, caps) => `已启用模型 "${id}"（${caps}）`,
      syncedOk: (id) => `已重新同步工作流 "${id}"`,
      deletedOk: (id) => `已移除模型 "${id}"`,
      needModelName: "请先填写模型名称（仅支持字母、数字及 ._:@-）",
      needWorkflow: "请先选择一个本地工作流，或点击「使用当前画布工作流」",
      advancedOutput: "输出节点 ID（留空自动选择）",
      advancedSizeMult: "尺寸对齐倍数",
      advancedTimeout: "单模型超时（秒，留空用全局）",
      advancedAllowNoImg: "允许无图输入（文生图）",
      advancedBindings: "参数绑定 JSON（bindings）",
      advancedImages: "图片输入配置 JSON（image_inputs）",
      corsNote: "提示：若由跨源浏览器网页直连，请在 ComfyUI 启动参数加上 --enable-cors-header \"*\" 并配置 API Keys。",
    };
  }
  return {
    extCategory: "OpenAI Image Gateway",
    secModels: "Workflow & Model Mappings",
    secGlobal: "Gateway Settings",
    baseUrl: "OpenAI Base URL",
    copy: "Copy",
    copied: "Copied",
    viewModels: "Open /v1/models",
    refresh: "Refresh",
    addTitle: "Add Local Workflow to /v1/models",
    selectWorkflow: "Local Workflow",
    loadingWorkflows: "Loading local workflows...",
    noWorkflows: "No saved local workflows found",
    useCurrentCanvas: "Use Current Canvas",
    modelName: "Model Name (listed in /v1/models)",
    modelNamePlaceholder: "e.g. qwen-edit or gpt-image-1",
    aliasesLabel: "Aliases (optional, comma-separated)",
    aliasesPlaceholder: "Leave empty unless needed",
    setDefault: "Set as default",
    addBtn: "Add & Enable",
    adding: "Converting & analyzing workflow...",
    configuredTitle: "Exposed Models",
    noModelsYet: "No workflows configured yet. Select a local workflow above or publish the current canvas.",
    defaultBadge: "Default",
    capMultiEdit: (n) => `Multi-Image Edit (up to ${n})`,
    capImg2Img: (n) => `Img2Img (${n} slot${n > 1 ? "s" : ""})`,
    capT2I: "Text-to-Image",
    sourcePrefix: "Workflow: ",
    canvasSource: "Canvas snapshot",
    syncBtn: "Re-sync",
    editBtn: "Advanced",
    closeBtn: "Close",
    deleteBtn: "Delete",
    confirmDelete: (id) => `Remove workflow model "${id}" from /v1/models?`,
    saveModelBtn: "Save Changes",
    fallbackLabel: "Fallback unknown model names to default workflow",
    fallbackTip: "Routes unrecognized model strings (such as hardcoded gpt-image-1) to your default workflow.",
    exposeAliasesLabel: "Also list aliases in /v1/models",
    exposeAliasesTip: "When off, /v1/models returns only your primary model names.",
    respFormatLabel: "Default Response Format",
    apiKeysLabel: "API Keys (optional, comma-separated)",
    apiKeysPlaceholder: "Leave empty to disable auth",
    timeoutLabel: "Request Timeout (seconds)",
    maxNLabel: "Max images per request (n)",
    saveGlobalBtn: "Save Global Settings",
    savedOk: "Saved and applied immediately",
    addedOk: (id, caps) => `Enabled model "${id}" (${caps})`,
    syncedOk: (id) => `Re-synced workflow "${id}"`,
    deletedOk: (id) => `Removed model "${id}"`,
    needModelName: "Please enter a model name (letters, digits, and ._:@-)",
    needWorkflow: "Please choose a local workflow or click Use Current Canvas",
    advancedOutput: "Output Node ID (blank = auto)",
    advancedSizeMult: "Size Alignment Multiple",
    advancedTimeout: "Model Timeout (seconds, blank = global)",
    advancedAllowNoImg: "Allow text-only input (no image)",
    advancedBindings: "Parameter Bindings JSON (bindings)",
    advancedImages: "Image Inputs JSON (image_inputs)",
    corsNote: "Tip: For direct cross-origin browser requests, start ComfyUI with --enable-cors-header \"*\" and configure API Keys.",
  };
}

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    div:has(> .oig-settings-root),
    div:has(> div > .oig-settings-root),
    div:has(> div > div > .oig-settings-root) {
      display: block !important;
      width: 100% !important;
      max-width: 100% !important;
    }
    div:has(> div > div > .oig-settings-root) > div:first-child:not(:has(.oig-settings-root)) {
      display: none !important;
    }
    .oig-settings-root {
      width: 100%;
      box-sizing: border-box;
      color: var(--input-text, #e5e7eb);
      font-size: 13px;
      line-height: 1.45;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 4px 0 12px;
    }
    .oig-card {
      background: var(--comfy-input-bg, rgba(255, 255, 255, 0.04));
      border: 1px solid var(--border-color, rgba(255, 255, 255, 0.12));
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .oig-card-title {
      font-weight: 600;
      font-size: 13.5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .oig-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .oig-grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
    }
    .oig-field {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }
    .oig-field label {
      font-size: 12px;
      color: var(--descrip-text, #9ca3af);
    }
    .oig-input, .oig-select, .oig-textarea {
      width: 100%;
      box-sizing: border-box;
      background: var(--comfy-menu-bg, #1f2024);
      color: var(--input-text, #f3f4f6);
      border: 1px solid var(--border-color, rgba(255, 255, 255, 0.16));
      border-radius: 6px;
      padding: 6px 9px;
      font-size: 12.5px;
      font-family: inherit;
    }
    .oig-textarea {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11.5px;
      min-height: 82px;
      resize: vertical;
    }
    .oig-btn {
      background: var(--comfy-menu-bg, #2a2c32);
      color: var(--input-text, #f3f4f6);
      border: 1px solid var(--border-color, rgba(255, 255, 255, 0.18));
      border-radius: 6px;
      padding: 5px 11px;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.15s ease, border-color 0.15s ease;
    }
    .oig-btn:hover:not(:disabled) {
      background: rgba(255, 255, 255, 0.1);
    }
    .oig-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .oig-btn-primary {
      background: #2563eb;
      border-color: #3b82f6;
      color: #ffffff;
    }
    .oig-btn-primary:hover:not(:disabled) {
      background: #1d4ed8;
    }
    .oig-btn-danger {
      color: #f87171;
      border-color: rgba(248, 113, 113, 0.35);
    }
    .oig-btn-danger:hover:not(:disabled) {
      background: rgba(248, 113, 113, 0.14);
    }
    .oig-code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border-color, rgba(255, 255, 255, 0.12));
      border-radius: 5px;
      padding: 4px 8px;
      word-break: break-all;
    }
    .oig-badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 11px;
      background: rgba(59, 130, 246, 0.18);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .oig-badge-default {
      background: rgba(16, 185, 129, 0.18);
      color: #6ee7b7;
      border-color: rgba(16, 185, 129, 0.35);
    }
    .oig-model-item {
      border: 1px solid var(--border-color, rgba(255, 255, 255, 0.12));
      border-radius: 7px;
      padding: 10px 12px;
      background: rgba(0, 0, 0, 0.16);
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .oig-muted {
      font-size: 11.5px;
      color: var(--descrip-text, #9ca3af);
    }
    .oig-status {
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 6px;
      display: none;
    }
    .oig-status.show { display: block; }
    .oig-status.ok {
      background: rgba(16, 185, 129, 0.16);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #6ee7b7;
    }
    .oig-status.err {
      background: rgba(239, 68, 68, 0.16);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }
    .oig-checkbox-row {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 12.5px;
      cursor: pointer;
      user-select: none;
    }
    .oig-advanced-box {
      border-top: 1px dashed var(--border-color, rgba(255, 255, 255, 0.14));
      padding-top: 10px;
      margin-top: 4px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
  `;
  document.head.appendChild(style);
}

function suggestModelName(workflowPath) {
  const base = String(workflowPath || "")
    .replace(/^.*\//, "")
    .replace(/\.json$/i, "")
    .trim();
  const cleaned = base
    .replace(/[^A-Za-z0-9._:@-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "")
    .replace(/-+/g, "-")
    .slice(0, 64);
  return cleaned || "comfy-workflow";
}

function isApiPromptFormat(obj) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
  if (Array.isArray(obj.nodes) && Array.isArray(obj.links)) return false;
  const values = Object.values(obj);
  return values.length > 0 && values.every((v) => v && typeof v === "object" && typeof v.class_type === "string");
}

async function fetchLocalWorkflowList() {
  try {
    const resp = await api.fetchApi("/userdata?dir=workflows&recurse=true");
    if (!resp.ok) return [];
    const list = await resp.json();
    if (!Array.isArray(list)) return [];
    return list
      .filter((item) => typeof item === "string" && item.toLowerCase().endsWith(".json"))
      .sort((a, b) => a.localeCompare(b));
  } catch (err) {
    console.warn("[OpenAI-Image-Gateway] Failed to list local workflows:", err);
    return [];
  }
}

async function fetchLocalWorkflowJson(workflowRelPath) {
  const encoded = encodeURIComponent(`workflows/${workflowRelPath}`);
  const resp = await api.fetchApi(`/userdata/${encoded}`);
  if (!resp.ok) {
    throw new Error(`Failed to read workflow "${workflowRelPath}" (HTTP ${resp.status})`);
  }
  return await resp.json();
}

async function convertWorkflowJsonToApiPrompt(rawJson) {
  if (isApiPromptFormat(rawJson)) {
    return rawJson;
  }

  const hasSubgraphs =
    Array.isArray(rawJson?.definitions?.subgraphs) && rawJson.definitions.subgraphs.length > 0;

  // Fast path: offscreen LGraph for standard workflows without subgraphs (no canvas touch).
  if (!hasSubgraphs) {
    try {
      const GraphCtor = app.rootGraph?.constructor || app.graph?.constructor;
      if (GraphCtor) {
        const tmpGraph = new GraphCtor();
        tmpGraph.configure(structuredClone(rawJson));
        const result = await app.graphToPrompt(tmpGraph);
        if (result?.output && isApiPromptFormat(result.output)) {
          return result.output;
        }
      }
    } catch (err) {
      console.debug("[OpenAI-Image-Gateway] Offscreen conversion fell back to loadGraphData:", err);
    }
  }

  // Full ComfyUI path (supports Subgraphs, widget migrations, and custom nodes):
  // snapshot current canvas, load target workflow silently, export API prompt, then restore.
  const prevGraph = app.rootGraph?.serialize ? app.rootGraph.serialize() : app.graph?.serialize?.();
  try {
    await app.loadGraphData(structuredClone(rawJson), true, false, null, {
      silent: true,
      skipAssetScans: true,
      silentAssetErrors: true,
    });
    const result = await app.graphToPrompt();
    if (!result?.output || !isApiPromptFormat(result.output)) {
      throw new Error("ComfyUI graphToPrompt() produced an empty or invalid API workflow.");
    }
    return result.output;
  } finally {
    if (prevGraph) {
      try {
        await app.loadGraphData(prevGraph, true, false, null, {
          silent: true,
          skipAssetScans: true,
          silentAssetErrors: true,
        });
      } catch (restoreErr) {
        console.warn("[OpenAI-Image-Gateway] Failed to restore previous canvas:", restoreErr);
      }
    }
  }
}

async function adminRequest(path, options = {}) {
  const resp = await api.fetchApi(`${ADMIN_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const msg = data?.error?.message || `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  return data;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k in node && typeof v === "boolean") node[k] = v;
    else node.setAttribute(k, String(v));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function formatCapabilities(model, t) {
  const parts = [];
  if (model.image_inputs) {
    if (model.image_inputs.mode === "dynamic_load_image") {
      parts.push(t.capMultiEdit(model.max_images || 10));
    } else {
      parts.push(t.capImg2Img(model.max_images || 1));
    }
  }
  if (model.allow_no_image) {
    parts.push(t.capT2I);
  }
  return parts;
}

function createGatewaySettingsPanel() {
  injectStyles();
  const t = getStrings();
  const root = el("div", { class: "oig-settings-root" });

  let state = null;
  let localWorkflows = [];
  let expandedModelId = null;

  const statusBox = el("div", { class: "oig-status" });
  function showStatus(msg, isError = false) {
    statusBox.textContent = msg;
    statusBox.className = `oig-status show ${isError ? "err" : "ok"}`;
  }
  function clearStatus() {
    statusBox.className = "oig-status";
  }

  // ---- Card 1: Endpoint Summary ----
  const baseUrlCode = el("span", { class: "oig-code", text: `${window.location.origin}/v1` });
  const copyBtn = el("button", {
    class: "oig-btn",
    type: "button",
    text: t.copy,
    onclick: async () => {
      try {
        await navigator.clipboard.writeText(baseUrlCode.textContent);
        copyBtn.textContent = t.copied;
        setTimeout(() => (copyBtn.textContent = t.copy), 1500);
      } catch {
        // fallback ignored
      }
    },
  });
  const openModelsBtn = el("button", {
    class: "oig-btn",
    type: "button",
    text: t.viewModels,
    onclick: () => window.open("/v1/models", "_blank", "noopener"),
  });
  const refreshBtn = el("button", {
    class: "oig-btn",
    type: "button",
    text: t.refresh,
    onclick: () => loadAll(),
  });
  const listedPreview = el("div", { class: "oig-muted" });

  const headerCard = el("div", { class: "oig-card" }, [
    el("div", { class: "oig-card-title" }, [
      el("span", { text: t.baseUrl }),
      el("div", { class: "oig-row" }, [copyBtn, openModelsBtn, refreshBtn]),
    ]),
    baseUrlCode,
    listedPreview,
    el("div", { class: "oig-muted", text: t.corsNote }),
  ]);

  // ---- Card 2: Add Workflow ----
  const wfSelect = el("select", { class: "oig-select" });
  const nameInput = el("input", {
    class: "oig-input",
    type: "text",
    placeholder: t.modelNamePlaceholder,
  });
  const aliasesInput = el("input", {
    class: "oig-input",
    type: "text",
    placeholder: t.aliasesPlaceholder,
  });
  const defaultCheck = el("input", { type: "checkbox" });
  let nameTouchedByUser = false;
  nameInput.addEventListener("input", () => {
    nameTouchedByUser = nameInput.value.trim().length > 0;
  });
  wfSelect.addEventListener("change", () => {
    if (!nameTouchedByUser && wfSelect.value) {
      nameInput.value = suggestModelName(wfSelect.value);
    }
  });

  const addSelectedBtn = el("button", {
    class: "oig-btn oig-btn-primary",
    type: "button",
    text: t.addBtn,
    onclick: () => handleAddWorkflow(false),
  });
  const addCanvasBtn = el("button", {
    class: "oig-btn",
    type: "button",
    text: t.useCurrentCanvas,
    onclick: () => handleAddWorkflow(true),
  });

  async function handleAddWorkflow(fromCanvas) {
    clearStatus();
    const selectedWf = wfSelect.value;
    if (!fromCanvas && !selectedWf) {
      showStatus(t.needWorkflow, true);
      return;
    }
    let modelId = nameInput.value.trim();
    if (!modelId) {
      if (!fromCanvas && selectedWf) {
        modelId = suggestModelName(selectedWf);
        nameInput.value = modelId;
      } else {
        showStatus(t.needModelName, true);
        return;
      }
    }

    addSelectedBtn.disabled = true;
    addCanvasBtn.disabled = true;
    showStatus(t.adding, false);
    try {
      let apiPrompt;
      let sourceWorkflow = "";
      if (fromCanvas) {
        const res = await app.graphToPrompt();
        if (!res?.output || !isApiPromptFormat(res.output)) {
          throw new Error("Current canvas could not be exported as a valid ComfyUI API workflow.");
        }
        apiPrompt = res.output;
      } else {
        const rawJson = await fetchLocalWorkflowJson(selectedWf);
        apiPrompt = await convertWorkflowJsonToApiPrompt(rawJson);
        sourceWorkflow = selectedWf;
      }

      const aliases = aliasesInput.value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);

      const res = await adminRequest("/models", {
        method: "POST",
        body: JSON.stringify({
          id: modelId,
          workflow_prompt: apiPrompt,
          source_workflow: sourceWorkflow,
          aliases,
          set_default: defaultCheck.checked,
          auto_analyze: true,
        }),
      });
      state = res.state;
      nameInput.value = "";
      aliasesInput.value = "";
      defaultCheck.checked = false;
      nameTouchedByUser = false;
      renderState();
      const added = state.models.find((m) => m.id === modelId);
      const caps = added ? formatCapabilities(added, t).join(" / ") : "";
      showStatus(t.addedOk(modelId, caps || "OK"), false);
    } catch (err) {
      showStatus(String(err?.message || err), true);
    } finally {
      addSelectedBtn.disabled = false;
      addCanvasBtn.disabled = false;
    }
  }

  const addCard = el("div", { class: "oig-card" }, [
    el("div", { class: "oig-card-title" }, [
      el("span", { text: t.addTitle }),
      addCanvasBtn,
    ]),
    el("div", { class: "oig-field" }, [
      el("label", { text: t.selectWorkflow }),
      wfSelect,
    ]),
    el("div", { class: "oig-grid-2" }, [
      el("div", { class: "oig-field" }, [
        el("label", { text: t.modelName }),
        nameInput,
      ]),
      el("div", { class: "oig-field" }, [
        el("label", { text: t.aliasesLabel }),
        aliasesInput,
      ]),
    ]),
    el("div", { class: "oig-row", style: "justify-content: space-between;" }, [
      el("label", { class: "oig-checkbox-row" }, [defaultCheck, el("span", { text: t.setDefault })]),
      addSelectedBtn,
    ]),
  ]);

  // ---- Card 3: Configured Models List ----
  const modelsListBox = el("div", { style: "display:flex;flex-direction:column;gap:8px;" });
  const modelsCard = el("div", { class: "oig-card" }, [
    el("div", { class: "oig-card-title" }, [el("span", { text: t.configuredTitle })]),
    modelsListBox,
  ]);

  // ---- Card 4: Global Settings ----
  const fallbackCheck = el("input", { type: "checkbox" });
  const exposeAliasesCheck = el("input", { type: "checkbox" });
  const respFormatSelect = el("select", { class: "oig-select" }, [
    el("option", { value: "b64_json", text: "b64_json (OpenAI gpt-image-1)" }),
    el("option", { value: "url", text: "url (/view?filename=...)" }),
  ]);
  const apiKeysInput = el("input", {
    class: "oig-input",
    type: "text",
    placeholder: t.apiKeysPlaceholder,
  });
  const timeoutInput = el("input", { class: "oig-input", type: "number", min: "0", step: "10" });
  const maxNInput = el("input", { class: "oig-input", type: "number", min: "1", max: "16", step: "1" });
  const saveGlobalBtn = el("button", {
    class: "oig-btn oig-btn-primary",
    type: "button",
    text: t.saveGlobalBtn,
    onclick: async () => {
      clearStatus();
      saveGlobalBtn.disabled = true;
      try {
        const res = await adminRequest("/settings", {
          method: "PUT",
          body: JSON.stringify({
            fallback_to_default_model: fallbackCheck.checked,
            expose_aliases: exposeAliasesCheck.checked,
            default_response_format: respFormatSelect.value,
            api_keys: apiKeysInput.value,
            timeout: Number(timeoutInput.value || 600),
            max_n: Number(maxNInput.value || 4),
          }),
        });
        state = res.state;
        renderState();
        showStatus(t.savedOk, false);
      } catch (err) {
        showStatus(String(err?.message || err), true);
      } finally {
        saveGlobalBtn.disabled = false;
      }
    },
  });

  const globalCard = el("div", { class: "oig-card" }, [
    el("div", { class: "oig-card-title" }, [el("span", { text: t.secGlobal })]),
    el("label", { class: "oig-checkbox-row", title: t.fallbackTip }, [
      fallbackCheck,
      el("span", { text: t.fallbackLabel }),
    ]),
    el("label", { class: "oig-checkbox-row", title: t.exposeAliasesTip }, [
      exposeAliasesCheck,
      el("span", { text: t.exposeAliasesLabel }),
    ]),
    el("div", { class: "oig-grid-2" }, [
      el("div", { class: "oig-field" }, [el("label", { text: t.respFormatLabel }), respFormatSelect]),
      el("div", { class: "oig-field" }, [el("label", { text: t.apiKeysLabel }), apiKeysInput]),
      el("div", { class: "oig-field" }, [el("label", { text: t.timeoutLabel }), timeoutInput]),
      el("div", { class: "oig-field" }, [el("label", { text: t.maxNLabel }), maxNInput]),
    ]),
    el("div", { class: "oig-row", style: "justify-content: flex-end;" }, [saveGlobalBtn]),
  ]);

  root.appendChild(statusBox);
  root.appendChild(headerCard);
  root.appendChild(addCard);
  root.appendChild(modelsCard);
  root.appendChild(globalCard);

  function renderWorkflowOptions() {
    const prev = wfSelect.value;
    wfSelect.innerHTML = "";
    if (!localWorkflows.length) {
      wfSelect.appendChild(el("option", { value: "", text: t.noWorkflows }));
      return;
    }
    for (const wf of localWorkflows) {
      wfSelect.appendChild(el("option", { value: wf, text: wf }));
    }
    if (prev && localWorkflows.includes(prev)) {
      wfSelect.value = prev;
    }
    if (!nameTouchedByUser && wfSelect.value && !nameInput.value) {
      nameInput.value = suggestModelName(wfSelect.value);
    }
  }

  function renderModelCard(model) {
    const badges = [];
    if (model.is_default) {
      badges.push(el("span", { class: "oig-badge oig-badge-default", text: t.defaultBadge }));
    }
    for (const cap of formatCapabilities(model, t)) {
      badges.push(el("span", { class: "oig-badge", text: cap }));
    }

    const bindSummary = [];
    if (model.bindings?.prompt) {
      const p = Array.isArray(model.bindings.prompt) ? model.bindings.prompt.join(", ") : model.bindings.prompt;
      bindSummary.push(`Prompt: ${p}`);
    }
    if (model.output_node) {
      bindSummary.push(`Output: ${model.output_node}`);
    }
    if (model.aliases?.length) {
      bindSummary.push(`Aliases: ${model.aliases.join(", ")}`);
    }

    const actions = [];
    if (!model.is_default) {
      actions.push(
        el("button", {
          class: "oig-btn",
          type: "button",
          text: t.setDefault,
          onclick: async () => {
            clearStatus();
            try {
              const res = await adminRequest("/settings", {
                method: "PUT",
                body: JSON.stringify({ default_model: model.id }),
              });
              state = res.state;
              renderState();
              showStatus(t.savedOk, false);
            } catch (err) {
              showStatus(String(err?.message || err), true);
            }
          },
        })
      );
    }
    if (model.source_workflow) {
      actions.push(
        el("button", {
          class: "oig-btn",
          type: "button",
          text: t.syncBtn,
          onclick: async () => {
            clearStatus();
            showStatus(t.adding, false);
            try {
              const rawJson = await fetchLocalWorkflowJson(model.source_workflow);
              const apiPrompt = await convertWorkflowJsonToApiPrompt(rawJson);
              const res = await adminRequest("/models", {
                method: "POST",
                body: JSON.stringify({
                  id: model.id,
                  workflow_prompt: apiPrompt,
                  source_workflow: model.source_workflow,
                  aliases: model.aliases || [],
                  set_default: model.is_default,
                  auto_analyze: true,
                }),
              });
              state = res.state;
              renderState();
              showStatus(t.syncedOk(model.id), false);
            } catch (err) {
              showStatus(String(err?.message || err), true);
            }
          },
        })
      );
    }

    const isExpanded = expandedModelId === model.id;
    actions.push(
      el("button", {
        class: "oig-btn",
        type: "button",
        text: isExpanded ? t.closeBtn : t.editBtn,
        onclick: () => {
          expandedModelId = isExpanded ? null : model.id;
          renderState();
        },
      })
    );
    actions.push(
      el("button", {
        class: "oig-btn oig-btn-danger",
        type: "button",
        text: t.deleteBtn,
        onclick: async () => {
          if (!window.confirm(t.confirmDelete(model.id))) return;
          clearStatus();
          try {
            const res = await adminRequest(`/models/${encodeURIComponent(model.id)}`, {
              method: "DELETE",
            });
            state = res.state;
            if (expandedModelId === model.id) expandedModelId = null;
            renderState();
            showStatus(t.deletedOk(model.id), false);
          } catch (err) {
            showStatus(String(err?.message || err), true);
          }
        },
      })
    );

    const children = [
      el("div", { class: "oig-row", style: "justify-content: space-between;" }, [
        el("div", { class: "oig-row" }, [
          el("strong", { text: model.id }),
          ...badges,
        ]),
        el("div", { class: "oig-row" }, actions),
      ]),
      el("div", {
        class: "oig-muted",
        text: `${t.sourcePrefix}${model.source_workflow || model.workflow || t.canvasSource}`,
      }),
    ];
    if (bindSummary.length) {
      children.push(el("div", { class: "oig-muted", text: bindSummary.join("  ·  ") }));
    }

    if (isExpanded) {
      const editId = el("input", { class: "oig-input", type: "text", value: model.id });
      const editAliases = el("input", {
        class: "oig-input",
        type: "text",
        value: (model.aliases || []).join(", "),
      });
      const editOutput = el("input", {
        class: "oig-input",
        type: "text",
        value: model.output_node || "",
      });
      const editSizeMult = el("input", {
        class: "oig-input",
        type: "number",
        value: String(model.size_multiple || 16),
      });
      const editTimeout = el("input", {
        class: "oig-input",
        type: "number",
        value: model.timeout != null ? String(model.timeout) : "",
      });
      const editAllowNoImg = el("input", { type: "checkbox", checked: !!model.allow_no_image });
      const editBindings = el("textarea", { class: "oig-textarea" });
      editBindings.value = JSON.stringify(model.bindings || {}, null, 2);
      const editImages = el("textarea", { class: "oig-textarea" });
      editImages.value = model.image_inputs ? JSON.stringify(model.image_inputs, null, 2) : "";

      const saveEditBtn = el("button", {
        class: "oig-btn oig-btn-primary",
        type: "button",
        text: t.saveModelBtn,
        onclick: async () => {
          clearStatus();
          try {
            const parsedBindings = editBindings.value.trim() ? JSON.parse(editBindings.value) : {};
            const parsedImages = editImages.value.trim() ? JSON.parse(editImages.value) : null;
            const res = await adminRequest("/models", {
              method: "POST",
              body: JSON.stringify({
                previous_id: model.id,
                id: editId.value.trim(),
                aliases: editAliases.value,
                output_node: editOutput.value.trim() || null,
                size_multiple: Number(editSizeMult.value || 16),
                timeout: editTimeout.value.trim() ? Number(editTimeout.value) : null,
                allow_no_image: editAllowNoImg.checked,
                bindings: parsedBindings,
                image_inputs: parsedImages,
                auto_analyze: false,
              }),
            });
            state = res.state;
            expandedModelId = res.model_id;
            renderState();
            showStatus(t.savedOk, false);
          } catch (err) {
            showStatus(String(err?.message || err), true);
          }
        },
      });

      children.push(
        el("div", { class: "oig-advanced-box" }, [
          el("div", { class: "oig-grid-2" }, [
            el("div", { class: "oig-field" }, [el("label", { text: t.modelName }), editId]),
            el("div", { class: "oig-field" }, [el("label", { text: t.aliasesLabel }), editAliases]),
            el("div", { class: "oig-field" }, [el("label", { text: t.advancedOutput }), editOutput]),
            el("div", { class: "oig-field" }, [el("label", { text: t.advancedSizeMult }), editSizeMult]),
            el("div", { class: "oig-field" }, [el("label", { text: t.advancedTimeout }), editTimeout]),
          ]),
          el("label", { class: "oig-checkbox-row" }, [
            editAllowNoImg,
            el("span", { text: t.advancedAllowNoImg }),
          ]),
          el("div", { class: "oig-field" }, [el("label", { text: t.advancedBindings }), editBindings]),
          el("div", { class: "oig-field" }, [el("label", { text: t.advancedImages }), editImages]),
          el("div", { class: "oig-row", style: "justify-content: flex-end;" }, [saveEditBtn]),
        ])
      );
    }

    return el("div", { class: "oig-model-item" }, children);
  }

  function renderState() {
    if (!state) return;
    baseUrlCode.textContent = state.base_url || `${window.location.origin}/v1`;
    const listed = state.listed_models || [];
    listedPreview.textContent = listed.length
      ? `/v1/models → [ ${listed.join(", ")} ]`
      : "/v1/models → [ ]";

    modelsListBox.innerHTML = "";
    if (!state.models?.length) {
      modelsListBox.appendChild(el("div", { class: "oig-muted", text: t.noModelsYet }));
    } else {
      for (const m of state.models) {
        modelsListBox.appendChild(renderModelCard(m));
      }
    }

    const s = state.settings || {};
    fallbackCheck.checked = !!s.fallback_to_default_model;
    exposeAliasesCheck.checked = !!s.expose_aliases;
    respFormatSelect.value = s.default_response_format || "b64_json";
    apiKeysInput.value = (s.api_keys || []).join(", ");
    timeoutInput.value = String(s.timeout ?? 600);
    maxNInput.value = String(s.max_n ?? 4);
  }

  async function loadAll() {
    wfSelect.innerHTML = "";
    wfSelect.appendChild(el("option", { value: "", text: t.loadingWorkflows }));
    try {
      const [cfgRes, wfList] = await Promise.all([
        adminRequest("/config"),
        fetchLocalWorkflowList(),
      ]);
      state = cfgRes;
      localWorkflows = wfList;
      renderWorkflowOptions();
      renderState();
    } catch (err) {
      showStatus(String(err?.message || err), true);
    }
  }

  loadAll();
  return root;
}

app.registerExtension({
  name: "OpenAIImageGateway.Settings",
  settings: [
    {
      id: "OpenAIImageGateway.WorkflowManager",
      name: "Workflow & Model Mappings",
      category: ["OpenAI Image Gateway", "Workflows", "Manager"],
      type: () => createGatewaySettingsPanel(),
      defaultValue: null,
    },
  ],
});
