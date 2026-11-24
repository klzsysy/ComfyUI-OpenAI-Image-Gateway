# ComfyUI-OpenAI-Image-Gateway

A lightweight **ComfyUI custom node** that exposes your local ComfyUI workflows as an **OpenAI-compatible Images API** directly on ComfyUI's existing HTTP server — with first-class **multi-image editing** support (`image[]` multipart and JSON multi-image).

将本地 ComfyUI 工作流直接以 **OpenAI 兼容图片接口** 对外暴露的轻量插件，复用 ComfyUI 原有端口，原生支持 **多图编辑（`image[]` multipart 与 JSON 多图）**，无需额外安装任何 Python 第三方包。

## Features

- **In-process ComfyUI plugin**: Runs directly inside ComfyUI (`PromptServer` + `PromptQueue`), requiring **zero extra dependencies** (uses only ComfyUI's built-in `aiohttp`, `PyYAML`, and `Pillow`) and no loopback HTTP port guessing.
- **OpenAI-compatible endpoints**:
  - `GET /v1/models` & `GET /v1/models/{model_id}` (for client "Test Connection" and model discovery)
  - `POST /v1/images/edits` (`multipart/form-data` supporting `image[]`, `image`, `images[]`, `images`, `image_1..N`, optional `mask`, plus JSON fallback)
  - `POST /v1/images/generations` (`application/json` supporting both text-to-image and multi-image JSON payloads via `image`, `images`, or `reference_images`)
  - `GET /v1/gateway/health` & `POST /v1/gateway/reload` (hot-reload config & workflows without restarting ComfyUI)
- **Multi-image workflow wiring**:
  - `dynamic_load_image`: Automatically creates `LoadImage` nodes for 1–N uploaded images and connects them to autogrow nodes such as `TextEncodeQwenImage21` (`images.image_1..N`) or `BatchImagesNode`.
  - `load_image_nodes`: Maps 1–N uploaded images into pre-existing `LoadImage` nodes and prunes unused slots cleanly.
- **`gpt-image-1` friendly defaults**:
  - Defaults to `b64_json` output (supports `response_format=url` and `output_format=png|jpeg|webp`).
  - Supports model aliases (e.g., expose your local workflow as `gpt-image-1`) and optional fallback to `default_model`.
  - Built-in CORS headers, `OPTIONS` preflight, and Chrome Private Network Access (`Access-Control-Allow-Private-Network: true`).
  - Optional Bearer / `X-API-Key` authentication and automatic temporary input-file cleanup.

## Installation

Clone directly into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/klzsysy/ComfyUI-OpenAI-Image-Gateway.git
```

No `pip install` step is required.

## Configuration

### Option A: ComfyUI Settings UI (Recommended)

Open ComfyUI → **Settings** → **OpenAI Image Gateway**:

1. Select any saved local ComfyUI workflow from the dropdown (or click **Use Current Canvas**).
2. Enter the **Model Name** you want exposed in `/v1/models` (for example, `qwen-edit` or `gpt-image-1`).
3. Click **Add & Enable** — the plugin automatically converts the workflow to API format (including Subgraphs), detects `prompt`, `negative_prompt`, `seed`, `size`, `output_node`, and multi-image edit inputs (`TextEncodeQwenImage21` / `LoadImage`), saves `config.local.yaml`, and hot-reloads immediately without restarting ComfyUI.
4. Add as many workflows as you like — calling `"model": "<name>"` routes directly to that workflow.

### Option B: Manual YAML Config

1. Export your ComfyUI workflow in **API format** (`Workflow -> Export (API)` in ComfyUI) and save it under `workflows/local/` (gitignored).
2. Copy `config.example.yaml` to `config.local.yaml` (gitignored) and point `workflow`, `output_node`, `image_inputs`, and `bindings` to your workflow's node IDs:

```bash
cd ComfyUI/custom_nodes/ComfyUI-OpenAI-Image-Gateway
cp config.example.yaml config.local.yaml
```

3. Restart ComfyUI (or call `POST /v1/gateway/reload` after editing `config.local.yaml`).

### Environment Overrides

| Variable | Description |
|---|---|
| `COMFY_OPENAI_GATEWAY_CONFIG` | Explicit path to a YAML or JSON config file |
| `COMFY_OPENAI_GATEWAY_API_KEYS` | Comma-separated Bearer / `X-API-Key` tokens (empty disables auth) |
| `COMFY_OPENAI_GATEWAY_CORS_ORIGINS` | Allowed CORS origins (`*` or comma-separated list; empty disables) |
| `COMFY_OPENAI_GATEWAY_RESPONSE_FORMAT` | Default `response_format` (`b64_json` or `url`) |
| `COMFY_OPENAI_GATEWAY_PUBLIC_BASE_URL` | Base URL prefix used when `response_format=url` |
| `COMFY_OPENAI_GATEWAY_TIMEOUT` | Workflow execution timeout in seconds (`0` disables timeout) |
| `COMFY_OPENAI_GATEWAY_KEEP_INPUT_FILES` | Set `true` to keep temporary uploaded images in `input/openai_gateway/` |

## API Usage Examples

Assuming ComfyUI is listening on `http://127.0.0.1:8188`:

### 1. Test Connection / List Models

```bash
curl http://127.0.0.1:8188/v1/models
```

### 2. Multi-Image Edit (`POST /v1/images/edits`, OpenAI `image[]` multipart)

```bash
curl -X POST http://127.0.0.1:8188/v1/images/edits \
  -H "Authorization: Bearer sk-optional" \
  -F "model=gpt-image-1" \
  -F "prompt=Dress the person in image 1 wearing the outfit from image 2" \
  -F "image[]=@person.png" \
  -F "image[]=@outfit.png"
```

### 3. Multi-Image Edit via JSON (`POST /v1/images/generations`)

```bash
curl -X POST http://127.0.0.1:8188/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-1",
    "prompt": "Combine the subject from image 1 with the outfit in image 2",
    "image": [
      "data:image/png;base64,...",
      "data:image/png;base64,..."
    ],
    "response_format": "b64_json"
  }'
```

### 4. Client Configuration Notes

- **Base URL**: `http://<comfyui-host>:<port>/v1` (e.g. `http://127.0.0.1:8188/v1`)
- **Model**: `gpt-image-1` (or any configured model ID / alias)
- **API Key**: Any non-empty dummy string if `api_keys` is empty, or one of your configured keys.
- **Browser CORS & ComfyUI**: By default, ComfyUI enables `origin_only_middleware` (blocking `Sec-Fetch-Site: cross-site` browser requests with HTTP 403 and intercepting `OPTIONS` preflight). For server-side callers (OpenAI SDK, curl, backend relays), no extra ComfyUI flags are needed. If you call `/v1/*` directly from a web app on another browser origin, start ComfyUI with `--enable-cors-header "*"` (or your specific origin) and configure `api_keys` in `config.local.yaml`.
- **LAN vs Public Web Clients**: If your frontend is hosted on a public HTTPS site, a server-side relay cannot reach a private LAN IP (`192.168.x.x`), and browsers block `https://` -> `http://` mixed content. For direct browser calls to a LAN ComfyUI server, run the frontend locally or put an HTTPS reverse proxy in front of ComfyUI.

## Running Tests

```bash
pytest -q
```

## License

MIT
