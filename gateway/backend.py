"""ComfyUI execution backends (in-process PromptServer backend + abstract interface)."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from .models import GatewayError, RenderedImage

log = logging.getLogger("comfyui_openai_image_gateway")


class ComfyBackend(ABC):
    """Abstract interface between gateway routes and ComfyUI execution."""

    @abstractmethod
    async def save_input_image(self, data: bytes, filename: str, subfolder: str) -> str:
        """Persist an uploaded image for ComfyUI LoadImage and return its annotated path."""

    @abstractmethod
    async def cleanup_input_files(self, annotated_paths: list[str]) -> None:
        """Delete temporary uploaded input images after ComfyUI finishes using them."""

    @abstractmethod
    async def run_workflow(
        self,
        prompt: dict[str, Any],
        *,
        output_node: str | None = None,
        timeout: float = 600.0,
        client_id: str = "openai-image-gateway",
    ) -> list[RenderedImage]:
        """Validate, enqueue, wait for completion, and return output images."""


def _extract_error_message(status: dict[str, Any]) -> str:
    messages = status.get("messages") or []
    for entry in messages:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == "execution_error":
            detail = entry[1]
            if isinstance(detail, dict):
                node_type = detail.get("node_type") or detail.get("node_id") or "node"
                exc_msg = detail.get("exception_message") or detail.get("exception_type") or "execution error"
                return f"ComfyUI execution failed in `{node_type}`: {str(exc_msg).strip()}"
            return f"ComfyUI execution failed: {detail}"
    return "ComfyUI workflow execution failed."


def _collect_image_refs(outputs: dict[str, Any], output_node: str | None) -> list[dict[str, Any]]:
    if output_node is not None:
        node_out = outputs.get(str(output_node)) or {}
        imgs = node_out.get("images") or []
        return [item for item in imgs if isinstance(item, dict) and item.get("filename")]

    saved: list[dict[str, Any]] = []
    temp: list[dict[str, Any]] = []
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for item in node_out.get("images") or ():
            if not isinstance(item, dict) or not item.get("filename"):
                continue
            if item.get("type", "output") == "output":
                saved.append(item)
            else:
                temp.append(item)
    return saved if saved else temp


def _is_prompt_active(prompt_queue: Any, prompt_id: str) -> bool:
    running, queued = prompt_queue.get_current_queue_volatile()
    for item in running:
        if len(item) > 1 and item[1] == prompt_id:
            return True
    for item in queued:
        if len(item) > 1 and item[1] == prompt_id:
            return True
    return False


def _is_prompt_queued(prompt_queue: Any, prompt_id: str) -> bool:
    _, queued = prompt_queue.get_current_queue_volatile()
    return any(len(item) > 1 and item[1] == prompt_id for item in queued)


def _is_within(base_dir: Path, target: Path) -> bool:
    try:
        base_real = base_dir.resolve()
        target_real = target.resolve()
        return os.path.commonpath((str(base_real), str(target_real))) == str(base_real)
    except ValueError:
        return False


class InProcessComfyBackend(ComfyBackend):
    """Executes workflows directly against ComfyUI's in-process PromptServer and PromptQueue."""

    def __init__(
        self,
        prompt_server: Any,
        *,
        execution_module: Any | None = None,
        folder_paths_module: Any | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        self._server = prompt_server
        self._execution = execution_module
        self._folder_paths = folder_paths_module
        self._poll_interval = max(0.01, float(poll_interval))

    def _get_execution(self) -> Any:
        if self._execution is None:
            import execution

            self._execution = execution
        return self._execution

    def _get_folder_paths(self) -> Any:
        if self._folder_paths is None:
            import folder_paths

            self._folder_paths = folder_paths
        return self._folder_paths

    async def save_input_image(self, data: bytes, filename: str, subfolder: str) -> str:
        folder_paths = self._get_folder_paths()
        input_root = Path(folder_paths.get_input_directory()).resolve()
        safe_sub = Path(os.path.normpath(subfolder.strip("/\\"))) if subfolder else Path()
        safe_name = os.path.basename(filename)
        if not safe_name:
            raise GatewayError("Invalid image filename.", status=400, param="image")

        target_dir = (input_root / safe_sub).resolve()
        target_path = (target_dir / safe_name).resolve()
        if not _is_within(input_root, target_path):
            raise GatewayError("Invalid input image storage path.", status=500, error_type="server_error")

        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, target_path)

        rel = target_path.relative_to(input_root).as_posix()
        return rel

    async def cleanup_input_files(self, annotated_paths: list[str]) -> None:
        if not annotated_paths:
            return
        folder_paths = self._get_folder_paths()
        input_root = Path(folder_paths.get_input_directory()).resolve()
        for rel in annotated_paths:
            try:
                target = (input_root / rel).resolve()
                if _is_within(input_root, target) and target.is_file():
                    target.unlink(missing_ok=True)
            except OSError as exc:
                log.debug("Failed to remove temporary input file %s: %s", rel, exc)

    def _read_output_image(self, ref: dict[str, Any]) -> RenderedImage:
        folder_paths = self._get_folder_paths()
        file_type = str(ref.get("type") or "output")
        subfolder = str(ref.get("subfolder") or "")
        filename = os.path.basename(str(ref.get("filename") or ""))
        if not filename:
            raise GatewayError("ComfyUI returned an empty output filename.", status=502, error_type="server_error")

        base_dir_str = folder_paths.get_directory_by_type(file_type)
        if not base_dir_str:
            raise GatewayError(
                f"Unknown ComfyUI output directory type `{file_type}`.",
                status=502,
                error_type="server_error",
            )
        base_dir = Path(base_dir_str).resolve()
        target = (base_dir / os.path.normpath(subfolder) / filename).resolve() if subfolder else (base_dir / filename).resolve()
        if not _is_within(base_dir, target) or not target.is_file():
            raise GatewayError(
                f"Output image `{filename}` was not found in ComfyUI `{file_type}` directory.",
                status=502,
                error_type="server_error",
            )

        raw = target.read_bytes()
        width = 0
        height = 0
        ext = target.suffix.lstrip(".").lower() or "png"
        try:
            with Image.open(io.BytesIO(raw)) as img:
                width, height = img.size
                if img.format:
                    ext = "jpg" if img.format.upper() == "JPEG" else img.format.lower()
        except OSError:
            pass

        return RenderedImage(
            data=raw,
            filename=filename,
            subfolder=subfolder,
            file_type=file_type,
            ext=ext,
            width=width,
            height=height,
        )

    async def run_workflow(
        self,
        prompt: dict[str, Any],
        *,
        output_node: str | None = None,
        timeout: float = 600.0,
        client_id: str = "openai-image-gateway",
    ) -> list[RenderedImage]:
        execution = self._get_execution()
        prompt_id = str(uuid.uuid4())
        payload: dict[str, Any] = {"prompt": prompt, "client_id": client_id}
        if hasattr(self._server, "trigger_on_prompt"):
            payload = self._server.trigger_on_prompt(payload)
        prompt = payload.get("prompt", prompt)

        replace_mgr = getattr(self._server, "node_replace_manager", None)
        if replace_mgr is not None and hasattr(replace_mgr, "apply_replacements"):
            replace_mgr.apply_replacements(prompt)

        valid = await execution.validate_prompt(prompt_id, prompt, None)
        if not valid[0]:
            err_info = valid[1]
            node_errors = valid[3]
            msg = err_info.get("message", str(err_info)) if isinstance(err_info, dict) else str(err_info)
            details = err_info.get("details") if isinstance(err_info, dict) else None
            full_msg = f"Workflow validation failed: {msg}"
            if details:
                full_msg += f" ({details})"
            if node_errors:
                full_msg += f" | node_errors={node_errors}"
            raise GatewayError(
                full_msg,
                status=400,
                error_type="invalid_request_error",
                code="invalid_workflow",
            )

        outputs_to_execute = valid[2]
        extra_data: dict[str, Any] = {
            "client_id": client_id,
            "create_time": int(time.time() * 1000),
        }
        sensitive: dict[str, Any] = {}
        number = self._server.number
        self._server.number += 1

        prompt_queue = self._server.prompt_queue
        prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))

        deadline = (time.monotonic() + timeout) if timeout and timeout > 0 else None
        missing_ticks = 0
        interval = self._poll_interval

        try:
            while True:
                history = prompt_queue.get_history(prompt_id=prompt_id)
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise GatewayError(
                            _extract_error_message(status),
                            status=502,
                            error_type="server_error",
                            code="workflow_execution_error",
                        )
                    refs = _collect_image_refs(entry.get("outputs") or {}, output_node)
                    if not refs:
                        raise GatewayError(
                            "ComfyUI workflow completed without producing any output images.",
                            status=502,
                            error_type="server_error",
                            code="no_output_image",
                        )
                    return [self._read_output_image(ref) for ref in refs]

                if _is_prompt_active(prompt_queue, prompt_id):
                    missing_ticks = 0
                else:
                    missing_ticks += 1
                    if missing_ticks >= 2:
                        raise GatewayError(
                            "ComfyUI job was cancelled before completion.",
                            status=502,
                            error_type="server_error",
                            code="job_cancelled",
                        )

                if deadline is not None and time.monotonic() >= deadline:
                    if _is_prompt_queued(prompt_queue, prompt_id):
                        prompt_queue.delete_queue_item(lambda item: len(item) > 1 and item[1] == prompt_id)
                    elif hasattr(prompt_queue, "interrupt_if_running"):
                        prompt_queue.interrupt_if_running(prompt_id)
                    raise GatewayError(
                        f"Image generation timed out after {timeout:.0f}s.",
                        status=504,
                        error_type="server_error",
                        code="timeout",
                    )

                await asyncio.sleep(interval)
                interval = min(0.25, interval * 1.25)
        except asyncio.CancelledError:
            if _is_prompt_queued(prompt_queue, prompt_id):
                prompt_queue.delete_queue_item(lambda item: len(item) > 1 and item[1] == prompt_id)
            raise
