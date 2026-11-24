"""Tests for InProcessComfyBackend using a realistic fake PromptServer/PromptQueue."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from gateway.backend import InProcessComfyBackend
from gateway.models import GatewayError


def _png_bytes(size=(24, 16), color=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeFolderPaths:
    def __init__(self, root: Path) -> None:
        self.input_dir = root / "input"
        self.output_dir = root / "output"
        self.temp_dir = root / "temp"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_input_directory(self) -> str:
        return str(self.input_dir)

    def get_directory_by_type(self, name: str) -> str | None:
        return {
            "input": str(self.input_dir),
            "output": str(self.output_dir),
            "temp": str(self.temp_dir),
        }.get(name)


class FakePromptQueue:
    def __init__(self) -> None:
        self.running: list[Any] = []
        self.queued: list[Any] = []
        self.history: dict[str, Any] = {}
        self.interrupted: list[str] = []
        self.on_put = None

    def put(self, item: Any) -> None:
        self.queued.append(item)
        if self.on_put is not None:
            self.on_put(item)

    def get_history(self, prompt_id: str | None = None) -> dict[str, Any]:
        if prompt_id is None:
            return dict(self.history)
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}

    def get_current_queue_volatile(self):
        return list(self.running), list(self.queued)

    def delete_queue_item(self, fn) -> None:
        self.queued = [item for item in self.queued if not fn(item)]

    def interrupt_if_running(self, prompt_id: str) -> bool:
        self.interrupted.append(prompt_id)
        return True


class FakePromptServer:
    def __init__(self) -> None:
        self.number = 0
        self.prompt_queue = FakePromptQueue()

    def trigger_on_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload


class FakeExecution:
    def __init__(self, valid_result=(True, None, ["461"], {})) -> None:
        self._result = valid_result

    async def validate_prompt(self, prompt_id: str, prompt: dict[str, Any], partial: Any):
        return self._result


def test_save_cleanup_and_run_workflow_success(tmp_path: Path):
    folders = FakeFolderPaths(tmp_path)
    server = FakePromptServer()
    out_bytes = _png_bytes((48, 32))
    (folders.output_dir / "result_0001.png").write_bytes(out_bytes)

    def _complete(item):
        _, pid, prompt, _, _, _ = item
        server.prompt_queue.queued.clear()
        server.prompt_queue.history[pid] = {
            "prompt": item,
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {
                "461": {"images": [{"filename": "result_0001.png", "subfolder": "", "type": "output"}]},
                "472": {"a_images": [{"filename": "temp.png", "subfolder": "", "type": "temp"}]},
            },
        }

    server.prompt_queue.on_put = _complete
    backend = InProcessComfyBackend(
        server,
        execution_module=FakeExecution(),
        folder_paths_module=folders,
        poll_interval=0.01,
    )

    async def _run():
        rel = await backend.save_input_image(_png_bytes(), "abc.png", "openai_gateway")
        assert rel == "openai_gateway/abc.png"
        assert (folders.input_dir / rel).is_file()

        rendered = await backend.run_workflow(
            {"461": {"class_type": "SaveImageAdvanced", "inputs": {}}},
            output_node="461",
            timeout=5.0,
        )
        assert len(rendered) == 1
        assert rendered[0].data == out_bytes
        assert (rendered[0].width, rendered[0].height) == (48, 32)

        await backend.cleanup_input_files([rel])
        assert not (folders.input_dir / rel).exists()

    asyncio.run(_run())


def test_workflow_validation_error_and_execution_error(tmp_path: Path):
    folders = FakeFolderPaths(tmp_path)
    server = FakePromptServer()

    bad_exec = FakeExecution(
        valid_result=(False, {"message": "Invalid input", "details": "bad node"}, [], {"1": {"errors": ["boom"]}})
    )
    backend_bad = InProcessComfyBackend(
        server,
        execution_module=bad_exec,
        folder_paths_module=folders,
        poll_interval=0.01,
    )
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(backend_bad.run_workflow({"1": {"class_type": "X", "inputs": {}}}))
    assert exc_info.value.status == 400
    assert "Invalid input" in exc_info.value.message

    def _fail(item):
        _, pid, *_ = item
        server.prompt_queue.queued.clear()
        server.prompt_queue.history[pid] = {
            "prompt": item,
            "status": {
                "status_str": "error",
                "messages": [
                    ["execution_error", {"node_type": "UNETLoader", "exception_message": "model missing"}]
                ],
            },
            "outputs": {},
        }

    server.prompt_queue.on_put = _fail
    backend_err = InProcessComfyBackend(
        server,
        execution_module=FakeExecution(),
        folder_paths_module=folders,
        poll_interval=0.01,
    )
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(backend_err.run_workflow({"1": {"class_type": "X", "inputs": {}}}))
    assert exc_info.value.status == 502
    assert "UNETLoader" in exc_info.value.message
    assert "model missing" in exc_info.value.message
