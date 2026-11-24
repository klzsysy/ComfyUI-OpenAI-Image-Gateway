"""Image decoding, validation, format conversion, and size parsing."""

from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp
from PIL import Image, UnidentifiedImageError

from .models import DecodedImage, GatewayError

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;charset=[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)
_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")

_PASSTHROUGH_FORMATS = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
}

_MIN_DIMENSION = 64
_MAX_DIMENSION = 8192


def parse_size(size_str: str | None, *, multiple: int = 16) -> tuple[int, int] | None:
    """Parse an OpenAI size string (`auto` or `WxH`) into aligned (width, height)."""
    if size_str is None:
        return None
    cleaned = str(size_str).strip().lower()
    if not cleaned or cleaned == "auto":
        return None

    match = _SIZE_RE.match(cleaned)
    if not match:
        raise GatewayError(
            f"Invalid `size` value `{size_str}`. Expected `auto` or `WIDTHxHEIGHT` (e.g. `1024x1024`).",
            status=400,
            param="size",
            code="invalid_size",
        )
    width = int(match.group(1))
    height = int(match.group(2))
    if not (_MIN_DIMENSION <= width <= _MAX_DIMENSION and _MIN_DIMENSION <= height <= _MAX_DIMENSION):
        raise GatewayError(
            f"Requested size `{width}x{height}` is out of supported range ({_MIN_DIMENSION}-{_MAX_DIMENSION}).",
            status=400,
            param="size",
            code="invalid_size",
        )
    step = max(1, int(multiple))
    if step > 1:
        width = max(step, round(width / step) * step)
        height = max(step, round(height / step) * step)
    return width, height


def decode_image_bytes(
    raw: bytes,
    *,
    max_bytes: int,
    param: str = "image",
) -> DecodedImage:
    """Validate raw image bytes with Pillow and normalize uncommon formats to PNG."""
    if not raw:
        raise GatewayError("Uploaded image is empty.", status=400, param=param, code="invalid_image")
    if len(raw) > max_bytes:
        raise GatewayError(
            f"Image exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MiB.",
            status=413,
            param=param,
            code="image_too_large",
        )

    try:
        with Image.open(io.BytesIO(raw)) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
            img.load()
            if width <= 0 or height <= 0:
                raise GatewayError("Image has invalid dimensions.", status=400, param=param, code="invalid_image")

            is_animated = bool(getattr(img, "is_animated", False)) and getattr(img, "n_frames", 1) > 1
            if fmt in _PASSTHROUGH_FORMATS and not is_animated and img.mode in ("RGB", "RGBA", "L", "LA", "P"):
                ext, mime = _PASSTHROUGH_FORMATS[fmt]
                return DecodedImage(data=raw, ext=ext, mime=mime, width=width, height=height)

            normalized = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            buf = io.BytesIO()
            normalized.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            return DecodedImage(data=png_bytes, ext="png", mime="image/png", width=width, height=height)
    except GatewayError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise GatewayError(
            f"Failed to decode image: {exc}",
            status=400,
            param=param,
            code="invalid_image",
        ) from exc


def _decode_base64_payload(text: str, *, param: str) -> bytes:
    stripped = text.strip()
    match = _DATA_URL_RE.match(stripped)
    if match:
        if not match.group("b64"):
            raise GatewayError(
                "Only base64-encoded data URLs are supported for images.",
                status=400,
                param=param,
                code="invalid_image",
            )
        payload = match.group("data")
    else:
        payload = stripped

    compact = "".join(payload.split())
    if not compact:
        raise GatewayError("Empty base64 image string.", status=400, param=param, code="invalid_image")
    padding = (-len(compact)) % 4
    try:
        return base64.b64decode(compact + ("=" * padding), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GatewayError(
            f"Invalid base64 image payload: {exc}",
            status=400,
            param=param,
            code="invalid_image",
        ) from exc


async def _fetch_remote_image(
    url: str,
    *,
    max_bytes: int,
    timeout: float,
    param: str,
    session: aiohttp.ClientSession | None = None,
) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise GatewayError(
            f"Unsupported image URL scheme: `{url}`.",
            status=400,
            param=param,
            code="invalid_image_url",
        )

    own_session = session is None
    client = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
    try:
        async with client.get(url) as resp:
            if resp.status >= 400:
                raise GatewayError(
                    f"Failed to download image URL (HTTP {resp.status}).",
                    status=400,
                    param=param,
                    code="invalid_image_url",
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.content.iter_chunked(65536):
                total += len(chunk)
                if total > max_bytes:
                    raise GatewayError(
                        f"Remote image exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MiB.",
                        status=413,
                        param=param,
                        code="image_too_large",
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except GatewayError:
        raise
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise GatewayError(
            f"Failed to download remote image URL: {exc}",
            status=400,
            param=param,
            code="invalid_image_url",
        ) from exc
    finally:
        if own_session:
            await client.close()


def _unwrap_image_object(source: Any, *, param: str) -> Any:
    """Unwrap dict wrappers like {url: ...}, {b64_json: ...}, or {image_url: {url: ...}}."""
    if not isinstance(source, dict):
        return source
    if "b64_json" in source and isinstance(source["b64_json"], str):
        return source["b64_json"]
    if "base64" in source and isinstance(source["base64"], str):
        return source["base64"]
    if "data" in source and isinstance(source["data"], str):
        return source["data"]
    if "url" in source and isinstance(source["url"], str):
        return source["url"]
    if "image_url" in source:
        nested = source["image_url"]
        if isinstance(nested, str):
            return nested
        if isinstance(nested, dict) and isinstance(nested.get("url"), str):
            return nested["url"]
    raise GatewayError(
        "Unsupported image object format. Provide binary upload, data URL, base64 string, or HTTP(S) URL.",
        status=400,
        param=param,
        code="invalid_image",
    )


async def decode_image_source(
    source: Any,
    *,
    max_bytes: int,
    allow_remote_urls: bool = True,
    remote_timeout: float = 30.0,
    param: str = "image",
    session: aiohttp.ClientSession | None = None,
) -> DecodedImage:
    """Decode an image from raw bytes, data URL, bare base64 string, HTTP(S) URL, or dict wrapper."""
    unwrapped = _unwrap_image_object(source, param=param)
    if isinstance(unwrapped, (bytes, bytearray)):
        return decode_image_bytes(bytes(unwrapped), max_bytes=max_bytes, param=param)

    if isinstance(unwrapped, str):
        text = unwrapped.strip()
        if not text:
            raise GatewayError("Image value cannot be empty.", status=400, param=param, code="invalid_image")
        if text.startswith(("http://", "https://")):
            if not allow_remote_urls:
                raise GatewayError(
                    "Remote image URLs are disabled on this gateway. Upload image bytes or base64 instead.",
                    status=400,
                    param=param,
                    code="remote_urls_disabled",
                )
            raw = await _fetch_remote_image(
                text,
                max_bytes=max_bytes,
                timeout=remote_timeout,
                param=param,
                session=session,
            )
            return decode_image_bytes(raw, max_bytes=max_bytes, param=param)
        raw = _decode_base64_payload(text, param=param)
        return decode_image_bytes(raw, max_bytes=max_bytes, param=param)

    raise GatewayError(
        f"Unsupported image input type `{type(unwrapped).__name__}`.",
        status=400,
        param=param,
        code="invalid_image",
    )


def convert_output_image(
    raw: bytes,
    *,
    output_format: str | None = None,
    compression: int | None = None,
) -> tuple[bytes, str, int, int]:
    """Inspect and optionally re-encode output image bytes to png/jpeg/webp."""
    requested = (output_format or "").strip().lower()
    if requested == "jpg":
        requested = "jpeg"
    if requested and requested not in ("png", "jpeg", "webp"):
        raise GatewayError(
            f"Unsupported `output_format` `{output_format}`. Expected `png`, `jpeg`, or `webp`.",
            status=400,
            param="output_format",
            code="invalid_output_format",
        )

    with Image.open(io.BytesIO(raw)) as img:
        orig_fmt = (img.format or "PNG").lower()
        width, height = img.size
        target_fmt = requested or ("jpeg" if orig_fmt == "jpg" else orig_fmt)
        if target_fmt == orig_fmt and compression is None:
            return raw, target_fmt, width, height

        buf = io.BytesIO()
        quality = 90 if compression is None else max(1, min(100, int(compression)))
        if target_fmt == "jpeg":
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                out_img = bg
            else:
                out_img = img.convert("RGB")
            out_img.save(buf, format="JPEG", quality=quality)
        elif target_fmt == "webp":
            out_img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            out_img.save(buf, format="WEBP", quality=quality)
        else:
            out_img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            out_img.save(buf, format="PNG")
            target_fmt = "png"
        return buf.getvalue(), target_fmt, width, height
