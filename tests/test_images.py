"""Tests for image decoding, validation, format conversion, and size parsing."""

from __future__ import annotations

import asyncio
import base64
import io

import pytest
from PIL import Image

from gateway.images import convert_output_image, decode_image_bytes, decode_image_source, parse_size
from gateway.models import GatewayError


def _make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 24), mode: str = "RGB", color=(120, 40, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


def test_parse_size():
    assert parse_size(None) is None
    assert parse_size("") is None
    assert parse_size("auto") is None
    assert parse_size("1024x1536", multiple=32) == (1024, 1536)
    assert parse_size("1000X750", multiple=32) == (992, 736)

    with pytest.raises(GatewayError) as exc_info:
        parse_size("not-a-size")
    assert exc_info.value.status == 400

    with pytest.raises(GatewayError) as exc_info:
        parse_size("16x16")
    assert exc_info.value.status == 400


def test_decode_image_bytes_passthrough_and_normalization():
    png = _make_image_bytes("PNG", (40, 30))
    dec = decode_image_bytes(png, max_bytes=1024 * 1024)
    assert dec.ext == "png"
    assert dec.width == 40 and dec.height == 30
    assert dec.data == png

    bmp = _make_image_bytes("BMP", (20, 16))
    dec_bmp = decode_image_bytes(bmp, max_bytes=1024 * 1024)
    assert dec_bmp.ext == "png"
    assert dec_bmp.width == 20 and dec_bmp.height == 16

    with pytest.raises(GatewayError) as exc_info:
        decode_image_bytes(b"not an image", max_bytes=1024)
    assert exc_info.value.status == 400

    with pytest.raises(GatewayError) as exc_info:
        decode_image_bytes(png, max_bytes=10)
    assert exc_info.value.status == 413


def test_decode_image_source_variants():
    png = _make_image_bytes("PNG", (16, 16))
    b64 = base64.b64encode(png).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    async def _run():
        d1 = await decode_image_source(data_url, max_bytes=1024 * 1024)
        d2 = await decode_image_source(b64, max_bytes=1024 * 1024)
        d3 = await decode_image_source({"b64_json": b64}, max_bytes=1024 * 1024)
        d4 = await decode_image_source({"image_url": {"url": data_url}}, max_bytes=1024 * 1024)
        return d1, d2, d3, d4

    for dec in asyncio.run(_run()):
        assert dec.ext == "png"
        assert dec.width == 16 and dec.height == 16

    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(decode_image_source("https://example.com/a.png", max_bytes=1024, allow_remote_urls=False))
    assert exc_info.value.code == "remote_urls_disabled"


def test_convert_output_image_rgba_to_jpeg_and_webp():
    rgba_png = _make_image_bytes("PNG", (32, 24), mode="RGBA", color=(255, 0, 0, 128))
    jpg_bytes, fmt, w, h = convert_output_image(rgba_png, output_format="jpg", compression=85)
    assert fmt == "jpeg"
    assert (w, h) == (32, 24)
    with Image.open(io.BytesIO(jpg_bytes)) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"

    webp_bytes, fmt2, w2, h2 = convert_output_image(rgba_png, output_format="webp", compression=80)
    assert fmt2 == "webp"
    assert (w2, h2) == (32, 24)
    with Image.open(io.BytesIO(webp_bytes)) as img:
        assert img.format == "WEBP"
