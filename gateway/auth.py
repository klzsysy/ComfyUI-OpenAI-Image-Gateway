"""Bearer token authentication and CORS helpers for gateway routes."""

from __future__ import annotations

import hmac
from typing import Sequence

from aiohttp import web

from .models import GatewayError

_ALLOWED_METHODS = "GET, POST, OPTIONS"
_ALLOWED_HEADERS = "Authorization, Content-Type, X-API-Key, X-Requested-With, Accept, Origin"


def verify_api_key(request: web.Request, api_keys: Sequence[str]) -> None:
    """Verify Authorization Bearer token or x-api-key header when keys are configured."""
    if not api_keys:
        return

    token = ""
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
    if not token:
        token = request.headers.get("X-API-Key", "").strip()

    if not token or not any(hmac.compare_digest(token, expected) for expected in api_keys):
        raise GatewayError(
            "Invalid or missing API key.",
            status=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )


def _resolve_allow_origin(request: web.Request, cors_origins: str) -> str | None:
    spec = (cors_origins or "").strip()
    if not spec:
        return None
    if spec == "*":
        return "*"

    origin = request.headers.get("Origin", "").strip()
    allowed = [item.strip() for item in spec.split(",") if item.strip()]
    if origin and origin in allowed:
        return origin
    return None


def apply_cors_headers(
    request: web.Request,
    response: web.StreamResponse,
    cors_origins: str,
) -> web.StreamResponse:
    """Attach CORS and Private Network Access headers if not already present."""
    allow_origin = _resolve_allow_origin(request, cors_origins)
    if not allow_origin:
        return response

    response.headers.setdefault("Access-Control-Allow-Origin", allow_origin)
    response.headers.setdefault("Access-Control-Allow-Methods", _ALLOWED_METHODS)
    response.headers.setdefault("Access-Control-Allow-Headers", _ALLOWED_HEADERS)
    response.headers.setdefault("Access-Control-Max-Age", "86400")
    if allow_origin != "*":
        existing_vary = response.headers.get("Vary", "")
        if "Origin" not in [v.strip() for v in existing_vary.split(",") if v.strip()]:
            response.headers["Vary"] = f"{existing_vary}, Origin".lstrip(", ")
    if request.headers.get("Access-Control-Request-Private-Network", "").lower() == "true":
        response.headers.setdefault("Access-Control-Allow-Private-Network", "true")
    return response


def preflight_response(request: web.Request, cors_origins: str) -> web.Response:
    """Build a 204 response for CORS preflight OPTIONS requests."""
    resp = web.Response(status=204)
    apply_cors_headers(request, resp, cors_origins)
    return resp
