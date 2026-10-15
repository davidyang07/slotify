"""HTTP helpers shared by the pipeline's calls back to the Node API.

Every request the pipeline makes to the API (TTS, merge, clone) goes through
:func:`send`, so they all carry the same internal credential and timeout:

``SLOTIFY_INTERNAL_TOKEN``
    Set by the Node API in the environment of the pipelines that call back
    (``cli``, ``insert_ad``). Sent as ``Authorization: Bearer <token>``.

``SLOTIFY_API_TOKEN``
    Fallback credential when the internal token is absent (for example when
    the CLI is run by hand against a deployment that sets one). If neither is
    set, no ``Authorization`` header is sent.

``SLOTIFY_CALLBACK_TIMEOUT_S``
    Per-request timeout in seconds (default 300).

Requests to third parties (the ElevenLabs voice cleanup) pass
``internal_auth=False`` so the internal credential never leaves the deployment.

This module is deliberately not named ``http``: a sibling ``http.py`` would
shadow the standard library package that :mod:`urllib.request` imports
whenever this directory lands on ``sys.path``.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_CALLBACK_TIMEOUT_S = 300.0


def callback_timeout_s() -> float:
    """Timeout for API callbacks, from ``SLOTIFY_CALLBACK_TIMEOUT_S``."""
    raw = os.getenv("SLOTIFY_CALLBACK_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_CALLBACK_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CALLBACK_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_CALLBACK_TIMEOUT_S
    return value


def auth_headers() -> Dict[str, str]:
    """``Authorization`` header for API callbacks, or ``{}`` if no token is set."""
    token = (
        os.getenv("SLOTIFY_INTERNAL_TOKEN", "").strip()
        or os.getenv("SLOTIFY_API_TOKEN", "").strip()
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def encode_multipart(
    fields: Mapping[str, str],
    files: List[Dict[str, str | bytes]],
) -> tuple[str, bytes]:
    """Encode ``fields`` and ``files`` as ``multipart/form-data``."""
    boundary = uuid.uuid4().hex
    body = bytearray()

    def _add_line(line: str) -> None:
        body.extend(line.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in fields.items():
        _add_line(f"--{boundary}")
        _add_line(f'Content-Disposition: form-data; name="{name}"')
        _add_line("")
        _add_line(value)

    for file_info in files:
        name = str(file_info["name"])
        filename = str(file_info["filename"])
        content_type = str(file_info["content_type"])
        data = bytes(file_info["data"])

        _add_line(f"--{boundary}")
        _add_line(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
        )
        _add_line(f"Content-Type: {content_type}")
        _add_line("")
        body.extend(data)
        body.extend(b"\r\n")

    _add_line(f"--{boundary}--")
    return boundary, bytes(body)


def send(
    url: str,
    label: str,
    data: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    method: str = "POST",
    internal_auth: bool = True,
    timeout_s: Optional[float] = None,
) -> bytes:
    """Send one request and return the response body.

    Raises ``RuntimeError`` (prefixed with ``label``) on a non-2xx status, a
    connection failure or a timeout.
    """
    request_headers: Dict[str, str] = {}
    if internal_auth:
        request_headers.update(auth_headers())
    if headers:
        request_headers.update(headers)
    timeout = callback_timeout_s() if timeout_s is None else timeout_s

    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"{label} request failed with status {response.status}."
                )
            return response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"{label} request failed with status {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"{label} request failed: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError(
            f"{label} request timed out after {timeout:g}s."
        ) from error


def post_json(url: str, payload: Any, label: str) -> bytes:
    """POST ``payload`` as JSON to an API callback."""
    return send(
        url,
        label,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def post_multipart(
    url: str,
    fields: Mapping[str, str],
    files: List[Dict[str, str | bytes]],
    label: str,
) -> bytes:
    """POST a ``multipart/form-data`` body to an API callback."""
    boundary, body = encode_multipart(fields, files)
    return send(
        url,
        label,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
