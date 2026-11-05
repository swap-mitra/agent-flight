"""The webhook approval channel (FR-G4).

Outbound: each request is POSTed as JSON to the policy's ``[gate.webhook]``
url, signed with HMAC-SHA256, from a background thread (stdlib ``urllib``, no
dependency). A failed delivery is retried and then logged; the request stays
pending and expires, so an outage denies the call (the gate fails closed).

Inbound: ``fr.approvals.asgi_app(secret)`` is a dependency-free ASGI app the
user mounts in their own service. It takes a signed POST of
``{"id": ..., "decision": "approve" | "deny", "approver": ..., "reason": ...}``.

Both directions sign ``<timestamp>.<body>`` and send ``X-FR-Timestamp`` and
``X-FR-Signature: sha256=<hex>``; the app refuses a signature older than five
minutes, so a captured callback can't be replayed later.
"""

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from flight_recorder import approvals

logger = logging.getLogger("flight_recorder")

MAX_AGE = 300  # seconds a signature stays valid
_ATTEMPTS = 3


def sign(secret: str | bytes, timestamp: str, body: bytes) -> str:
    """The ``X-FR-Signature`` value for ``body`` sent at ``timestamp``."""
    key = secret.encode() if isinstance(secret, str) else secret
    digest = hmac.new(key, timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + digest.hexdigest()


def headers(secret: str | bytes, body: bytes) -> dict[str, str]:
    """The headers for a signed POST of ``body``."""
    timestamp = str(int(time.time()))
    return {
        "Content-Type": "application/json",
        "X-FR-Timestamp": timestamp,
        "X-FR-Signature": sign(secret, timestamp, body),
    }


def notify(
    request: approvals.Request, url: str, secret_env: str, timeout: float
) -> None:
    """POST ``request`` to ``url`` in a background thread."""
    thread = threading.Thread(
        target=_deliver,
        args=(request, url, secret_env, timeout),
        name="flight-recorder-webhook",
        daemon=True,
    )
    thread.start()


def _deliver(
    request: approvals.Request, url: str, secret_env: str, timeout: float
) -> None:
    body = json.dumps(request.to_json()).encode()
    secret = os.environ.get(secret_env, "")
    error: Exception = RuntimeError("not sent")
    for attempt in range(_ATTEMPTS):
        post = urllib.request.Request(
            url, data=body, headers=headers(secret, body), method="POST"
        )
        try:
            with urllib.request.urlopen(post, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.HTTPError as exc:
            exc.close()  # it holds the open response
            error = exc
        except Exception as exc:  # noqa: BLE001  retried, then logged
            error = exc
        else:
            error = RuntimeError(f"HTTP {response.status}")
        if attempt < _ATTEMPTS - 1:
            time.sleep(2**attempt)
    logger.warning(
        "approval %s: webhook delivery to %s failed (%s); it stays pending"
        " until decided another way or it expires",
        request.id,
        url,
        error,
    )


Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


def app(secret: str | bytes, *, dir: str | Path | None = None) -> Any:
    """The callback ASGI app; decisions go to the state store in ``dir``
    (default: FR_DIR, or .fr)."""
    if not secret:
        raise ValueError("the webhook callback needs its HMAC secret")
    root = Path(dir) if dir is not None else Path(os.environ.get("FR_DIR") or ".fr")
    store = root / "state.sqlite"

    async def application(scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            return
        if scope["method"] != "POST":
            await _reply(send, 405, {"error": "POST a decision"})
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        status, payload = _decide(secret, store, dict(scope["headers"]), body)
        await _reply(send, status, payload)

    return application


def _decide(
    secret: str | bytes, store: Path, raw: dict[bytes, bytes], body: bytes
) -> tuple[int, dict[str, Any]]:
    timestamp = raw.get(b"x-fr-timestamp", b"").decode()
    signature = raw.get(b"x-fr-signature", b"").decode()
    if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > MAX_AGE:
        return 401, {"error": "missing or stale X-FR-Timestamp"}
    if not hmac.compare_digest(signature, sign(secret, timestamp, body)):
        return 401, {"error": "bad signature"}
    try:
        data = json.loads(body)
        request_id, decision = str(data["id"]), data["decision"]
    except (ValueError, KeyError, TypeError):
        return 400, {"error": 'send {"id": ..., "decision": "approve" or "deny"}'}
    if decision not in ("approve", "deny"):
        return 400, {"error": "decision must be approve or deny"}
    if approvals.get(request_id, at=store) is None:
        return 404, {"error": f"no approval {request_id}"}
    resolved = approvals.resolve(
        request_id,
        decision,
        approver=str(data.get("approver") or "webhook"),
        reason=data.get("reason"),
        at=store,
    )
    if not resolved:
        return 409, {"error": "already decided or expired"}
    return 200, {"id": request_id, "decision": decision}


async def _reply(send: Send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
