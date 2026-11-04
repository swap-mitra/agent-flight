"""Serve recorded model responses at the SDK's HTTP layer (FR-P1, FR-P10).

``fr.wrap`` puts a shim around the client's HTTP transport. While fully-recorded
replay serves a call, the shim answers with the recorded response, rebuilt as
the HTTP body the SDK parsed it from, so every way of calling the SDK (create,
parse, streams, stream helpers, raw responses) gets exactly what it got when
recorded, and no request leaves the process. Otherwise it passes through.
"""

import json
from typing import Any

import httpx2

from flight_recorder.replay.engine import served_response


def body(response: Any) -> httpx2.Response:
    """The HTTP response a recorded response or stream was parsed from."""
    if not isinstance(response, list):
        return httpx2.Response(200, json=response)
    chunks: list[Any] = response  # pyright: ignore[reportUnknownVariableType]
    # Chat Completions streams unnamed events; Responses and Anthropic name
    # each event after its type. The SDKs end a stream at the end of the body.
    named = any(isinstance(c, dict) and "type" in c for c in chunks)
    text = "".join(
        (f"event: {c['type']}\n" if named else "") + f"data: {json.dumps(c)}\n\n"
        for c in chunks
    )
    headers = {"content-type": "text/event-stream"}
    return httpx2.Response(200, headers=headers, content=text.encode())


class _Shim(httpx2.BaseTransport):
    def __init__(self, inner: httpx2.BaseTransport) -> None:
        self.inner = inner

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        served = served_response.get()
        return self.inner.handle_request(request) if served is None else body(served)

    def close(self) -> None:
        self.inner.close()


class _AsyncShim(httpx2.AsyncBaseTransport):
    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self.inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        served = served_response.get()
        if served is None:
            return await self.inner.handle_async_request(request)
        return body(served)

    async def aclose(self) -> None:
        await self.inner.aclose()


def install(http: Any) -> None:
    """Put the shim around an SDK client's httpx client transports, once.

    ponytail: httpx keeps transports in the private ``_transport`` and
    ``_mounts``; the replay contract tests fail loudly if that changes.
    """
    client: Any = http  # its transports are private attributes
    shim: Any = _AsyncShim if isinstance(http, httpx2.AsyncClient) else _Shim
    if not isinstance(client._transport, (_Shim, _AsyncShim)):
        client._transport = shim(client._transport)
    for pattern, transport in list(client._mounts.items()):
        if transport is not None and not isinstance(transport, (_Shim, _AsyncShim)):
            client._mounts[pattern] = shim(transport)
