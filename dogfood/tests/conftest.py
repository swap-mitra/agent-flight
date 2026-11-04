"""Heptapod's mock backends, served in-process, and a scripted Claude.

Heptapod's mocks are not in its installed package, so this suite needs a
checkout of it: set HEPTAPOD_DIR (CI checks out a pinned commit). Each mock
runs on a free local port, as in Heptapod's own tests, so no Docker is needed.
The model is the real ChatAnthropic over the real Anthropic SDK; only the
HTTP transport is scripted, so nothing leaves the machine.
"""

import importlib
import json
import os
import sys
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
import uvicorn
from langchain_anthropic import ChatAnthropic

import flight_recorder as fr

# The replay suite's baselines live with this project, wherever pytest runs from.
if not os.environ.get("FR_DIR"):
    fr.configure(dir=Path(__file__).parents[1] / ".fr")

HEPTAPOD_DIR = os.environ.get("HEPTAPOD_DIR")
if not HEPTAPOD_DIR or not (Path(HEPTAPOD_DIR) / "core" / "registry.py").is_file():
    raise pytest.UsageError(
        "set HEPTAPOD_DIR to a checkout of https://github.com/swap-mitra/heptapod"
    )
sys.path.insert(0, HEPTAPOD_DIR)

# The mocks' throwaway fixture credentials, as in Heptapod's .env.example.
SECRETS = {
    "CRM_API_KEY": "crm-dev-key",
    "TICKETING_USER": "support",
    "TICKETING_PASSWORD": "ticketing-dev-pass",
    "INVENTORY_CLIENT_ID": "heptapod-agent",
    "INVENTORY_CLIENT_SECRET": "inventory-dev-secret",
}
MODEL = "claude-opus-5"


@contextmanager
def serve(app: Any) -> Generator[str]:
    """Run an ASGI app on a free local port in a thread; yields its base URL."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("mock server failed to start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A Heptapod Registry over fresh crm, ticketing and inventory mocks."""
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    apps = {
        name: importlib.import_module(f"mocks.{name}_service.app").create_app()
        for name in ("crm", "ticketing", "inventory")
    }
    adapters = {
        "crm": ("adapters.crm", "CrmAdapter"),
        "ticketing": ("adapters.ticketing", "TicketingAdapter"),
        "inventory": ("adapters.inventory", "InventoryAdapter"),
    }
    registry_cls = importlib.import_module("core.registry").Registry
    with (
        serve(apps["crm"]) as crm,
        serve(apps["ticketing"]) as tickets,
        serve(apps["inventory"]) as inventory,
    ):
        urls = {"crm": crm, "ticketing": tickets, "inventory": inventory}
        yield registry_cls(
            [
                getattr(importlib.import_module(module), cls)(base_url=urls[name])
                for name, (module, cls) in adapters.items()
            ]
        )


class ScriptedClaude:
    """Answers like the Messages API with each turn's content blocks, in order,
    and keeps every request body."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = turns
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        content = self.turns[len(self.bodies) - 1]
        calls = any(block["type"] == "tool_use" for block in content)
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{len(self.bodies)}",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": content,
                "stop_reason": "tool_use" if calls else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1200, "output_tokens": 80},
            },
        )

    def tool_results(self) -> list[dict[str, Any]]:
        """Every tool result the agent sent back, in order: the last request
        carries the whole conversation."""
        return [
            block
            for message in self.bodies[-1]["messages"]
            if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "tool_result"
        ]


def chat_model(script: ScriptedClaude) -> ChatAnthropic:
    """ChatAnthropic as an agent would build it, on the scripted transport."""
    # Heptapod's runtime caps a turn at 16,000 tokens; ChatAnthropic would
    # otherwise ask for the model's maximum, which needs streaming.
    settings: dict[str, Any] = {
        "model": MODEL,
        "api_key": "sk-ant-test",
        "max_tokens": 16000,
        "max_retries": 0,
    }
    model = ChatAnthropic(**settings)
    http = httpx2.Client(transport=httpx2.MockTransport(script))
    # The client ChatAnthropic would build itself, on the scripted transport.
    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)
    object.__setattr__(model, "_client", client)
    return model


def tool_use(call_id: str, name: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": input}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}
