"""Recorded model calls in the GenAI conventions' message schema (FR-R7).

Converts a call's provider-native JSON into the values of
``gen_ai.system_instructions``, ``gen_ai.input.messages`` and
``gen_ai.output.messages``, as defined by the JSON schemas at the conventions
commit pinned in ``otel.SEMCONV``. A streamed call is rebuilt from its chunks.
Content with no part type of its own (citations, server tool calls, refusals)
becomes a generic part: its provider type with its fields kept.

The input is whatever the agent sent and the provider returned, so every
lookup tolerates any JSON shape: conversion never raises, since an exception
would cut a run's export short.
"""

import json
from typing import Any, cast

from flight_recorder.model import ModelCall

Part = dict[str, Any]
Message = dict[str, Any]


def convert(call: ModelCall) -> tuple[list[Part], list[Message], list[Message]]:
    """System instructions, input messages and output messages of one call.

    Each is empty when the call has none: no separate system prompt, or no
    output because the call failed before any arrived.
    """
    messages: Any = call.messages
    response: Any = call.response
    if call.provider == "anthropic":
        system = _anthropic_system(call.params.get("system"))
        return system, _anthropic_input(messages), _anthropic_output(response)
    # OpenAI: Chat Completions and Responses share one reader, since their
    # inputs differ by item type and their outputs by top-level key.
    instructions = call.params.get("instructions")
    system = [_text(instructions)] if isinstance(instructions, str) else []
    return system, _openai_input(messages), _openai_output(response)


# Parts the schema defines.


def _text(content: Any) -> Part:
    return {"type": "text", "content": content if isinstance(content, str) else ""}


def _tool_call(call_id: Any, name: Any, arguments: Any) -> Part:
    if isinstance(arguments, str):  # OpenAI sends arguments as a JSON string
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass  # not JSON (a stream cut short): keep the string
    return {
        "type": "tool_call",
        "id": call_id,
        "name": _str(name),
        "arguments": arguments,
    }


def _tool_response(call_id: Any, response: Any) -> Part:
    return {"type": "tool_call_response", "id": call_id, "response": response}


def _media(modality: str, url: Any, mime_type: Any = None) -> Part:
    """A blob part for a data: URL, otherwise a uri part."""
    url = _str(url)
    if url.startswith("data:") and "," in url:
        header, data = url[5:].split(",", 1)
        mime = header.split(";")[0] or mime_type
        return {
            "type": "blob",
            "modality": modality,
            "mime_type": mime,
            "content": data,
        }
    return {"type": "uri", "modality": modality, "mime_type": mime_type, "uri": url}


def _file(modality: str, file_id: Any) -> Part:
    return {"type": "file", "modality": modality, "file_id": _str(file_id)}


def _generic(value: Any) -> Part:
    """The value as a generic part: its own type and fields, or type unknown."""
    if isinstance(value, dict):
        return {"type": "unknown", **_dict(value)}
    return {"type": "unknown", "value": value}


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _texts(blocks: Any, key: str = "text") -> str:
    return "".join(_str(_dict(b).get(key)) for b in _list(blocks))


def _add(messages: list[Message], role: str, part: Part) -> None:
    """Append part to the last message if it has this role, else a new one:
    a tool call or tool result item continues its turn."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["parts"].append(part)
    else:
        messages.append({"role": role, "parts": [part]})


# OpenAI


def _openai_input(messages: Any) -> list[Message]:
    if isinstance(messages, str):  # Responses input as a plain string
        return [{"role": "user", "parts": [_text(messages)]}]
    out: list[Message] = []
    for item in _list(messages):
        item = _dict(item)
        kind = item.get("type", "message")
        if kind == "message":
            role = _str(item.get("role"))
            role = "system" if role == "developer" else role
            if role == "tool":  # Chat: a tool result
                part = _tool_response(item.get("tool_call_id"), item.get("content"))
                _add(out, "tool", part)
                continue
            parts = _openai_content(item.get("content"))
            parts += _chat_tool_calls(item.get("tool_calls"))
            message: Message = {"role": role, "parts": parts}
            if isinstance(item.get("name"), str):
                message["name"] = item["name"]
            out.append(message)
        elif kind == "function_call":
            call = _tool_call(
                item.get("call_id"), item.get("name"), item.get("arguments")
            )
            _add(out, "assistant", call)
        elif kind == "function_call_output":
            _add(out, "tool", _tool_response(item.get("call_id"), item.get("output")))
        else:  # an output item sent back: reasoning, a server tool call
            for part in _responses_parts(item):
                _add(out, "assistant", part)
    return out


def _openai_content(content: Any) -> list[Part]:
    if isinstance(content, str):
        return [_text(content)]
    parts: list[Part] = []
    for raw in _list(content):
        part = _dict(raw)
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append(_text(part.get("text")))
        elif kind == "image_url":  # Chat
            parts.append(_media("image", _dict(part.get("image_url")).get("url")))
        elif kind == "input_image":  # Responses
            if isinstance(part.get("file_id"), str):
                parts.append(_file("image", part["file_id"]))
            else:
                parts.append(_media("image", part.get("image_url")))
        elif kind == "input_audio":
            audio = _dict(part.get("input_audio"))
            mime = f"audio/{_str(audio.get('format'))}"
            parts.append(
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": mime,
                    "content": _str(audio.get("data")),
                }
            )
        elif kind in ("file", "input_file"):
            file = _dict(part.get("file")) if kind == "file" else part
            if isinstance(file.get("file_id"), str):
                parts.append(_file("document", file["file_id"]))
            elif isinstance(file.get("file_url"), str):
                parts.append(_media("document", file["file_url"]))
            else:
                data = _str(file.get("file_data"))
                parts.append(
                    _media(
                        "document",
                        data if data.startswith("data:") else f"data:;base64,{data}",
                    )
                )
        else:
            parts.append(_generic(raw))
    return parts


def _chat_tool_calls(calls: Any) -> list[Part]:
    return [
        _tool_call(_dict(c).get("id"), function.get("name"), function.get("arguments"))
        for c in _list(calls)
        if (function := _dict(_dict(c).get("function")))
    ]


def _openai_output(response: Any) -> list[Message]:
    if isinstance(response, list):  # a stream's chunks or events
        response = _openai_stream(_list(response))
    response = _dict(response)
    if "choices" in response:  # Chat: one message per choice
        choices = sorted(
            _list(response["choices"]), key=lambda c: _dict(c).get("index") or 0
        )
        out = [_chat_message(_dict(_dict(c).get("message"))) for c in choices]
    else:  # Responses: the whole output is one generation
        items = _list(response.get("output"))
        parts = [p for i in items for p in _responses_parts(_dict(i))]
        out = [{"role": "assistant", "parts": parts}]
    return [m for m in out if m["parts"]]


def _chat_message(message: dict[str, Any]) -> Message:
    parts = _openai_content(message.get("content"))
    if isinstance(message.get("refusal"), str):
        parts.append({"type": "refusal", "content": message["refusal"]})
    parts += _chat_tool_calls(message.get("tool_calls"))
    return {"role": "assistant", "parts": parts}


def _responses_parts(item: dict[str, Any]) -> list[Part]:
    kind = item.get("type")
    if kind == "message":
        return _openai_content(item.get("content"))
    if kind == "function_call":
        call_id, name = item.get("call_id"), item.get("name")
        return [_tool_call(call_id, name, item.get("arguments"))]
    if kind == "reasoning":
        text = _texts(item.get("content")) or _texts(item.get("summary"))
        return [{"type": "reasoning", "content": text}]
    return [_generic(item)]


def _openai_stream(chunks: list[Any]) -> dict[str, Any]:
    events = [_dict(c) for c in chunks]
    if any("choices" in e for e in events):
        return _chat_stream(events)
    # Responses: a terminal event carries the whole response; a stream cut
    # short has only the items that finished.
    for event in reversed(events):
        if event.get("type") in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            return _dict(event.get("response"))
    done = [
        e.get("item") for e in events if e.get("type") == "response.output_item.done"
    ]
    return {"output": done}


def _chat_stream(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        for choice in _list(chunk.get("choices")):
            choice = _dict(choice)
            index = choice.get("index")
            index = index if isinstance(index, int) else 0
            message = choices.setdefault(index, {"content": "", "tool_calls": {}})
            delta = _dict(choice.get("delta"))
            message["content"] += _str(delta.get("content"))
            if isinstance(delta.get("refusal"), str):
                message["refusal"] = message.get("refusal", "") + delta["refusal"]
            for call in _list(delta.get("tool_calls")):
                call = _dict(call)
                slot = message["tool_calls"].setdefault(
                    call.get("index"),
                    {"id": None, "function": {"name": "", "arguments": ""}},
                )
                slot["id"] = call.get("id") or slot["id"]
                function = _dict(call.get("function"))
                slot["function"]["name"] += _str(function.get("name"))
                slot["function"]["arguments"] += _str(function.get("arguments"))
    return {
        "choices": [
            {
                "index": i,
                "message": {
                    **m,
                    "content": m["content"] or None,
                    "tool_calls": list(m["tool_calls"].values()),
                },
            }
            for i, m in choices.items()
        ]
    }


# Anthropic


def _anthropic_system(system: Any) -> list[Part]:
    if isinstance(system, str):
        return [_text(system)]
    return [_anthropic_part(b) for b in _list(system)]


def _anthropic_input(messages: Any) -> list[Message]:
    out: list[Message] = []
    for message in _list(messages):
        message = _dict(message)
        content = message.get("content")
        blocks = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else _list(content)
        )
        parts = [_anthropic_part(b) for b in blocks]
        role = _str(message.get("role"))
        # Tool results travel in a user message; the conventions give them
        # the tool role.
        if parts and all(p["type"] == "tool_call_response" for p in parts):
            role = "tool"
        out.append({"role": role, "parts": parts})
    return out


def _anthropic_part(raw: Any) -> Part:
    block = _dict(raw)
    kind = block.get("type")
    if kind == "text":
        return _text(block.get("text"))
    if kind == "tool_use":
        return _tool_call(block.get("id"), block.get("name"), block.get("input"))
    if kind == "tool_result":
        return _tool_response(block.get("tool_use_id"), block.get("content"))
    if kind == "thinking":
        return {"type": "reasoning", "content": _str(block.get("thinking"))}
    if kind in ("image", "document"):
        modality = "image" if kind == "image" else "document"
        source = _dict(block.get("source"))
        if source.get("type") == "base64":
            return {
                "type": "blob",
                "modality": modality,
                "mime_type": source.get("media_type"),
                "content": _str(source.get("data")),
            }
        if source.get("type") == "url":
            return _media(modality, source.get("url"))
        if source.get("type") == "file":
            return _file(modality, source.get("file_id"))
    return _generic(raw)


def _anthropic_output(response: Any) -> list[Message]:
    if isinstance(response, list):  # a stream's events
        response = _anthropic_stream(_list(response))
    parts = [_anthropic_part(b) for b in _list(_dict(response).get("content"))]
    return [{"role": "assistant", "parts": parts}] if parts else []


def _anthropic_stream(events: list[Any]) -> dict[str, Any]:
    blocks: dict[Any, dict[str, Any]] = {}
    for event in events:
        event = _dict(event)
        kind = event.get("type")
        if kind == "content_block_start":
            blocks[event.get("index")] = dict(_dict(event.get("content_block")))
        elif kind == "content_block_delta" and event.get("index") in blocks:
            block, delta = blocks[event.get("index")], _dict(event.get("delta"))
            for field, key in (
                ("text", "text"),
                ("thinking", "thinking"),
                ("partial_json", "input"),  # JSON text, parsed by _tool_call
            ):
                if isinstance(delta.get(field), str):
                    block[key] = _str(block.get(key)) + delta[field]
    return {"content": list(blocks.values())}
