"""Interceptors. Provider modules are imported only for a client of that SDK."""

from typing import TypeVar

C = TypeVar("C")


def wrap(client: C) -> C:
    """Record the model calls ``client`` makes inside ``fr.run()``.

    Patches this client instance only and returns it, so its type and every
    other method are unchanged. Wrapping twice records each call once.
    """
    sdk = type(client).__module__.split(".")[0]
    if sdk == "openai":
        from openai import AsyncOpenAI, OpenAI

        from flight_recorder.intercept import openai

        if isinstance(client, OpenAI | AsyncOpenAI):
            openai.wrap(client)
            return client
    if sdk == "anthropic":
        from anthropic import Anthropic, AsyncAnthropic

        from flight_recorder.intercept import anthropic

        if isinstance(client, Anthropic | AsyncAnthropic):
            anthropic.wrap(client)
            return client
    raise TypeError(f"fr.wrap does not support {type(client).__qualname__}")
