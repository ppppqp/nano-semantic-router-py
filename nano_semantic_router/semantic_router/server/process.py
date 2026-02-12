import json
from dataclasses import dataclass
from typing import Any, Union, cast

from aiohttp import web
from multidict import CIMultiDict
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
)

from nano_semantic_router.semantic_router.server.context import RouterContext


@dataclass
class ProcessedRequest:
    method: str
    path_and_query: str
    headers: CIMultiDict[str]
    body: bytes


ParsedOpenAIRequest = Union[
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
]


async def process(request: web.Request, ctx: RouterContext) -> ProcessedRequest:
    body = await request.read()

    # TODO: check headers, parse request metadata (request id) and classify.
    print(f"body ({len(body)} bytes): {body!r}")

    headers = CIMultiDict(request.headers)
    path_and_query = request.rel_url.human_repr()

    return ProcessedRequest(request.method, path_and_query, headers, body)


def parse_openai_request(body: bytes) -> ParsedOpenAIRequest:
    """Parse an incoming OpenAI request into the typed SDK structures."""

    try:
        raw_json = body.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("Request body must be valid UTF-8") from exc

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    if "messages" in payload:
        _require_keys(payload, ["model", "messages"])
        messages = payload["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        if payload.get("stream"):
            return cast(CompletionCreateParamsStreaming, payload)
        return cast(CompletionCreateParamsNonStreaming, payload)

    if "input" in payload:
        _require_keys(payload, ["model", "input"])
        if payload.get("stream"):
            return cast(ResponseCreateParamsStreaming, payload)
        return cast(ResponseCreateParamsNonStreaming, payload)

    raise ValueError(
        "Unsupported OpenAI payload; expected chat completions or responses request"
    )


def translate_request(body: bytes) -> bytes:
    # translate a Response API request to Chat Completion format and return the translated body
    return body


def _require_keys(payload: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
