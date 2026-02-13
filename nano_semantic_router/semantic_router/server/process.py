import json
from dataclasses import dataclass
from typing import Any, TypeGuard, Union, cast

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
from nano_semantic_router.config.config import RouterConfig
import logging
from nano_semantic_router.semantic_router.signal.signal import (
    get_signals_from_content,
)
from nano_semantic_router.semantic_router.server.context import RouterContext
from nano_semantic_router.semantic_router.decision.decision import make_routing_decision


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


async def process(
    request: web.Request, router_config: RouterConfig, ctx: RouterContext
) -> ProcessedRequest:
    body = await request.read()
    # TODO: check headers, parse request metadata (request id) and classify.
    print(f"body ({len(body)} bytes): {body!r}")

    headers = CIMultiDict(request.headers)
    path_and_query = request.rel_url.human_repr()
    parsed_request = parse_openai_request(body)
    user_content, _ = extract_user_content(parsed_request)
    if user_content == "":
        logging.warning(
            "No user content extracted from request; routing may be inaccurate. "
        )

    # use user_content to do routing.
    signals = get_signals_from_content(
        active_signals=router_config.signals,
        user_content=user_content,
    )
    decision = make_routing_decision(signals, router_config.decisions)
    if not decision:
        logging.warning(
            "No routing decision matched for request; Using default route. "
        )
        # TODO: implement default route
    else:
        logging.info(
            f"Routing decision: {decision.decision.name} (confidence: {decision.confidence:.2f}, matched_rules: {decision.matched_rules}) -> target model: {decision.decision.model_ref.model}"
        )
        # TODO: correctly handle rerouting by modifying the request
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


def _is_chat_completion_request(
    req: ParsedOpenAIRequest,
) -> TypeGuard[CompletionCreateParamsNonStreaming | CompletionCreateParamsStreaming]:
    return isinstance(req, dict) and "messages" in req


def _is_response_request(
    req: ParsedOpenAIRequest,
) -> TypeGuard[ResponseCreateParamsNonStreaming | ResponseCreateParamsStreaming]:
    return isinstance(req, dict) and "input" in req


def extract_user_content(req: ParsedOpenAIRequest) -> tuple[str, list[str]]:
    """Extract user content from the request for routing purposes."""
    if _is_chat_completion_request(req):
        user_content = ""
        non_user_contents = []
        for msg in req["messages"]:
            text_content = ""
            content = msg.get("content", "")
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item["type"] == "text":
                        parts.append(item.get("text", ""))
                text_content = " ".join(parts)
            role = msg.get("role", "")
            if role == "user":
                # only consider the last user message for routing
                user_content = text_content
            else:
                # for assistant or system messages, we can optionally include them in non_user_content for more advanced routing logic
                non_user_contents.append(text_content)
        return user_content, non_user_contents

    if _is_response_request(req):
        if "input" not in req:
            return "", []
        input_data = req["input"]
        if isinstance(input_data, str):
            return input_data, []

        if isinstance(input_data, list):
            user_parts: list[str] = []
            non_user_contents: list[str] = []
            for item in input_data:
                if isinstance(item, str):
                    user_parts.append(item)
                else:
                    non_user_contents.append(str(item))
            return "\n".join(user_parts), non_user_contents

    return "", []
