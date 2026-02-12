from dataclasses import dataclass
from typing import Any

from aiohttp import web
from multidict import CIMultiDict

from nano_semantic_router.semantic_router.server.context import RouterContext


@dataclass
class ProcessedRequest:
    method: str
    path_and_query: str
    headers: CIMultiDict[str]
    body: bytes


async def process(request: web.Request, ctx: RouterContext) -> ProcessedRequest:
    body = await request.read()

    # TODO: check headers, parse request metadata (request id) and classify.
    print(f"body ({len(body)} bytes): {body!r}")

    headers = CIMultiDict(request.headers)
    path_and_query = request.rel_url.human_repr()

    return ProcessedRequest(request.method, path_and_query, headers, body)


def parse_openai_request(body: bytes) -> None:
    # parse the request body as JSON and extract relevant fields for routing/classification
    # return an error if the body is not valid JSON or missing required fields
    _ = body


def translate_request(body: bytes) -> bytes:
    # translate a Response API request to Chat Completion format and return the translated body
    return body
