from dataclasses import dataclass
from aiohttp import ClientSession, web


@dataclass
class RouterContext:
    upstream_base: str
    client: ClientSession
    original_request: web.Request | None = None
