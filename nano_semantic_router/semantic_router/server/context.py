from dataclasses import dataclass
from aiohttp import ClientSession


@dataclass
class RouterContext:
    upstream_base: str
    client: ClientSession
