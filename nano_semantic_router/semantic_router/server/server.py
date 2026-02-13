import asyncio
from dataclasses import dataclass
from typing import Optional

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from multidict import CIMultiDict
from yarl import URL

import logging
from nano_semantic_router.semantic_router.server.context import RouterContext
from nano_semantic_router.semantic_router.server.process import (
    ProcessedRequest,
    process,
)
from nano_semantic_router.semantic_router.server.router import Router


@dataclass
class Config:
    upstream_base: str = "http://example.com:80"
    port: int = 8080
    secure: bool = False
    request_timeout: float = 30.0


class Server:
    def __init__(
        self, config: Optional[Config] = None, router: Optional[Router] = None
    ) -> None:
        self.config = config or Config()
        self.router = router or Router()
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[ClientSession] = None

    async def start(self) -> None:
        connector = TCPConnector(ssl=self.config.secure)
        timeout = ClientTimeout(total=self.config.request_timeout)
        self._session = ClientSession(connector=connector, timeout=timeout)
        ctx = RouterContext(self.config.upstream_base, self._session)

        app = web.Application()
        app["ctx"] = ctx
        app.router.add_route("*", "/{tail:.*}", self._handle_request)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        site = web.TCPSite(
            self._runner,
            host="0.0.0.0",
            port=self.config.port,
            ssl_context=None,
        )

        logging.info(
            f"Starting server on http://0.0.0.0:{self.config.port} "
            f"(secure: {self.config.secure}) -> upstream {self.config.upstream_base}"
        )

        await site.start()
        logging.info("Server started successfully.")
        try:
            await asyncio.Event().wait()
        finally:
            await self.close()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _handle_request(self, request: web.Request) -> web.Response:
        ctx: RouterContext = request.app["ctx"]
        try:
            # process function may modify the request.
            ctx.original_request = request.clone()
            processed = await process(request, self.router.config, ctx)
        except Exception as err:  # noqa: BLE001
            logging.error(f"processing error: {err}")
            return web.Response(status=500, text="Bad Gateway")

        try:
            return await self.proxy_to_upstream(processed, ctx)
        except Exception as err:  # noqa: BLE001
            print(f"router proxy error: {err}")
            return web.Response(status=502, text="Bad Gateway")

    async def proxy_to_upstream(
        self, processed: ProcessedRequest, ctx: RouterContext
    ) -> web.Response:
        target_base = URL(ctx.upstream_base)
        target = target_base.join(URL(processed.path_and_query))

        headers = CIMultiDict(processed.headers)
        if target.host:
            authority = target.host
            if target.port:
                authority = f"{authority}:{target.port}"
            headers["Host"] = authority

        assert self._session is not None, "Client session should be initialized"

        async with ctx.client.request(
            processed.method,
            target,
            data=processed.body,
            headers=headers,
        ) as upstream_resp:
            body = await upstream_resp.read()
            response_headers = CIMultiDict(upstream_resp.headers)
            return web.Response(
                status=upstream_resp.status,
                headers=response_headers,
                body=body,
            )
