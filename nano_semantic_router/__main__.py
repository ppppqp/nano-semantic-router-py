"""Enable `python -m nano_semantic_router` to start the server."""

import asyncio
import logging
import sys

from nano_semantic_router.semantic_router.server.server import Server


def configure_logging() -> None:
    """Send INFO+ logs to stdout with a simple format."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    asyncio.run(Server().start())


if __name__ == "__main__":
    main()
