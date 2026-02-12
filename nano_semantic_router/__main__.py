"""Enable `python -m nano_semantic_router` to start the server."""

import asyncio

from nano_semantic_router.semantic_router.server.server import Server


def main() -> None:
    asyncio.run(Server().start())


if __name__ == "__main__":
    main()
