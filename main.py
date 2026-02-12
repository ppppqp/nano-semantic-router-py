"""Entry point mirroring the Rust binary in main.rs."""

import asyncio

from nano_semantic_router.semantic_router.server.server import Server


def main() -> None:
    asyncio.run(Server().start())


if __name__ == "__main__":
    main()
