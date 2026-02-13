from nano_semantic_router.config.config import ModelRef, RouterConfig


class Classifier:
    def __init__(self) -> None:
        pass


class CacheBackend:
    def __init__(self) -> None:
        pass


class Router:
    def __init__(self, config: RouterConfig | None = None) -> None:
        if config is None:
            config = RouterConfig(
                default_model=ModelRef(
                    model="gpt-4o-mini",
                    endpoint="https://api.openai.com",
                    access_key="",
                    model_type="openai",
                )
            )

        self.config = config
        self.classifier = Classifier()
        self.cache = CacheBackend()
