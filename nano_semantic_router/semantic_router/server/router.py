from nano_semantic_router.config.config import RouterConfig


class Classifier:
    def __init__(self) -> None:
        pass


class CacheBackend:
    def __init__(self) -> None:
        pass


class Router:
    def __init__(self) -> None:
        self.config = RouterConfig()
        self.classifier = Classifier()
        self.cache = CacheBackend()
