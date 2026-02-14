from nano_semantic_router.config.config import Model, RouterConfig


def get_model_by_ref(model_ref: str, router_config: RouterConfig) -> Model:
    model = router_config.models.get(model_ref)
    if not model:
        raise ValueError(f"Model '{model_ref}' not found in router configuration")
    return model
