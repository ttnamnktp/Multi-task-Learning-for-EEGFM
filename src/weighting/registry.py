# src/weighting/registry.py

_WEIGHTING_REGISTRY = {}


def register_weighting(name):
    def decorator(cls):
        _WEIGHTING_REGISTRY[name] = cls
        return cls
    return decorator


def get_weighting(name):
    if name not in _WEIGHTING_REGISTRY:
        raise ValueError(f"Unknown weighting: {name}")
    return _WEIGHTING_REGISTRY[name]