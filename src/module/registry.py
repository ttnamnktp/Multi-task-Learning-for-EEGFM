MODULE_REGISTRY = {}

def register_module(name):
    def wrapper(cls):
        MODULE_REGISTRY[name] = cls
        return cls
    return wrapper


def get_module(name):
    if name not in MODULE_REGISTRY:
        raise ValueError(
            f"Unknown module {name}. "
            f"Available: {list(MODULE_REGISTRY.keys())}"
        )
    return MODULE_REGISTRY[name]