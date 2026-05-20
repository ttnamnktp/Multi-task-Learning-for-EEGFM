import torch

def build_optimizer(cfg, params):

    name = cfg.optimizer.get("name", "adamw")
    lr = cfg.optimizer.get("lr", 1e-3)
    wd = cfg.optimizer.get("weight_decay", 0.0)

    # nếu là iterator parameters()
    if not isinstance(params, (list, tuple)):
        params = [{"params": params, "lr": lr}]

    if name == "adam":
        return torch.optim.Adam(params)

    if name == "adamw":
        return torch.optim.AdamW(params, weight_decay=wd)

    if name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {name}")