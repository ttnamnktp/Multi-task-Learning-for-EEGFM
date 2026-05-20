import torch

def build_scheduler(cfg, optimizer):
    name = cfg.scheduler.get("name", None)

    if name is None:
        return None

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.trainer.max_epochs
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    raise ValueError(f"Unknown scheduler: {name}")