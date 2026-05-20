# src/analysis/gradient_monitor.py

import torch

class GradientMonitor:

    def __init__(self, pl_module):
        self.model = pl_module.model
        self.pl_module = pl_module

    def collect_modules(self):
        modules = {}

        for i, blk in enumerate(self.model.encoder.blocks):
            modules[f"enc.block.{i}"] = blk.attn
            modules[f"enc.mlp.{i}"] = blk.mlp

        return modules

    def get_grads(self, loss, modules):
        params = []
        names = []

        for name, m in modules.items():
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(name)

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True
        )

        out = {}
        for name, g in zip(names, grads):
            if g is None:
                continue
            out.setdefault(name, []).append(g.detach().flatten())

        return {k: torch.cat(v) for k, v in out.items()}

    def log_conflict(self, loss_a, loss_b, step=True):
        modules = self.collect_modules()

        ga = self.get_grads(loss_a, modules)
        gb = self.get_grads(loss_b, modules)

        for name in modules:
            if name not in ga or name not in gb:
                continue

            cos = torch.dot(ga[name], gb[name]) / (
                torch.norm(ga[name]) * torch.norm(gb[name]) + 1e-8
            )

            self.pl_module.log(
                f"conflict/{name}/cos",
                cos,
                on_step=True
            )