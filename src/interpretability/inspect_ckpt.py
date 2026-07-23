import torch

def inspect(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    def describe(obj, prefix="", depth=0, max_depth=4):
        if depth > max_depth:
            return
        if torch.is_tensor(obj):
            print(f"{prefix:50s} tensor  shape={tuple(obj.shape)}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                describe(v, f"{prefix}.{k}" if prefix else str(k), depth+1, max_depth)
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix:50s} {type(obj).__name__} len={len(obj)}")
            if len(obj) > 0:
                describe(obj[0], f"{prefix}[0]", depth+1, max_depth)
        else:
            print(f"{prefix:50s} {type(obj).__name__} = {str(obj)[:60]}")

    describe(ckpt)

inspect("/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/880534/2026-07-03/17-07-18/checkpoints/last.ckpt")