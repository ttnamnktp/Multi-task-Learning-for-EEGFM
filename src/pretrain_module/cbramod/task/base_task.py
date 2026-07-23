import torch.nn as nn
from typing import Any, Dict, Optional

class BaseTask(nn.Module):

    def build_task_context(
        self,
        batch,
        shared_ctx,
        module=None,
    ) -> Dict[str, Any]:
        return {}

    def forward(
        self,
        shared_output,
        ctx,
    ):
        raise NotImplementedError