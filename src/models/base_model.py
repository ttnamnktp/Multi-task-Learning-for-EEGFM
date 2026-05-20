import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_param_groups(self):
        """
        Default:
        all params use one LR
        """
        return self.parameters()