import torch.nn as nn

class BaseTask(nn.Module):

    def forward(self, shared_output, batch):
        raise NotImplementedError