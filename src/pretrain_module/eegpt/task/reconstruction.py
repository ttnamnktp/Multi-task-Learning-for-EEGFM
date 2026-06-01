import torch.nn as nn

from .base_task import BaseTask
from src.models.eegpt.eegpt import *

class ReconstructionTask(BaseTask):

    def __init__(self, online_encoder, models_configs):
        super().__init__()

        self.online_encoder = online_encoder
        self.models_configs = models_configs

        self.predictor = EEGTransformerPredictor(
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                **models_configs['predictor'])

        self.reconstructor = EEGTransformerReconstructor(
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                **models_configs['reconstructor'])

        self.loss_fn = nn.MSELoss()

    def forward(self, shared_output, x, x_aug, mask_x, mask_y):
        out_pred_fake, out_pred_comb = self.predictor(shared_output, mask_x=mask_x)
        out_rec = self.reconstructor(out_pred_comb, self.online_encoder.chans_id.to(x_aug), mask_y=mask_y)

        C, N = self.models_configs['num_patches']
        assert x_aug.shape[-1]%N==0 and x_aug.shape[-2]%C == 0
        block_size_c, block_size_n = x_aug.shape[-2]//C, x_aug.shape[-1]//N
        x_aug = x_aug.view(x_aug.shape[0], C, block_size_c, N, block_size_n)
        x_aug = x_aug.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
        x_aug = x_aug.view(x_aug.shape[0], C, N, block_size_c * block_size_n)
        
        out_rec_true = apply_mask(mask_y.to(x_aug.device), x_aug)
        out_rec_true = F.layer_norm(out_rec_true, (out_rec_true.size(-1),))

        loss = self.loss_fn(out_rec, out_rec_true)

        return {
            "loss": loss
        }