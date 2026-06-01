# EEGPTConfigBuilder
from omegaconf import OmegaConf

class EEGPTConfigBuilder:

    def build(self, cfg):

        model_cfg = OmegaConf.to_container(
            cfg.model,
            resolve=True
        )

        img_size = model_cfg['img_size']
        patch_size = model_cfg['patch_size']

        seq_len = img_size[1] // patch_size
        num_patches = (img_size[0], seq_len)

        embed_dim = model_cfg['embed_dim']
        embed_num = model_cfg['embed_num']
        num_heads = model_cfg['num_heads']

        complete_cfg = {

            'img_size': img_size,
            'patch_size': patch_size,
            'num_patches': num_patches,

            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'num_heads': num_heads,

            'encoder': {
                'img_size': img_size,
                'patch_size': patch_size,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'depth': model_cfg['encoder_depth'],
                'num_heads': num_heads,

                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),

                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),

                'init_std': model_cfg.get('init_std', 0.02),
            },

            'predictor': {
                'num_patches': num_patches,
                'embed_dim': embed_dim,
                'embed_num': embed_num,

                'predictor_embed_dim':
                    model_cfg.get(
                        'predictor_embed_dim',
                        embed_dim
                    ),

                'depth': model_cfg['predictor_depth'],
                'num_heads': num_heads,

                'use_part_pred':
                    model_cfg.get('use_part_pred', True),

                'mlp_ratio':
                    model_cfg.get('mlp_ratio', 4.0),

                'qkv_bias':
                    model_cfg.get('qkv_bias', True),

                'drop_rate':
                    model_cfg.get('drop_rate', 0.0),

                'attn_drop_rate':
                    model_cfg.get('attn_drop_rate', 0.0),

                'drop_path_rate':
                    model_cfg.get('drop_path_rate', 0.0),

                'init_std':
                    model_cfg.get('init_std', 0.02),
            },

            'reconstructor': {
                'num_patches': num_patches,
                'patch_size': patch_size,

                'embed_dim': embed_dim,
                'embed_num': embed_num,

                'reconstructor_embed_dim':
                    model_cfg.get(
                        'reconstructor_embed_dim',
                        embed_dim
                    ),

                'depth': model_cfg['reconstructor_depth'],
                'num_heads': num_heads,

                'mlp_ratio':
                    model_cfg.get('mlp_ratio', 4.0),

                'qkv_bias':
                    model_cfg.get('qkv_bias', True),

                'drop_rate':
                    model_cfg.get('drop_rate', 0.0),

                'attn_drop_rate':
                    model_cfg.get('attn_drop_rate', 0.0),

                'drop_path_rate':
                    model_cfg.get('drop_path_rate', 0.0),

                'init_std':
                    model_cfg.get('init_std', 0.02),
            },
        }

        return complete_cfg