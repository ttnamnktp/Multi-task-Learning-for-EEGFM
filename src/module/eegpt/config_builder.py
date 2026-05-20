# EEGPTConfigBuilder
from pathlib import Path
import yaml
from omegaconf import OmegaConf

class EEGPTConfigBuilder:
    """Build complete EEGPT config from variants + overrides"""
    
    def __init__(self):
        self.variants = self._load_variants()
        
    def _load_variants(self):
        """Load variants from yaml file"""
        variants_path = "/home/infres/ttran-25/eegfm/configs/model/eegpt/variants.yaml"
        with open(variants_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build(self, cfg):
        """
        Build complete model config from:
        - Base variant config
        - Override from cfg.model
        - Calculate derived values
        """
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        
        # Get variant
        variant_name = model_cfg.get('variant', 'L_822')
        variant_cfg = self.variants['variants'][variant_name].copy()
        
        # Merge: cfg.model overrides variant
        for key in variant_cfg:
            if key not in model_cfg or model_cfg[key] is None:
                model_cfg[key] = variant_cfg[key]
        
        # Calculate derived values
        img_size = model_cfg['img_size']
        patch_size = model_cfg['patch_size']
        seq_len = img_size[1] // patch_size
        num_patches = (img_size[0], seq_len)
        
        embed_dim = model_cfg['embed_dim']
        embed_num = model_cfg['embed_num']
        num_heads = model_cfg['num_heads']
        
        # Get defaults
        # gpt_default = self.variants['gpt_default']
        # order_default = self.variants['order_default']
        
        # Build complete config
        complete_cfg = {
            # Common params
            'img_size': img_size,
            'patch_size': patch_size,
            'num_patches': num_patches,
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'num_heads': num_heads,
            
            # Encoder
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
            
            # Predictor
            'predictor': {
                'num_patches': num_patches,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'predictor_embed_dim': model_cfg.get('predictor_embed_dim', embed_dim),
                'depth': model_cfg['predictor_depth'],
                'num_heads': num_heads,
                'use_part_pred': model_cfg.get('use_part_pred', True),
                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),
                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),
                'init_std': model_cfg.get('init_std', 0.02),
            },
            
            # Reconstructor
            'reconstructor': {
                'num_patches': num_patches,
                'patch_size': patch_size,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'reconstructor_embed_dim': model_cfg.get('reconstructor_embed_dim', embed_dim),
                'depth': model_cfg['reconstructor_depth'],
                'num_heads': num_heads,
                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),
                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),
                'init_std': model_cfg.get('init_std', 0.02),
            },
            
            # Projector
            # 'projector': {
            #     'in_dim': embed_dim,
            #     'hidden_dim': embed_dim * 2,
            #     'out_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'dropout_rate': model_cfg.get('projector_dropout', 0.2),
            # },
            
            # # Contrastive predictor
            # 'contrastive_predictor': {
            #     'in_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'hidden_dim': model_cfg.get('projector_out_dim', embed_dim) // 2,
            #     'out_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'dropout_rate': model_cfg.get('contrastive_predictor_dropout', 0.1),
            # },
            
            # # GPT decoder
            # 'gpt_decoder': {
            #     'embed_dim': model_cfg.get('gpt_embed_dim', gpt_default['embed_dim']),
            #     'num_hidden_layers': model_cfg.get('gpt_num_hidden_layers', gpt_default['num_hidden_layers']),
            #     'num_attention_heads': model_cfg.get('gpt_num_attention_heads', gpt_default['num_attention_heads']),
            #     'intermediate_dim_factor': gpt_default['intermediate_dim_factor'],
            #     'hidden_activation': gpt_default['hidden_activation'],
            #     'dropout': gpt_default['dropout'],
            #     'n_positions': seq_len,
            #     'in_dim': embed_dim,
            # },
            
            # # Order classifier
            # 'order_classifier': {
            #     'feature_dim': embed_dim * embed_num,
            #     'num_subsamples': seq_len,
            #     'num_classes': seq_len,
            #     'num_layers': model_cfg.get('order_num_layers', order_default['num_layers']),
            #     'num_heads': model_cfg.get('order_num_heads', order_default['num_heads']),
            #     'hidden_dim': embed_dim,
            # },
            
            # # Pairwise order classifier
            # 'pairwise_order_classifier': {
            #     'feature_dim': embed_dim * embed_num,
            #     'hidden_dim': (embed_dim * embed_num) // 2,
            #     'num_pair': seq_len * 2,
            # },
        }
        
        return complete_cfg