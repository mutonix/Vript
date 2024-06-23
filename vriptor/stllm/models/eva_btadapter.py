# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
import copy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .eva_vit import create_eva_vit_g, DropPath, Attention, Block, VisionTransformer,\
    interpolate_pos_embed, convert_weights_to_fp16

from einops import rearrange
try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_

from torch.utils.checkpoint import checkpoint


if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class EVAVisionTransformer_BTAdapter(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, depth=4, mask_rate=0):
        super().__init__()

        clip = create_eva_vit_g(
            224, 0, False, "fp16"
        )
        
        self.image_size = clip.image_size
        self.num_classes = clip.num_classes
        self.num_features = self.embed_dim = clip.embed_dim  # num_features for consistency with other models
        self.num_heads = clip.num_heads
        self.patch_embed = clip.patch_embed
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = clip.cls_token
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = clip.pos_embed
        self.pos_drop = clip.pos_drop
        self.rel_pos_bias = clip.rel_pos_bias

        self.rope = None

        self.use_rel_pos_bias = clip.use_rel_pos_bias
        self.blocks = clip.blocks

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.grad_checkpointing = False

        self.depth = depth
        self.mask_rate = mask_rate
        
        dpr = np.linspace(0, 0.1, depth)
        self.BTAdapter_cls =  nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.BTAdapter_S = nn.ModuleList([BTAdapter_Spatial(self.embed_dim, self.num_heads, drop_num=dpr[i]) for i in range(depth)])
        self.BTAdapter_T = nn.ModuleList([BTAdapter_Temp(self.embed_dim, self.num_heads, drop_num=dpr[i]) for i in range(depth)])
        self.BTAdapter_position = nn.Embedding(64, self.embed_dim)
        self.init_weights()

        del clip

    def init_weights(self):
        total_depth = len(self.blocks)
        self.num_layers = total_depth
        layer_para = self.blocks.state_dict()
        spatial_para = {}
        load_start = total_depth - self.depth
        for k, v in layer_para.items():
            num_layer = int(k.split(".")[0])
            if num_layer >= load_start:
                spatial_para[k.replace(str(num_layer),str(num_layer-load_start),1)] = v.clone()
        self.BTAdapter_S.load_state_dict(spatial_para)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask=None):
        
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        branch_x = None
        
        for idx,blk in enumerate(self.blocks):
            if self.training and self.grad_checkpointing:
                x = checkpoint(blk, x, rel_pos_bias, None)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            if idx >= self.num_layers-self.depth:
                num_layer = idx + self.depth - self.num_layers
                branch_x = self.forward_branch(x, branch_x, num_layer, mask)
        
        #x_patch, branch_patch = x[:,1:,], branch_x[:,1:,]
        #x_cls = rearrange(x[:,:1], '(b t) p m -> b t p m', t=self.T).mean(dim=1)
        #branch_cls = branch_x[:,:1,]
        #x_patch = rearrange(x_patch, '(b t) p m -> b t p m', t=self.T).mean(dim=1)
        #p = x_patch.size(1)
        #branch_patch = rearrange(branch_patch, 'b (p t) m -> b t p m', p=p).mean(dim=1)
        #x = torch.cat(((x_patch+branch_patch)/2,(x_cls+branch_cls)/2),dim=1)

        p = x.size(1) - 1
        branch_cls, branch_patch = branch_x[:,0], branch_x[:,1:,]
        branch_patch = rearrange(branch_patch, 'b (p t) m -> (b t) p m', p=p)
        branch_cls = branch_cls.repeat(1, self.T).view(branch_cls.size(0) * self.T, -1).unsqueeze(1)
        x = (x + torch.cat((branch_cls,branch_patch),dim=1))/2
        return x

    def forward_branch(self, x, branch_x, num_layer, mask=None):
        x = rearrange(x, '(b t) l d -> b t l d', t=self.T)
        if branch_x is not None:
            cls_x = x[:,:,0]
            cls_branch = cls_x.mean(dim=1).unsqueeze(1)
            x = rearrange(x[:,:,1:], 'b t l d -> b (l t) d')
            if mask is not None:
                B, _, D = x.size()
                x = x[~mask].reshape(B,-1,D)
            x = torch.cat((cls_branch,x), dim=1)
            x = x + branch_x
        
        if num_layer==0:
            x = self.init_input(x,mask)

        if self.grad_checkpointing and self.training:
            x = checkpoint(self.BTAdapter_T[num_layer],x,self.T)
            x = checkpoint(self.BTAdapter_S[num_layer],x,self.T)
        else: 
            x = self.BTAdapter_T[num_layer](x, self.T)
            x = self.BTAdapter_S[num_layer](x, self.T)
        return x
    
    def init_input(self, x, mask=None):
        cls_x = x[:,:,0].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        b,t,l,d = x.size()
        x = rearrange(x, 'b t l d -> (b t) l d')
        #cls_branch = self.class_embedding.expand(1, x.size(1), -1)
        cls_branch = self.BTAdapter_cls.expand(x.size(0), 1, -1)
        x = torch.cat((cls_branch, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_branch = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) l d -> (b l) t d', b=b)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0),-1)
        time_embed = self.BTAdapter_position(position_ids)
        x = x + time_embed

        x = rearrange(x, '(b l) t d -> b (l t) d', b=b)
        if mask is not None:
            x = x[~mask].reshape(b,-1,d)
        cls = (cls_x + cls_branch) / 2
        x = torch.cat((cls, x), dim=1)
        return x
    
    def forward(self, x, return_all_features=False):
        if x.ndim == 5:
            if x.shape[1]==3:
                B, D, T, H, W = x.shape             
                x = x.permute(0, 2, 1, 3, 4)
            else:
                B, T, D, H, W = x.shape   
            x = x.reshape((-1,) + x.shape[2:])
        elif x.ndim == 4:
            T, D, H, W = x.shape 
            B = 1
        else:
            B, _, _, _ = x.shape
            T = 1
        self.T = T

        if self.mask_rate>0 and self.training:
            mask = TubeMaskingGenerator((T,self.num_patches),self.mask_rate,B,x.device)
        else:
            mask = None

        x = self.forward_features(x, mask)
        return x

class BTAdapter_Spatial(Block):
    def __init__(self, d_model, n_head, drop_num=0.1):
        super().__init__(dim=d_model, num_heads=n_head, drop_path=drop_num, qkv_bias=True, mlp_ratio=4.3637)
        
    def forward(self, x, T):
        residual = x
        init_cls_token = x[:,:1,:]
        query_s = x[:, 1:, :]

        b, pt, m = query_s.size()
        p, t = pt//T, T

        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b*t, 1, m)
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)

        x = torch.cat((cls_token, query_s), 1)
        x = self.attn(self.norm1(x))
        res_spatial = self.drop_path(x.contiguous())
        cls_token = res_spatial[:, :1, :].reshape(b, t, 1, m).mean(1)
        res_spatial = rearrange(res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        
        x = residual + torch.cat((cls_token, res_spatial), 1)
        x = x + self.mlp(self.norm2(x))
        x = self.drop_path(x.contiguous())
        return x

class BTAdapter_Temp(nn.Module):
    def __init__(self, d_model, n_head, drop_num=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.drop_path = DropPath(drop_num) if drop_num > 0. else nn.Identity()
        self.attn = Attention(
            d_model, num_heads=n_head, qkv_bias=True)
        self.norm1 = norm_layer(d_model)

        self.temporal_fc = nn.Linear(d_model, d_model)
        constant_init(self.temporal_fc, val=0, bias=0)


    def forward(self, x, T):
        residual = x[:, 1:, :]

        init_cls_token = x[:, :1, :]
        query_t = x[:, 1:, :]
        b, pt, m = query_t.size()
        p, t = pt // T, T
        x = query_t.reshape(b * p, t, m)

        x = self.attn(self.norm1(x))
        res_temporal = self.drop_path(x.contiguous())
        res_temporal = self.temporal_fc(res_temporal)

        x = res_temporal.reshape(b, p * t, m) + residual
        x = torch.cat((init_cls_token, x), 1)
        return x
    
def create_eva_btadapter(precision="fp16"):
    model = EVAVisionTransformer_BTAdapter(depth=3)  
    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(model)
    return model
