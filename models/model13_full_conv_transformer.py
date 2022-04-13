
"""
Based on PoolFormer, change to static shift.
"""
import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple


try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    # print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    # print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    's': _cfg(crop_pct=0.9),
    'm': _cfg(crop_pct=0.95),
}


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = GroupNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SpatialAtt(nn.Module):
    def __init__(self, dim, s_att_ks=7, s_att_r=4):
        super().__init__()
        self.spatial_att = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=s_att_ks, stride=s_att_r, groups=dim, padding=s_att_ks//2),
                                    nn.BatchNorm2d(dim),
                                    nn.Sigmoid())

    def forward(self,x):
        _,_,H,W = x.size()
        # print(f"spatial att shape: {x.size()}")
        return x * F.interpolate(self.spatial_att(x), (H,W))


class ChannelAtt(nn.Module):
    def __init__(self, dim, c_att_ks=7, c_att_r=8):
        super().__init__()
        self.hidden_dim=max(8, dim//c_att_r)
        self.channel_att = nn.Sequential(
            nn.AvgPool2d(c_att_ks, c_att_ks, padding=c_att_ks//2),
            nn.Conv2d(dim, self.hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        _,_,H,W = x.size()
        # print(f"channel att shape: {x.size()}")
        return x * F.interpolate(self.channel_att(x), (H,W))


class TokenMixer(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, useBN=True, useSpatialAtt=True):
        super().__init__()
        self.act = act_layer()
        self.useBN = useBN
        self.useSpatialAtt = useSpatialAtt
        self.dw3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.fc = nn.Conv2d(dim,dim,kernel_size=1, padding=0, stride=1, groups=1)
        self.dw5x5dilated = nn.Conv2d(dim, dim, kernel_size=5, padding=4, stride=1, groups=dim, dilation=2)
        if useBN:
            self.dw3x3BN = nn.BatchNorm2d(dim)
            self.dw5x5dilatedBN = nn.BatchNorm2d(dim)
        if useSpatialAtt:
            self.spatial_att = SpatialAtt(dim=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.act(self.dw3x3BN(self.dw3x3(x))) if self.useBN else self.act(self.dw3x3(x))
        x = self.act((self.fc(x)))
        x = self.act(self.dw5x5dilatedBN(self.dw5x5dilated(x))) if self.useBN else self.act(self.dw5x5dilated(x))
        if self.useSpatialAtt:
            x = self.spatial_att(x)
        return x


class ChannelMixer(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0., useChannelAtt=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.useChannelAtt = useChannelAtt
        self.act = act_layer()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)
        if useChannelAtt:
            self.channel_att = ChannelAtt(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.dwconv(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.useChannelAtt:
            x = self.channel_att(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 useBN=False, useSpatialAtt=False, useChannelAtt=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = TokenMixer(dim=dim, act_layer=act_layer,
                                      useBN=useBN, useSpatialAtt=useSpatialAtt)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = ChannelMixer(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer,
                                          drop=drop, useChannelAtt=useChannelAtt)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.channel_mixer(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.channel_mixer(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 useBN=False, useSpatialAtt=False, useChannelAtt=False):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(BasicBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            useBN=useBN, useSpatialAtt=useSpatialAtt, useChannelAtt=useChannelAtt
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class BaseFormer(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 useBN=False, useSpatialAtt=False, useChannelAtt=False,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 useBN=useBN, useSpatialAtt=useSpatialAtt, useChannelAtt=useChannelAtt)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    OverlapPatchEmbed(patch_size= 3, stride= 2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1])
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def fct_s12_32(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    useBN, useSpatialAtt, useChannelAtt = False, False, False
    downsamples = [True, True, True, True]
    model = BaseFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        useBN=useBN, useSpatialAtt=useSpatialAtt, useChannelAtt=useChannelAtt,
        **kwargs)
    model.default_cfg = default_cfgs['s']
    return model




if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = fct_s12_32()
    out = model(input)
    print(model)
    print(out.shape)
