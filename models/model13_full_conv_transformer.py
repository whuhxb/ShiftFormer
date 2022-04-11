import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = GroupNorm(embed_dim)
        # self.apply(self._init_weights)

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

        # self.apply(self._init_weights)

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
        # self.apply(self._init_weights)

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


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, useSpatialAtt=True, useChannelAtt=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = TokenMixer(dim=dim, useSpatialAtt=useSpatialAtt)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = ChannelMixer(dim=dim, hidden_dim=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop, useChannelAtt=useChannelAtt)

        # self.apply(self._init_weights)

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
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.channel_mixer(self.norm2(x)))
        return x




class BaseTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                  mlp_ratios=[4, 4, 4, 4], drop_rate=0.
                 , drop_path_rate=0., norm_layer=GroupNorm, act_layer=nn.GELU,
                 depths=[3, 4, 6, 3], num_stages=4, useSpatialAtt=True, useChannelAtt=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i],  mlp_ratio=mlp_ratios[i],
                drop=drop_rate, drop_path=dpr[cur + j], act_layer=act_layer, norm_layer=norm_layer,
                useSpatialAtt=useSpatialAtt, useChannelAtt=useChannelAtt)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)

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


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean([-2, -1])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x




@register_model
def fct_s12_32(pretrained=False, **kwargs):
    model = BaseTransformer(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=GroupNorm, depths=[2, 2, 6, 2], useSpatialAtt=False, useChannelAtt=False,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def fct_s12_32_att(pretrained=False, **kwargs):
    model = BaseTransformer(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=GroupNorm, depths=[2, 2, 6, 2], useSpatialAtt=True, useChannelAtt=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def fct_s12_64_att(pretrained=False, **kwargs):
    model = BaseTransformer(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=GroupNorm, depths=[2, 2, 6, 2], useSpatialAtt=True, useChannelAtt=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def fct_s24_64_att(pretrained=False, **kwargs):
    model = BaseTransformer(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=GroupNorm, depths=[4, 4, 12, 4], useSpatialAtt=True, useChannelAtt=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = fct_s12_32()
    out = model(input)
    # print(model)
    print(out.shape)
