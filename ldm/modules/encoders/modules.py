import torch
import torch.nn as nn
from functools import partial
import clip
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer, AutoModel
#from ldm.modules.encoders.transformer import BasicTransformerBlock
# import kornia
from typing import Optional

from ldm.modules.x_transformer import Encoder, \
    TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


from ldm.modules.diffusionmodules.model import Encoder as AEEncoder  # Encoder module of KL-reg autoencoder


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class AEEncoderEmbedder(AbstractEncoder):
    """the encoder part of kl/vq-reg autoencoder, used for conditioning"""
    def __init__(self, **ddconfig):
        super().__init__()
        self.embed_dim = ddconfig.pop("embed_dim")
        ddconfig["conditioning"] = True
        self.encoder = AEEncoder(**ddconfig)
        scale_factor = 2 if ddconfig["double_z"] else 1
        self.out = torch.nn.Conv2d(scale_factor * ddconfig["z_channels"], self.embed_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.out(x)
        return x

    def encode(self, x):
        return self(x)

class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.LayerNorm(query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # linear norm
        hidden_states = self.to_out[1](hidden_states)
        # dropout
        hidden_states = self.to_out[2](hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query[start_idx:end_idx],
                key[start_idx:end_idx].transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        # inner_dim = int(dim * mult)
        inner_dim = dim
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        # project in
        self.net.append(nn.GELU())
        # self.net.append(geglu)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the context vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )  # is self-attn if context is none

        # layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.use_ada_layer_norm = num_embeds_ada_norm is not None

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = (self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states))

        if self.only_cross_attention:
            hidden_states = self.attn1(norm_hidden_states, context) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = (self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states))
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer(
                'dct_weight_{}'.format(freq_idx),
                self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels)
            )

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class TextGuidedMultiFrequencyAttention(nn.Module):
    """文本引导的多频率注意力"""

    def __init__(self, in_channels, text_dim=768, dct_h=32, dct_w=32):
        super().__init__()

        print(f"[TextGuidedMultiFrequencyAttention] Initializing with in_channels={in_channels}, text_dim={text_dim}")

        # 动态计算合适的reduction值（确保不会为0）
        # 对于 in_channels=4: reduction=2 → 4 // 2 = 2 (valid)
        reduction = max(2, in_channels // 2)
        print(f"[TextGuidedMultiFrequencyAttention] Using reduction={reduction} (in_channels={in_channels})")

        # 三个频率分支（高频、中频、低频）
        self.high_freq = MultiFrequencyChannelAttention(
            in_channels, dct_h, dct_w,
            frequency_branches=16,
            frequency_selection='top',
            reduction=reduction
        )
        self.mid_freq = MultiFrequencyChannelAttention(
            in_channels, dct_h, dct_w,
            frequency_branches=8,
            frequency_selection='low',
            reduction=reduction
        )
        self.low_freq = MultiFrequencyChannelAttention(
            in_channels, dct_h, dct_w,
            frequency_branches=8,
            frequency_selection='bot',
            reduction=reduction
        )

        # 文本到频率权重的映射
        self.text_to_freq_weights = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # [high_weight, mid_weight, low_weight]
            nn.Softmax(dim=-1)
        )

        print(f"[TextGuidedMultiFrequencyAttention] ✅ Initialized successfully")

    def forward(self, x, text_embedding):
        """
        Args:
            x: [B, C, H, W] 融合后的特征
            text_embedding: [B, text_dim] 文本的全局表示
        """
        # 三个频率分支的增强
        x_high = self.high_freq(x)  # 高频（细节）
        x_mid = self.mid_freq(x)  # 中频（边缘）
        x_low = self.low_freq(x)  # 低频（整体）

        # 文本预测频率权重
        freq_weights = self.text_to_freq_weights(text_embedding)  # [B, 3]

        # 自适应加权融合
        x_enhanced = (
                freq_weights[:, 0:1, None, None] * x_high +
                freq_weights[:, 1:2, None, None] * x_mid +
                freq_weights[:, 2:3, None, None] * x_low
        )

        return x_enhanced

# ldm/modules/encoders/modules.py

class ImageTextFusionEmbedder(AbstractEncoder):
    """融合图像和文本的条件编码器"""

    def __init__(self,
                 image_encoder_config,
                 text_encoder_config,
                 fusion_type="cross_attn",
                 embed_dim=4,
                 use_frequency_enhancement=True):  # ← 新参数
        super().__init__()

        # 1. 图像编码器
        self.image_encoder = AEEncoderEmbedder(**image_encoder_config)

        # 2. 文本编码器配置
        self.text_model_name = text_encoder_config.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
        self.max_length = text_encoder_config.get('max_length', 10)
        self.freeze_text = text_encoder_config.get('freeze', False)

        # 3. 加载BERT
        from transformers import AutoTokenizer, AutoModel
        import os

        local_model_path = "models/Bio_ClinicalBERT"
        if os.path.exists(local_model_path):
            print(f"[ImageTextFusionEmbedder] ✅ Loading BERT from local path: {local_model_path}")
            model_path = local_model_path
        else:
            print(f"[ImageTextFusionEmbedder] ⚠️ Local model not found, trying to download: {self.text_model_name}")
            model_path = self.text_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_encoder = AutoModel.from_pretrained(model_path)

        if self.freeze_text:
            print(f"[ImageTextFusionEmbedder] Freezing text encoder")
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            print(f"[ImageTextFusionEmbedder] Text encoder is trainable")

        # 4. 融合模块
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == "cross_attn":
            self.fusion = BasicTransformerBlock(
                dim=embed_dim,  # ← 注意这里是 embed_dim，不是 embed_dim * 32 * 32
                num_attention_heads=8,
                attention_head_dim=64,
                cross_attention_dim=768,
                only_cross_attention=True
            )

            # 🔥 新增：频域增强模块
            self.use_frequency_enhancement = use_frequency_enhancement
            if use_frequency_enhancement:
                self.freq_enhancement = TextGuidedMultiFrequencyAttention(
                    in_channels=embed_dim,  # 4
                    text_dim=768,  # BERT输出维度
                    dct_h=32,  # latent高度
                    dct_w=32  # latent宽度
                )
                print(f"[ImageTextFusionEmbedder] ✅ Text-Guided Frequency Enhancement enabled")

            self.proj_out = nn.Conv2d(embed_dim, embed_dim, 1)
        else:
            raise NotImplementedError(f"Fusion type {fusion_type} not implemented")

    def encode_text(self, text_list, device):
        """编码文本列表"""
        text_inputs = self.tokenizer(
            text_list,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        if self.freeze_text:
            with torch.no_grad():
                text_outputs = self.text_encoder(**text_inputs)
        else:
            text_outputs = self.text_encoder(**text_inputs)

        return text_outputs['last_hidden_state']  # [B, max_length, 768]

    def forward(self, img, text_list):
        """
        Args:
            img: [B, 3, 256, 256] 原始图像
            text_list: List[str] 文本描述

        Returns:
            z_fused: [B, 4, 32, 32] 融合后的条件latent
        """
        device = img.device

        # 1. 编码图像
        z_img = self.image_encoder(img)  # [B, 4, 32, 32]

        # 2. 编码文本
        z_text = self.encode_text(text_list, device)  # [B, max_length, 768]

        # 3. Cross-Attention融合
        if self.fusion_type == "cross_attn":
            B, C, H, W = z_img.shape

            if H != 32 or W != 32:
                print(f"[Warning] Expected latent size 32x32, got {H}x{W}")

            # 展平图像特征
            z_img_flat = z_img.flatten(2).permute(0, 2, 1)  # [B, 1024, 4]

            # Cross-Attention
            z_fused = self.fusion(z_img_flat, context=z_text)  # [B, 1024, 4]

            # 还原空间形状
            z_fused = z_fused.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 4, 32, 32]

            # 🔥 4. 频域增强（如果启用）
            if self.use_frequency_enhancement:
                # 提取文本的全局表示（mean pooling）
                text_global = z_text.mean(dim=1)  # [B, 768]

                # 文本引导的频域增强
                z_fused = self.freq_enhancement(z_fused, text_global)  # [B, 4, 32, 32]

            # 5. 投影输出
            z_fused = self.proj_out(z_fused)

            return z_fused
        else:
            raise NotImplementedError(f"Fusion type {self.fusion_type} not implemented")

    def encode(self, img, text_list=None):
        """兼容旧版调用"""
        if text_list is None:
            batch_size = img.shape[0]
            text_list = ["polyp lesion in colon mucosa"] * batch_size
        return self.forward(img, text_list)

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class FrozenBioClinicalBERTEmbedder(AbstractEncoder):
    """Uses the Bio_ClinicalBERT transformer encoder for medical text"""

    def __init__(self, version="emilyalsentzer/Bio_ClinicalBERT", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.transformer = AutoModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        return_length=True, return_overflowing_tokens=False,
                                        padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """

    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params

    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
