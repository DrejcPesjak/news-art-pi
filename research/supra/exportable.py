"""Numerically equivalent layout changes for NCNN's limited tensor rank."""
import types
import torch
from torch import nn
from torch.nn import functional as F

from .model import Attention


def attention_forward(self, x, ctx=None, ctx_mask=None):
    batch, length, channels = x.shape
    if self.is_self:
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, index].transpose(1, 2) for index in range(3))
    else:
        ctx_length = ctx.shape[1]
        q = self.q(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).view(batch, ctx_length, 2, self.n_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    # Inputs have 0/1 masks. -10000 underflows masked probabilities to zero in
    # fp32 and avoids a runtime bool cast unsupported by the NCNN converter.
    additive = (ctx_mask[:, None, None, :]-1)*10000 if ctx_mask is not None else None
    scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
    if additive is not None:
        scores = scores + additive
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return self.proj(out.transpose(1, 2).reshape(batch, length, channels))


class ExportDiT(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.model = original
        for module in original.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(attention_forward, module)

    def forward(self, z, t, context, mask):
        model = self.model
        # Original patchify goes through a six-dimensional tensor. A stride-two
        # convolution applies precisely the same learned patch projection.
        projected = F.conv2d(z, model.x_embed.weight.reshape(-1, 4, 2, 2),
                             model.x_embed.bias, stride=2)
        x = projected.flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embed(t)
        context = model.ctx_proj(context)
        for block in model.blocks:
            x = block(x, c, context, mask)
        x = model.final(x, c).transpose(1, 2).reshape(1, 16, 16, 16)
        return F.pixel_shuffle(x, 2)
