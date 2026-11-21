"""Fixed-shape encoder and decoder graphs for staged CPU inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pnnx
import torch
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKL

from .assets import MODEL_DIR, ROOT
from .model import VAE_SCALE
from .ncnn_runtime import NCNNModel


class EncoderCore(nn.Module):
    """T5 encoder math without tokenizer, embedding table, or HF mask builders."""
    def __init__(self, text):
        super().__init__()
        self.blocks = text.encoder.block
        self.final_norm = text.encoder.final_layer_norm
        bias = self.blocks[0].layer[0].SelfAttention.compute_bias(128,128)
        self.register_buffer("position_bias",bias.detach())

    def forward(self, embedded, mask):
        hidden = embedded
        bias = self.position_bias + (mask[:,None,None,:]-1)*1e9
        for block in self.blocks:
            attention_layer = block.layer[0]
            attention = attention_layer.SelfAttention
            normalized = attention_layer.layer_norm(hidden)
            q = attention.q(normalized).reshape(1,128,attention.n_heads,attention.key_value_proj_dim).transpose(1,2)
            k = attention.k(normalized).reshape(1,128,attention.n_heads,attention.key_value_proj_dim).transpose(1,2)
            v = attention.v(normalized).reshape(1,128,attention.n_heads,attention.key_value_proj_dim).transpose(1,2)
            # T5 attention intentionally has no 1/sqrt(d) scaling.
            weights = torch.softmax(torch.matmul(q,k.transpose(-2,-1))+bias,dim=-1)
            context = torch.matmul(weights,v).transpose(1,2).reshape(1,128,-1)
            hidden = hidden + attention.o(context)
            feedforward = block.layer[1]
            dense = feedforward.DenseReluDense
            normalized = feedforward.layer_norm(hidden)
            projected = dense.act(dense.wi_0(normalized))*dense.wi_1(normalized)
            hidden = hidden + dense.wo(projected)
        return self.final_norm(hidden)


class VAEAttentionProcessor:
    """The pinned decoder's self-attention, expressed without SDPA."""
    def __call__(self, attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("This decoder export only supports unmasked self-attention")
        residual = hidden_states
        batch,channels,height,width = hidden_states.shape
        hidden = hidden_states.reshape(batch,channels,height*width).transpose(1,2)
        if attention.group_norm is not None:
            hidden = attention.group_norm(hidden.transpose(1,2)).transpose(1,2)
        heads = attention.heads
        dimension = channels//heads
        q = attention.to_q(hidden).reshape(batch,height*width,heads,dimension).transpose(1,2)
        k = attention.to_k(hidden).reshape(batch,height*width,heads,dimension).transpose(1,2)
        v = attention.to_v(hidden).reshape(batch,height*width,heads,dimension).transpose(1,2)
        weights = torch.softmax(torch.matmul(q,k.transpose(-2,-1))*attention.scale,dim=-1)
        hidden = torch.matmul(weights,v).transpose(1,2).reshape(batch,height*width,channels)
        hidden = attention.to_out[1](attention.to_out[0](hidden))
        hidden = hidden.transpose(1,2).reshape(batch,channels,height,width)
        if attention.residual_connection:
            hidden = hidden+residual
        return hidden/attention.rescale_output_factor


class DecoderCore(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self,z):
        return self.vae.decode(z/VAE_SCALE,return_dict=False)[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("component",choices=["text","vae"])
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/supra/ncnn")
    parser.add_argument("--fp32-storage",action="store_true")
    parser.add_argument("--tile-size",type=int,default=32)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0)
    label=f"vae_tile{args.tile_size}" if args.component=="vae" and args.tile_size!=32 else args.component
    prefix=args.output/label
    if args.component=="text":
        text=T5EncoderModel.from_pretrained(MODEL_DIR/"text",local_files_only=True,
                                            dtype=torch.float32,attn_implementation="eager").eval().requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR/"text",local_files_only=True)
        tokens=tokenizer("A rocket launching from a coastal spaceport",padding="max_length",max_length=128,return_tensors="pt")
        embedded=text.shared(tokens["input_ids"])
        inputs=(embedded,tokens["attention_mask"].float())
        with torch.inference_mode():
            expected=text(**tokens).last_hidden_state.numpy()
        np.save(args.output/"embedding.npy",text.shared.weight.detach().numpy().astype(np.float16))
        model=EncoderCore(text).eval()
        with torch.inference_mode():rewritten=model(*inputs).numpy()
        np.savez(args.output/"text-fixture.npz",in0=embedded.detach().numpy(),in1=inputs[1].numpy(),expected=expected)
    else:
        vae=AutoencoderKL.from_pretrained(MODEL_DIR/"vae",local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False)
        vae.encoder=None;vae.quant_conv=None
        inputs=(torch.randn(1,4,args.tile_size,args.tile_size)*.2,)
        with torch.inference_mode():expected=vae.decode(inputs[0]/VAE_SCALE).sample.numpy()
        vae.set_attn_processor(VAEAttentionProcessor())
        model=DecoderCore(vae).eval()
        with torch.inference_mode():rewritten=model(*inputs).numpy()
        np.savez(args.output/(label+"-fixture.npz"),in0=inputs[0].numpy(),expected=expected)
    rewrite_rmse=float(np.sqrt(np.mean((expected-rewritten)**2)))
    rewrite_relative=rewrite_rmse/max(float(np.sqrt(np.mean(expected**2))),1e-8)
    print('Rewrite relative RMSE:',rewrite_relative,flush=True)
    if rewrite_relative>.0001:raise ValueError("Component math rewrite failed numerical validation")
    pnnx.export(model,str(prefix)+".pt",inputs,check_trace=False,fp16=not args.fp32_storage)
    runtime=NCNNModel(prefix,threads=4)
    started=time.perf_counter()
    actual=runtime(*(value.detach().numpy() for value in inputs))
    seconds=time.perf_counter()-started
    rmse=float(np.sqrt(np.mean((expected-actual)**2)))
    relative=rmse/max(float(np.sqrt(np.mean(expected**2))),1e-8)
    result={"component":label,"rewrite_relative_rmse":rewrite_relative,
            "ncnn_relative_rmse":relative,"max_abs_error":float(np.max(np.abs(expected-actual))),
            "x86_seconds":seconds,"weights_mib":Path(str(prefix)+".ncnn.bin").stat().st_size/2**20,
            "passed":relative<.01}
    (args.output/(label+"-validation.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if not result["passed"]:raise ValueError("NCNN component conversion failed numerical validation")


if __name__=="__main__":main()
