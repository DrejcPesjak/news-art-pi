"""Bounded-memory Inko/Qwen math: NumPy + NCNN, without PyTorch.

Weights remain packed/mapped on disk. Only the current matrix is dequantized.
This deliberately avoids NCNN's scalar int4 GEMM fallback on ARMv7, using its
optimized FP32 GEMM instead. It does not claim integer arithmetic speedups.
"""
import json
import mmap
import ctypes
from pathlib import Path
import time

import numpy as np

from .quant import dequantize

_LIBC=ctypes.CDLL(None)


def reclaim():
    """Release unused glibc heap pages between changing matrix sizes."""
    if hasattr(_LIBC,"malloc_trim"):
        _LIBC.malloc_trim(0)


def silu(x):
    return x / (1 + np.exp(np.clip(-x, -80, 80)))


def rms(x, weight, eps=1e-6):
    return x * (1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)) * weight


def layernorm(x, eps=1e-6):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered*centered, axis=-1, keepdims=True)+eps)


class Matmul:
    def __init__(self, threads=4, backend="ncnn"):
        self.backend = backend
        if backend == "ncnn":
            import ncnn
            self.ncnn = ncnn
            self.net = ncnn.Net()
            self.blob_pool=ncnn.UnlockedPoolAllocator()
            self.workspace_pool=ncnn.PoolAllocator()
            self.blob_pool.set_size_compare_ratio(0)
            self.workspace_pool.set_size_compare_ratio(0)
            self.net.opt.blob_allocator=self.blob_pool
            self.net.opt.workspace_allocator=self.workspace_pool
            self.net.opt.num_threads = threads
            self.net.opt.use_vulkan_compute = False
            self.net.opt.use_packing_layout = False
            self.net.opt.use_fp16_arithmetic = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_packed = False
            graph = "7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nGemm gemm 2 1 a b out 3=1 6=1 10=-1\n"
            if self.net.load_param_mem(graph) != 0:
                raise RuntimeError("NCNN dynamic GEMM unavailable")
            self.net.load_model_mem(b"")

    def release_workspace(self):
        if self.backend=="ncnn":
            # Completed calls return independent NumPy copies, so no live Mat
            # refers to these cached scratch buffers at the next matrix load.
            self.blob_pool.clear()
            self.workspace_pool.clear()
            reclaim()

    def __del__(self):
        if self.backend=="ncnn" and hasattr(self,"net"):
            self.net.clear()
            self.blob_pool.clear()
            self.workspace_pool.clear()

    def __call__(self, a, bt):
        a = np.ascontiguousarray(a, dtype=np.float32)
        bt = np.ascontiguousarray(bt, dtype=np.float32)
        if a.ndim != 2 or bt.ndim != 2 or a.shape[1] != bt.shape[1]:
            raise ValueError(f"invalid GEMM dimensions {a.shape}, {bt.shape}")
        if self.backend == "numpy":
            return a @ bt.T
        if bt.shape[0] > 2048:
            # Bound NCNN's transposed-weight packing workspace. In particular,
            # a 6144-row modulation/MLP matrix can exhaust RLIMIT_DATA after
            # allocator fragmentation over several 512px sampling passes.
            output = np.empty((a.shape[0],bt.shape[0]),dtype=np.float32)
            for start in range(0,bt.shape[0],2048):
                end=min(start+2048,bt.shape[0])
                output[:,start:end]=self(a,bt[start:end])
            return output
        with self.net.create_extractor() as ex:
            for name, value in (("a",a),("b",bt)):
                if ex.input(name, self.ncnn.Mat(value)) != 0:
                    raise RuntimeError("NCNN GEMM input failed")
            code, result = ex.extract("out")
            if code != 0:
                raise RuntimeError(f"NCNN GEMM failed: {code}")
            return result.numpy().copy()


class Weights:
    def __init__(self, directory, threads=4, backend="ncnn"):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/"manifest.json").read_text())
        self.matmul = Matmul(threads, backend)
        self.mapped = {}

    def mapped_file(self, name):
        if name not in self.mapped:
            self.mapped[name] = np.load(self.directory/name, mmap_mode="r")
        return self.mapped[name]

    def get(self, name):
        meta = self.manifest["tensors"][name]
        if "q" in meta:
            packed=self.mapped_file(meta["q"])
            scales=self.mapped_file(meta["scales"])
            result=dequantize(packed,scales,meta["shape"],meta["bits"],meta["group"])
            # Discard this process's file-backed pages after copying. Linux may
            # retain a reclaimable page-cache copy; this is not an OS cache flush.
            # Keeping all packed weights mapped into RSS is unnecessary here.
            if hasattr(mmap,"MADV_DONTNEED"):
                packed._mmap.madvise(mmap.MADV_DONTNEED)
                scales._mmap.madvise(mmap.MADV_DONTNEED)
            return result
        return np.asarray(self.mapped_file(meta["file"]),dtype=np.float32)

    def linear(self, x, name):
        self.matmul.release_workspace()
        weight = self.get(name+".weight")
        shape = x.shape
        output = self.matmul(x.reshape(-1,shape[-1]), weight)
        if name+".bias" in self.manifest["tensors"]:
            output += self.get(name+".bias")
        return output.reshape(*shape[:-1],weight.shape[0])


def attention(q, k, v, matmul, *, causal=False, chunk=128):
    """Head-at-a-time, query-chunked attention; no H*S*S allocation."""
    heads, tokens, dim = q.shape
    result = np.empty_like(q)
    for h in range(heads):
        for start in range(0,tokens,chunk):
            stop = min(tokens,start+chunk)
            scores = matmul(q[h,start:stop],k[h]) * np.float32(dim**-.5)
            if causal:
                scores[np.arange(start,stop)[:,None] < np.arange(k.shape[1])[None,:]] = -np.inf
            scores -= np.max(scores,axis=-1,keepdims=True)
            np.exp(scores,out=scores)
            scores /= scores.sum(axis=-1,keepdims=True)
            result[h,start:stop] = matmul(scores,v[h].T)
    return result


def image_rope(height, width, text_length, head_dim):
    axis_dim = head_dim//2
    inverse = np.float32(10000.) ** (-np.arange(0,axis_dim,2,dtype=np.float32)/axis_dim)
    y = np.repeat(np.arange(height,dtype=np.float32)-(height-1)/2,width)
    x = np.tile(np.arange(width,dtype=np.float32)-(width-1)/2,height)
    text = np.arange(max(height,width),max(height,width)+text_length,dtype=np.float32)
    phase = np.concatenate([np.concatenate([y,text])[:,None]*inverse,
                            np.concatenate([x,text])[:,None]*inverse],axis=-1)
    return np.cos(phase),np.sin(phase)


def rotate(x, cosine, sine):
    pairs = x.reshape(*x.shape[:-1],-1,2)
    result = np.empty_like(pairs)
    result[...,0] = pairs[...,0]*cosine[:,None,:] - pairs[...,1]*sine[:,None,:]
    result[...,1] = pairs[...,0]*sine[:,None,:] + pairs[...,1]*cosine[:,None,:]
    return result.reshape(x.shape)


class InkoArray:
    def __init__(self, directory, threads=4, backend="ncnn"):
        self.weights = Weights(directory,threads,backend)
        config = self.weights.manifest["config"]
        self.hidden = config["hidden_size"]
        self.heads = config["num_heads"]
        self.double = config["double_stream_depth"]
        self.depth = self.double+config["single_stream_depth"]
        self.timing = []

    def feedforward(self,x,name):
        w = self.weights
        up = w.linear(x,name+".up_proj")
        gate,linear = np.split(up,2,axis=-1)
        return w.linear(silu(gate)*linear,name+".down_proj")

    def qkv(self,x,prefix):
        w = self.weights
        sequence = x.shape[0]
        q,k,v = np.split(w.linear(x,prefix),3,axis=-1)
        return [a.reshape(sequence,self.heads,-1) for a in (q,k,v)]

    def forward(self,z,t,text,pooled, *, trace=None):
        """One conditional forward, batch size one, no padded text tokens."""
        reclaim()
        w = self.weights
        _,channels,height,width = z.shape
        if height%2 or width%2 or max(height//2,width//2)>74:
            raise ValueError("supports even latent sizes within the original RoPE range")
        patches = z[0].reshape(channels,height//2,2,width//2,2).transpose(1,3,0,2,4).reshape(-1,channels*4)
        conv = w.get("x_embedder.weight").reshape(self.hidden,-1)
        x = w.matmul(patches,conv)+w.get("x_embedder.bias")
        text = w.linear(text.reshape(-1,1024),"txt_embedder")
        half=self.hidden//2
        exponent=-np.log(np.float32(10000))*np.arange(half,dtype=np.float32)/half
        phase=np.exp(exponent)*np.float32(t)*1000
        time_features=np.concatenate([np.sin(phase),np.cos(phase)])[None,:]
        c=np.concatenate([time_features,w.linear(pooled.reshape(1,1024),"txt_pooled_proj")],axis=-1)
        c=w.linear(silu(w.linear(c,"c_mlp.0")),"c_mlp.2")
        cosine,sine=image_rope(height//2,width//2,len(text),self.hidden//self.heads)
        image_tokens=len(x)
        for i in range(self.depth):
            started=time.perf_counter()
            prefix=f"blocks.{i}"
            if i<self.double:
                mods=[]
                for which in ("img","txt"):
                    mods.append(np.split(w.linear(silu(c),prefix+".modulation_"+which+".1"),6,axis=-1))
                norms=[]
                for value,mod in zip((x,text),mods):
                    shift,scale=mod[:2]
                    norms.append(layernorm(value)*(1+scale)+shift)
                qkv=[]
                for value,which in zip(norms,("img","txt")):
                    q,k,v=self.qkv(value,prefix+".attn.qkv_"+which)
                    q=rms(q,w.get(prefix+".attn.q_norm_"+which+".weight"))
                    k=rms(k,w.get(prefix+".attn.k_norm_"+which+".weight"))
                    qkv.append((q,k,v))
                q,k,v=[np.concatenate([qkv[0][j],qkv[1][j]],axis=0) for j in range(3)]
            else:
                params=np.split(w.linear(silu(c),prefix+".modulation.1"),3,axis=-1)
                shift,scale,gate=params
                combined=np.concatenate([x,text],axis=0)
                q,k,v=self.qkv(layernorm(combined)*(1+scale)+shift,prefix+".attn.qkv")
                q=rms(q,w.get(prefix+".attn.q_norm.weight"))
                k=rms(k,w.get(prefix+".attn.k_norm.weight"))
            q=rotate(q,cosine,sine).transpose(1,0,2)
            k=rotate(k,cosine,sine).transpose(1,0,2)
            attended=attention(q,k,v.transpose(1,0,2),w.matmul).transpose(1,0,2).reshape(-1,self.hidden)
            if i<self.double:
                output=[]
                for value,part,mod,which in zip((x,text),np.split(attended,[image_tokens]),mods,("img","txt")):
                    shift_sa,scale_sa,gate_sa,shift_ff,scale_ff,gate_ff=mod
                    projected=w.linear(part,prefix+".attn.proj_"+which)
                    value=value+gate_sa*rms(projected,w.get(prefix+".norm_msa_"+which+"_out.weight"))
                    ff=self.feedforward(layernorm(value)*(1+scale_ff)+shift_ff,prefix+".mlp_"+which)
                    value=value+gate_ff*rms(ff,w.get(prefix+".norm_mlp_"+which+"_out.weight"))
                    output.append(value)
                x,text=output
            else:
                projected=w.linear(attended,prefix+".attn.proj")
                combined=combined+gate*rms(projected,w.get(prefix+".norm_msa_out.weight"))
                ff=self.feedforward(layernorm(combined)*(1+scale)+shift,prefix+".mlp")
                combined=combined+gate*rms(ff,w.get(prefix+".norm_mlp_out.weight"))
                x,text=np.split(combined,[image_tokens])
            self.timing.append(dict(block=i,seconds=time.perf_counter()-started))
            if trace is not None:
                trace(i,x,text)
            reclaim()
        output=w.linear(layernorm(x),"final_layer.1")
        return output.reshape(height//2,width//2,2,2,16).transpose(4,0,2,1,3).reshape(1,16,height,width)


class QwenArray:
    """Qwen3 prefill through layer 20, before final normalization, without KV cache."""
    def __init__(self,directory,threads=4,backend="ncnn"):
        self.weights=Weights(directory,threads,backend)
        self.embedding=np.load(Path(directory)/"embedding.npy",mmap_mode="r")
        self.config=self.weights.manifest["config"]

    def encode(self,ids):
        w=self.weights
        config=self.config
        heads=config["num_attention_heads"]
        kv_heads=config["num_key_value_heads"]
        dim=config["head_dim"]
        hidden=self.embedding[np.asarray(ids,dtype=np.int32)].astype(np.float32)
        inverse=1/(np.float32(config["rope_theta"])**(np.arange(0,dim,2,dtype=np.float32)/dim))
        phase=np.arange(len(ids),dtype=np.float32)[:,None]*inverse
        cosine=np.cos(np.concatenate([phase,phase],axis=-1))[:,None,:]
        sine=np.sin(np.concatenate([phase,phase],axis=-1))[:,None,:]
        for index in range(self.weights.manifest["exit_layer"]):
            prefix=f"model.layers.{index}"
            x=rms(hidden,w.get(prefix+".input_layernorm.weight"),config["rms_norm_eps"])
            q=w.linear(x,prefix+".self_attn.q_proj").reshape(len(ids),heads,dim)
            k=w.linear(x,prefix+".self_attn.k_proj").reshape(len(ids),kv_heads,dim)
            v=w.linear(x,prefix+".self_attn.v_proj").reshape(len(ids),kv_heads,dim)
            q=rms(q,w.get(prefix+".self_attn.q_norm.weight"),config["rms_norm_eps"])
            k=rms(k,w.get(prefix+".self_attn.k_norm.weight"),config["rms_norm_eps"])
            def rotary(value):
                a,b=np.split(value,2,axis=-1)
                return value*cosine+np.concatenate([-b,a],axis=-1)*sine
            q=rotary(q).transpose(1,0,2)
            k=np.repeat(rotary(k),heads//kv_heads,axis=1).transpose(1,0,2)
            v=np.repeat(v,heads//kv_heads,axis=1).transpose(1,0,2)
            attended=attention(q,k,v,w.matmul,causal=True).transpose(1,0,2).reshape(len(ids),heads*dim)
            hidden=hidden+w.linear(attended,prefix+".self_attn.o_proj")
            x=rms(hidden,w.get(prefix+".post_attention_layernorm.weight"),config["rms_norm_eps"])
            projected=silu(w.linear(x,prefix+".mlp.gate_proj"))*w.linear(x,prefix+".mlp.up_proj")
            hidden=hidden+w.linear(projected,prefix+".mlp.down_proj")
        retained=hidden[38:2086]
        if len(retained)==0:
            retained=np.zeros((1,config["hidden_size"]),dtype=np.float32)
        return retained[None,:,:],np.mean(retained,axis=0,keepdims=True)
