"""Format, bounded attention and tokenizer checks for the portable prototype."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from research.edge_quality.quant import quantize,dequantize
from research.edge_quality.array_runtime import Matmul,attention,image_rope,rotate


class QuantizationTests(unittest.TestCase):
    def test_known_signed_nibbles_and_group_scales(self):
        packed=np.array([[0xF1,0x87]],dtype=np.uint8)
        actual=dequantize(packed,np.array([[2]],dtype=np.float32),(1,4),4,4)
        np.testing.assert_array_equal(actual,[[.5,-.5,3.5,-4]])

    def test_roundtrip_odd_and_incomplete_groups(self):
        random=np.random.default_rng(91)
        for shape in ((1,1),(7,17),(129,65),(8,128)):
            for bits in (4,8):
                x=random.normal(size=shape).astype(np.float32)
                packed,scales=quantize(x,bits,64)
                actual=dequantize(packed,scales,shape,bits,64)
                error=np.abs(actual-x)
                bound=np.repeat(.5/scales,64,axis=1)[:,:shape[1]]
                self.assertTrue(np.all(error<=bound+1e-6))

    def test_zero_matrix_remains_finite(self):
        for bits in (4,8):
            x=np.zeros((3,65),np.float32)
            packed,scales=quantize(x,bits,64,mse=True)
            self.assertTrue(np.isfinite(scales).all())
            np.testing.assert_array_equal(dequantize(packed,scales,x.shape,bits,64),x)

    def test_mse_scale_search_does_not_increase_weight_error(self):
        x=np.random.default_rng(3).normal(size=(32,128)).astype(np.float32)
        a,s=quantize(x,4,64,mse=False)
        b,t=quantize(x,4,64,mse=True)
        self.assertLessEqual(np.mean((dequantize(b,t,x.shape)-x)**2),
                             np.mean((dequantize(a,s,x.shape)-x)**2)+1e-8)


class AttentionTests(unittest.TestCase):
    def test_query_chunking_matches_dense_causal_attention(self):
        random=np.random.default_rng(5)
        q,k,v=[random.normal(size=(2,17,8)).astype(np.float32) for _ in range(3)]
        scores=(q@k.transpose(0,2,1))/np.sqrt(np.float32(8))
        scores[:,np.triu_indices(17,k=1)[0],np.triu_indices(17,k=1)[1]]=-np.inf
        scores-=scores.max(axis=-1,keepdims=True)
        expected=np.exp(scores)
        expected/=expected.sum(axis=-1,keepdims=True)
        actual=attention(q,k,v,Matmul(backend="numpy"),causal=True,chunk=3)
        np.testing.assert_allclose(actual,expected@v,rtol=2e-6,atol=2e-6)

    def test_rope_rotation_preserves_vector_norm(self):
        x=np.random.default_rng(8).normal(size=(9,2,8)).astype(np.float32)
        cos,sin=image_rope(2,3,3,8)
        np.testing.assert_allclose(np.linalg.norm(rotate(x,cos,sin),axis=-1),
                                   np.linalg.norm(x,axis=-1),rtol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("ncnn"),"NCNN not installed")
    def test_dynamic_ncnn_gemm_shape_and_values(self):
        random=np.random.default_rng(41)
        runtime=Matmul(threads=2)
        for m,n,k in ((1,9,17),(7,5,64),(13,23,31),(3,4103,31)):
            a=random.normal(size=(m,k)).astype(np.float32)
            b=random.normal(size=(n,k)).astype(np.float32)
            np.testing.assert_allclose(runtime(a,b),a@b.T,rtol=1e-5,atol=1e-5)


class TokenizerTests(unittest.TestCase):
    @unittest.skipUnless(Path("models/alternatives/inko-text/tokenizer.json").exists(),"local Qwen assets unavailable")
    def test_ids_match_released_tokenizer(self):
        from research.edge_quality.qwen_tokenizer import QwenTokenizer,SYSTEM_PROMPT,PROMPT_TEMPLATE
        from transformers import AutoTokenizer
        path=Path("models/alternatives/inko-text")
        actual=QwenTokenizer(path/"tokenizer.json")
        expected=AutoTokenizer.from_pretrained(path,local_files_only=True)
        for text in ("", "A ship at sea. oil painting", "水墨山水画，远处有山。",
                     "café cafe\u0301 Slovenščina 12345!\n\n    a", "🔥 ship\tfoo\r\nbar"):
            formatted=PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT,user_prompt=text)
            self.assertEqual(actual.prompt(text),expected(formatted)["input_ids"])


class DecoderTileTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ncnn") and Path("artifacts/edge-quality/decoder/parts-48.json").exists(),
                         "exported decoder fixtures unavailable")
    def test_complete_halos_match_full_parts_at_odd_boundaries(self):
        import json
        from research.edge_quality.decoder_runtime import DecoderPart
        directory=Path("artifacts/edge-quality/decoder")
        parts=json.loads((directory/"parts-48.json").read_text())
        for part in parts:
            name=part["name"]
            if name not in ("up2_res0","up2_resize","head"):
                continue
            kind="resize" if name.endswith("_resize") else "residual" if "_res" in name else name
            model=DecoderPart(directory/part["prefix"],threads=2)
            x=np.random.default_rng(113).normal(size=(1,part["input_shape"][1],131,19)).astype(np.float32)
            expected=model(x)
            actual=model.tiled(x,kind,part["output_shape"][1],rows=31)
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5,err_msg=name)

    def test_runtime_imports_do_not_load_training_frameworks(self):
        import os
        import subprocess
        import sys
        command="import research.edge_quality.staged, research.edge_quality.array_runtime, sys; assert not any(x in sys.modules for x in ('torch','transformers','diffusers'))"
        subprocess.run([sys.executable,"-c",command],check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))


class TrainingCacheTests(unittest.TestCase):
    @unittest.skipUnless(Path("models/alternatives/inko-reviewed/src/models/inko.py").exists(),"pinned Inko source unavailable")
    def test_evaluation_cache_can_be_cleared_before_backward(self):
        import torch
        from research.edge_quality.inko_quant import imports
        from research.edge_quality.distill import clear_rope_caches
        _,Inko,_,_,_=imports()
        model=Inko(hidden_size=64,num_heads=4,double_stream_depth=1,single_stream_depth=2,mlp_ratio=2).eval()
        inputs=(torch.randn(1,16,8,8),torch.tensor([.5]),torch.randn(1,5,1024),torch.randn(1,1024))
        with torch.inference_mode():
            model(*inputs)
        clear_rope_caches(model)
        model(*inputs).square().mean().backward()
        self.assertIsNotNone(model.final_layer[1].weight.grad)


if __name__=="__main__":
    unittest.main()
