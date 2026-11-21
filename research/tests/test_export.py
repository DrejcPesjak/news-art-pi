import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from research.supra.model import SupraDiT
from research.supra.exportable import ExportDiT


class ExportMathTests(unittest.TestCase):
    def test_patch_and_attention_rewrite_with_multiple_heads_and_padding(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        model = SupraDiT(d_model=64, depth=2, n_heads=4).eval()
        values = (torch.randn(1, 4, 32, 32), torch.tensor([.25]),
                  torch.randn(1, 128, 768),
                  torch.cat([torch.ones(1, 17), torch.zeros(1, 111)], dim=1))
        with torch.inference_mode():
            expected = model(*values)
            rewritten = ExportDiT(model)(*values)
        self.assertEqual(tuple(rewritten.shape), (1, 4, 32, 32))
        torch.testing.assert_close(expected, rewritten, atol=3e-6, rtol=3e-6)

    def test_tiled_decoder_preserves_constant_brightness_and_covers_every_pixel(self):
        from research.supra.tiled_decode import decode
        class ConstantDecoder:
            def __init__(self,*args,**kwargs):pass
            def __call__(self,latent):return np.full((1,3,160,160),.37,dtype=np.float32)
        with patch("research.supra.tiled_decode.NCNNModel",ConstantDecoder):
            actual=decode(np.zeros((1,4,32,32),dtype=np.float32),Path("unused"))
        self.assertEqual(actual.shape,(1,3,256,256))
        np.testing.assert_allclose(actual,.37,atol=1e-6)

    def test_bundle_verification_rejects_changed_weights(self):
        from research.supra.bundle import verify
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/"weights.bin").write_bytes(b"abc")
            manifest={"weights.bin":{"bytes":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}
            (directory/"manifest.json").write_text(json.dumps(manifest))
            verify(directory)
            (directory/"weights.bin").write_bytes(b"xyz")
            with self.assertRaises(ValueError):verify(directory)


if __name__ == "__main__":
    unittest.main()
