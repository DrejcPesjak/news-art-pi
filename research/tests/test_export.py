import unittest
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


if __name__ == "__main__":
    unittest.main()
