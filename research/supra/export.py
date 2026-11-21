"""Export only verified components; failed numerical checks are explicit errors."""
import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import pnnx

from .assets import MODEL_DIR, ROOT
from .exportable import ExportDiT
from .model import SupraDiT
from .ncnn_runtime import NCNNModel


def load_dit():
    state = torch.load(MODEL_DIR/"supra/model_final_ema.pt", map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        model = SupraDiT()
    model.load_state_dict(state.get("ema", state.get("model", state)), assign=True, strict=True)
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["dit"])
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/supra/ncnn")
    parser.add_argument("--fp16-storage", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    prefix = args.output / args.component
    original = load_dit()
    checkpoint = torch.load(MODEL_DIR/"supra/model_final_ema.pt", map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["config"]
    # Use an actual encoder embedding, not unit-variance Gaussian context (the
    # checkpoint's text embeddings have a much smaller scale).
    inputs = (torch.randn(1,4,32,32), torch.tensor([.5]),
              config["uncond_text"].unsqueeze(0).float(),
              config["uncond_mask"].unsqueeze(0).float())
    with torch.inference_mode():
        reference = original(*inputs).numpy()
        model = ExportDiT(original).eval()
        layout = model(*inputs).numpy()
    layout_error = float(np.max(np.abs(reference-layout)))
    layout_relative = float(np.sqrt(np.mean((reference-layout)**2) / np.mean(reference**2)))
    # The trained model amplifies float32 accumulation differences: CPU flash
    # SDPA vs PyTorch's own math SDPA also differs (~0.3% on this fixture).
    # Record that difference separately from converter error; do not call it exact.
    if layout_relative > .01:
        raise ValueError(f"Layout rewrite changed model output: {layout_error}")
    logging.info("Layout rewrite max error: %s", layout_error)
    pnnx.export(model, str(prefix)+".pt", inputs, check_trace=False, fp16=args.fp16_storage)
    net = NCNNModel(prefix, threads=4)
    start = time.perf_counter()
    actual = net(*(value.numpy() for value in inputs), scalar_inputs=(1,))
    seconds = time.perf_counter()-start
    absolute = float(np.max(np.abs(layout-actual)))
    rmse = float(np.sqrt(np.mean((layout-actual)**2)))
    relative = rmse / max(float(np.sqrt(np.mean(layout**2))), 1e-8)
    result = {"component": args.component, "layout_max_abs_error": layout_error,
              "fixture": "checkpoint unconditional text embedding; conditional validation required separately",
              "layout_relative_rmse": layout_relative,
              "ncnn_max_abs_error": absolute, "ncnn_rmse": rmse,
              "ncnn_relative_rmse": relative, "x86_forward_seconds": seconds,
              "fp16_storage": args.fp16_storage,
              "weights_mib": Path(str(prefix)+".ncnn.bin").stat().st_size/2**20,
              "passed": relative < (.01 if args.fp16_storage else .0001)}
    (args.output/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez(args.output/"dit-fixture.npz", **{f"in{i}": value.numpy() for i,value in enumerate(inputs)}, expected=reference)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise ValueError("NCNN numerical validation failed; do not deploy these weights")


if __name__ == "__main__":
    main()
