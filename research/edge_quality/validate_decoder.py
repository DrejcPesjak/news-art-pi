"""Validate the bounded split/tiled decoder against independent FP32 fixtures."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import resource
import time

os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np

from .common import OUTPUT,save_json
from .decoder_runtime import DecoderPart


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--latent-size",type=int,default=48)
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_DATA,(480*2**20,480*2**20))
    directory=OUTPUT/"decoder"
    fixture=np.load(directory/f"fixture-{args.latent_size}.npz")
    # Decoder-parts exports use the same final random fixture as the full export.
    x=fixture["latent"]
    start=time.perf_counter()
    for part in json.loads((directory/f"parts-{args.latent_size}.json").read_text()):
        print("decoder",part["name"],list(x.shape),flush=True)
        model=DecoderPart(directory/part["prefix"],threads=4)
        name=part["name"]
        kind=part.get("kind") or ("resize" if name.endswith("_resize") else "residual" if "_res" in name else name)
        x=model.tiled(x,kind,part["output_shape"][1],rows=32)
        del model
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
    expected=fixture["expected"]
    error=x-expected
    report=dict(latent_size=args.latent_size,seconds=time.perf_counter()-start,
        relative_rmse=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(expected**2))),
        max_abs=float(np.max(np.abs(error))),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        allocation_limit_mib=480,full_frame_attention=True,exact_local_halos=True)
    save_json(directory/f"bounded-validation-{args.latent_size}.json",report)
    print(json.dumps(report),flush=True)
    if report["relative_rmse"]>.002:
        raise RuntimeError("Bounded decoder differs from the independent FP32 fixture")


if __name__=="__main__":
    main()
