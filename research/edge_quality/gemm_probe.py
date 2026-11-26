"""Measure portable GEMM paths, including constant-weight setup each call."""
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import time

import numpy as np

from .array_runtime import Matmul


def constant(a,bt,threads=4):
    import ncnn
    n,k=bt.shape
    net=ncnn.Net()
    net.opt.num_threads=threads
    net.opt.use_vulkan_compute=False
    net.opt.use_packing_layout=False
    net.opt.use_fp16_arithmetic=False
    net.opt.use_fp16_storage=False
    net.opt.use_fp16_packed=False
    graph=f"7767517\n2 2\nInput a 0 1 a\nGemm gemm 1 1 a out 3=1 5=1 6=1 8={n} 9={k} 10=-1\n"
    if net.load_param_mem(graph)!=0:
        raise RuntimeError("Could not load constant GEMM")
    data=struct.pack("<I",0)+bt.astype(np.float32,copy=False).tobytes()
    net.load_model_mem(data)
    with net.create_extractor() as ex:
        ex.input("a",ncnn.Mat(np.ascontiguousarray(a)))
        code,out=ex.extract("out")
        if code:
            raise RuntimeError(f"Constant GEMM failed: {code}")
        return np.asarray(out).copy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=["dynamic","constant","numpy"],default="dynamic")
    args=parser.parse_args()
    random=np.random.default_rng(11)
    runtime=Matmul(4,"numpy" if args.backend=="numpy" else "ncnn")
    records=[]
    for rows,columns,inner in ((576,3456,1152),(576,6144,1152),(576,1152,3072)):
        a=random.normal(size=(rows,inner)).astype(np.float32)
        bt=random.normal(size=(columns,inner)).astype(np.float32)
        times=[]
        for _ in range(3):
            runtime.release_workspace()
            start=time.perf_counter()
            value=constant(a,bt) if args.backend=="constant" else runtime(a,bt)
            times.append(time.perf_counter()-start)
        record=dict(backend=args.backend,blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            shape=[rows,columns,inner],median_seconds=statistics.median(times),
            checksum=float(value.sum(dtype=np.float64)))
        records.append(record)
        print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
