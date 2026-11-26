"""Compare channel-batched NCNN MatMul with separate optimized head GEMMs."""
import json
import statistics
import time

import numpy as np
import ncnn

from .array_runtime import Matmul


def net(transpose):
    model=ncnn.Net()
    model.opt.num_threads=4
    model.opt.use_packing_layout=False
    model.opt.use_vulkan_compute=False
    graph=f"7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nMatMul mm 2 1 a b out 0={int(transpose)}\n"
    if model.load_param_mem(graph)!=0:
        raise RuntimeError("MatMul graph unavailable")
    model.load_model_mem(b"")
    return model


def batch(model,a,b):
    with model.create_extractor() as ex:
        ex.input("a",ncnn.Mat(np.ascontiguousarray(a)).clone())
        ex.input("b",ncnn.Mat(np.ascontiguousarray(b)).clone())
        code,out=ex.extract("out")
        if code:
            raise RuntimeError(f"Batched MatMul failed: {code}")
        return np.asarray(out).copy()


def main():
    random=np.random.default_rng(8)
    q=random.normal(size=(16,128,72)).astype(np.float32)
    k=random.normal(size=(16,600,72)).astype(np.float32)
    v=random.normal(size=(16,600,72)).astype(np.float32)
    direct=Matmul(4)
    qk=net(True)
    pv=net(False)
    expected=q@k.transpose(0,2,1)
    functions={"head_gemm":lambda:np.stack([direct(q[h],k[h]) for h in range(16)]),
               "batch_matmul":lambda:batch(qk,q,k),"numpy_batch":lambda:q@k.transpose(0,2,1)}
    scores=np.abs(expected);scores/=scores.sum(-1,keepdims=True)
    functions.update(head_values=lambda:np.stack([direct(scores[h],v[h].T) for h in range(16)]),
                     batch_values=lambda:batch(pv,scores,v),numpy_values=lambda:scores@v)
    for name,fn in functions.items():
        times=[]
        for _ in range(5):
            start=time.perf_counter();actual=fn();times.append(time.perf_counter()-start)
        target=scores@v if "values" in name else expected
        print(json.dumps(dict(method=name,median_seconds=statistics.median(times),
              relative_rmse=float(np.linalg.norm(actual-target)/np.linalg.norm(target)))),flush=True)


if __name__=="__main__":
    main()
