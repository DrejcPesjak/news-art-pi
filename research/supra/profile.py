"""Run one exported component under an external memory/time limit on the Pi."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
from .ncnn_runtime import NCNNModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("component",choices=["dit","text","vae","vae_tile20"])
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=3)
    args=parser.parse_args()
    fixture=np.load(args.model_dir/(args.component+"-fixture.npz"))
    inputs=[fixture[name] for name in sorted(fixture.files) if name.startswith("in")]
    start=time.perf_counter()
    model=NCNNModel(args.model_dir/args.component,threads=args.threads)
    loaded=time.perf_counter()
    actual=model(*inputs,scalar_inputs=(1,) if args.component=="dit" else ())
    done=time.perf_counter()
    expected=fixture["expected"]
    rmse=float(np.sqrt(np.mean((actual-expected)**2)))
    relative=rmse/max(float(np.sqrt(np.mean(expected**2))),1e-8)
    report={"component":args.component,"machine":platform.machine(),"threads":args.threads,
            "load_seconds":loaded-start,"forward_seconds":done-loaded,
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "relative_rmse":relative,"max_abs_error":float(np.max(np.abs(actual-expected))),
            "passed":relative<.01,"note":"One component forward, not a complete generated image."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise ValueError("On-device numerical validation failed")


if __name__=="__main__":main()
