import json
from pathlib import Path
import gc
import numpy as np
import torch

from .assets import ROOT
from .pipeline import Pipeline
from .ncnn_runtime import NCNNModel


def main():
    torch.set_num_threads(4)
    pipeline = Pipeline("baseline")
    prompts = json.loads((ROOT/"research/supra/prompts.json").read_text())
    fixtures = []
    generator = torch.Generator(device="cuda").manual_seed(0)
    z = torch.randn(1,4,32,32,device="cuda",generator=generator).cpu().numpy()
    for prompt in prompts:
        context, mask = pipeline.encode(prompt["prompt"])
        for index, name in [(0,prompt["id"]),(1,"unconditional")]:
            if name == "unconditional" and any(existing == name for existing, _ in fixtures):
                continue
            fixtures.append((name, (z, np.array([.5],dtype=np.float32),
                                     context[index:index+1].cpu().numpy(), mask[index:index+1].cpu().numpy())))
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    reference_model = torch.jit.load(str(ROOT/"artifacts/supra/ncnn-fp32/dit.pt")).eval()
    results = []
    for folder, tolerance in [("ncnn-fp32",.0001),("ncnn",.01)]:
        directory = ROOT/"artifacts/supra"/folder
        model = NCNNModel(directory/"dit",threads=4)
        for name, values in fixtures:
            with torch.inference_mode():
                expected = reference_model(*(torch.from_numpy(value) for value in values)).numpy()
            actual = model(*values,scalar_inputs=(1,))
            rmse = float(np.sqrt(np.mean((expected-actual)**2)))
            relative = rmse/max(float(np.sqrt(np.mean(expected**2))),1e-8)
            results.append({"storage":folder,"context":name,"context_std":float(values[2].std()),
                            "rmse":rmse,"relative_rmse":relative,"passed":relative<tolerance})
            print(results[-1],flush=True)
            if name == prompts[0]["id"]:
                np.savez(directory/"dit-fixture.npz",**{f"in{i}":value for i,value in enumerate(values)},expected=expected)
        (directory/"validation.json").write_text(json.dumps({"fixtures":"four real prompts and checkpoint unconditional context",
              "results":[row for row in results if row["storage"]==folder],
              "passed":all(row["passed"] for row in results if row["storage"]==folder)},indent=2)+"\n")
        del model
        gc.collect()
    (ROOT/"research/results/ncnn-validation.json").write_text(json.dumps(results,indent=2)+"\n")
    if not all(row["passed"] for row in results):
        raise ValueError("Some conditional-context NCNN validation failed")


if __name__ == "__main__":
    main()
