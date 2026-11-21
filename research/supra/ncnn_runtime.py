"""NumPy + NCNN only: importable on the Pi without PyTorch."""
from pathlib import Path
import numpy as np
import ncnn


class NCNNModel:
    def __init__(self, prefix, threads=3):
        prefix = str(prefix)
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_bf16_storage = False
        self.net.opt.use_int8_inference = False
        self.net.opt.use_packing_layout = False
        # Winograd/SGEMM convolution transforms made the VAE peak at 1.64 GiB
        # on x86. Direct convolution trades speed for a Pi-compatible workspace.
        self.net.opt.use_winograd_convolution = False
        self.net.opt.use_winograd23_convolution = False
        self.net.opt.use_winograd43_convolution = False
        self.net.opt.use_winograd63_convolution = False
        self.net.opt.use_sgemm_convolution = False
        if self.net.load_param(prefix+".ncnn.param") != 0:
            raise RuntimeError(f"NCNN could not load graph {prefix}")
        if self.net.load_model(prefix+".ncnn.bin") != 0:
            raise RuntimeError(f"NCNN could not load weights {prefix}")

    def __call__(self, *inputs, scalar_inputs=()):
        with self.net.create_extractor() as extractor:
            for index, value in enumerate(inputs):
                array = np.ascontiguousarray(value, dtype=np.float32)
                batch_index = 233 if index in scalar_inputs else 0
                result = extractor.input(f"in{index}", ncnn.Mat(array, batch_index=batch_index).clone())
                if result != 0:
                    raise RuntimeError(f"NCNN input {index} failed: {result}")
            result, output = extractor.extract("out0")
            if result != 0:
                raise RuntimeError(f"NCNN inference failed: {result}")
            array = output.numpy(batch_index=0).copy()
            if not np.isfinite(array).all():
                raise ValueError("NCNN returned non-finite values")
            return array
