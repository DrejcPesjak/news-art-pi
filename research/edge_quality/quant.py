"""Portable symmetric grouped weight quantization and exact unpacking.

Scales are inverse step sizes, matching NCNN's block-quantized Gemm convention.
Only weights are quantized; inference activations and solver state remain FP32.
"""
import numpy as np


def quantize(weight, bits=4, group=64, mse=False):
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or bits not in (4, 8) or group < 1:
        raise ValueError("expected a matrix, 4/8 bits and a positive group size")
    rows, columns = weight.shape
    padded = ((columns + group - 1) // group) * group
    blocks = np.pad(weight, ((0, 0), (0, padded-columns))).reshape(rows, -1, group)
    limit = 2 ** (bits-1)-1
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scale = np.divide(limit, maximum, out=np.ones_like(maximum), where=maximum != 0)
    if mse:
        best_error = np.full(maximum.shape, np.inf, dtype=np.float32)
        best_scale = scale.copy()
        for fraction in (1., .98, .95, .9, .85, .8):
            candidate = scale / fraction
            quantized = np.clip(np.rint(blocks*candidate), -limit, limit)
            error = np.mean((blocks-quantized/candidate)**2, axis=-1, keepdims=True)
            better = error < best_error
            best_error = np.where(better, error, best_error)
            best_scale = np.where(better, candidate, best_scale)
        scale = best_scale
    integers = np.clip(np.rint(blocks*scale), -limit, limit).astype(np.int8).reshape(rows, padded)[:, :columns]
    if bits == 4:
        if columns % 2:
            integers = np.pad(integers, ((0, 0), (0, 1)))
        packed = (integers[:, 0::2].astype(np.uint8) & 15) | ((integers[:, 1::2].astype(np.uint8) & 15) << 4)
    else:
        packed = integers.view(np.uint8)
    return np.ascontiguousarray(packed), scale.squeeze(-1).copy()


def dequantize(packed, scales, shape, bits=4, group=64):
    rows, columns = shape
    if bits == 4:
        result = np.empty((rows, columns), dtype=np.float32)
        # Row chunks avoid materializing several full-size unpacking temporaries.
        for start in range(0,rows,128):
            part=packed[start:start+128]
            result[start:start+128,0::2]=((part & 15).astype(np.int8)^8)-8
            result[start:start+128,1::2]=((part[:,:columns//2] >> 4).astype(np.int8)^8)-8
    elif bits == 8:
        result = packed.view(np.int8)[:, :columns].astype(np.float32)
    else:
        raise ValueError("unsupported bits")
    if columns%group==0:
        # Traverse contiguous groups in row order. Visiting one column group
        # across every row repeatedly causes poor cache locality on large MLPs.
        result.reshape(rows,-1,group)[:] /= scales[:,:,None]
    else:
        for start in range(0, columns, group):
            result[:, start:start+group] /= scales[:, start//group, None]
    return result
