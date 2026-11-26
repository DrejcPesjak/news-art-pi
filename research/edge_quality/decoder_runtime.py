"""NCNN decoder parts without redundant batch-stack copies."""
import numpy as np

from research.supra.ncnn_runtime import NCNNModel


class DecoderPart(NCNNModel):
    def __init__(self,prefix,threads=4):
        # Unpacked channel planes expose a strided NumPy view directly. The
        # binding's numpy(batch_index=0) stacks/copies those planes, then the
        # generic runtime copies again; at 384px those extra arrays are large.
        super().__init__(prefix,threads=threads,packing=False)

    def __call__(self,value):
        import ncnn
        array=np.ascontiguousarray(value,dtype=np.float32)
        if array.ndim!=4 or array.shape[0]!=1:
            raise ValueError("Decoder supports batch size one")
        with self.net.create_extractor() as extractor:
            # Residual graphs may use in-place layers. Give NCNN an owned,
            # aligned input with a real reference count before those layers run.
            if extractor.input("in0",ncnn.Mat(array[0]).clone())!=0:
                raise RuntimeError("NCNN decoder input failed")
            code,output=extractor.extract("out0")
            if code!=0:
                raise RuntimeError(f"NCNN decoder failed: {code}")
            if output.elempack!=1:
                raise RuntimeError("Expected unpacked decoder output")
            # Copy once while the extractor/allocator still own the Mat. The
            # returned NumPy allocation survives release of this model part.
            result=np.asarray(output).copy()[None,...]
        if not np.isfinite(result).all():
            raise ValueError("NCNN decoder returned non-finite pixels")
        return result

    def tiled(self,value,kind,output_channels,rows=64):
        """Exact row tiles for local decoder parts, after full-frame attention.

        Residual blocks contain two spatial 3x3 convolutions (halo 2). Resize
        parts double by nearest interpolation then convolve 3x3 (input halo 1).
        The output head has one 3x3 convolution (halo 1). Channel normalization
        is pixel-local. No attention or spatial normalization is tiled.
        """
        if kind not in ("residual","resize","head") or value.shape[2]<=rows:
            return self(value)
        scale=2 if kind=="resize" else 1
        halo=2 if kind=="residual" else 1
        # Some platform convolution kernels use an im2col workspace even when
        # transform optimizations are disabled. Bound the enlarged resize tile.
        if kind=="resize":
            rows=min(rows,16)
        height,width=value.shape[-2:]
        result=np.empty((1,output_channels,height*scale,width*scale),np.float32)
        for start in range(0,height,rows):
            end=min(height,start+rows)
            left=max(0,start-halo)
            right=min(height,end+halo)
            tile=self(value[:,:,left:right,:])
            offset=(start-left)*scale
            result[:,:,start*scale:end*scale,:]=tile[:,:,offset:offset+(end-start)*scale,:]
        return result
