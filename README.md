# deepstream-roi-blur-plugin

A CUDA-accelerated ROI (Region of Interest) blur library for NVIDIA DeepStream pipelines. Applies Gaussian-approximated blur to detected object bounding boxes **directly on GPU memory (NVMM)** — no CPU round-trip, no OpenCV dependency.

## Key Features

- **Zero-copy NVMM operation** — reads and writes `NvBufSurface.dataPtr` (GPU device memory) directly via CUDA kernels. The buffer never leaves GPU memory.
- **Separable box blur** — 3-pass separable box blur approximates Gaussian blur (Central Limit Theorem) with O(1) per-pixel cost regardless of kernel size.
- **ROI-only processing** — only the bounding box region is blurred, not the entire frame. Minimal GPU overhead per object.
- **DeepStream probe integration** — designed to be called from a `GstPadProbe` callback via Python `ctypes`. Compatible with any DeepStream Python pipeline.
- **Split compilation** — CUDA kernels (`nvcc`) and GStreamer glue code (`gcc`) are compiled separately to avoid GLib/CUDA header conflicts.

## NVMM Compliance

This library is fully NVMM-compliant. Here is how the data flow works:

```
DeepStream Pipeline (NVMM buffer on GPU)
  │
  ├─ GstPadProbe callback (Python)
  │    │
  │    ├─ hash(gst_buffer) → buffer address
  │    ├─ ctypes call → roi_blur_apply()
  │    │    │
  │    │    ├─ gst_buffer_map() → NvBufSurface* (metadata only, NOT pixel data)
  │    │    ├─ Read sp->dataPtr (GPU pointer), sp->pitch, sp->width, sp->height
  │    │    ├─ gst_buffer_unmap()
  │    │    │
  │    │    └─ roi_blur_cuda(gpu_ptr, ...)
  │    │         ├─ CUDA kernel blur_h<<<>>> (GPU → GPU temp buffer)
  │    │         ├─ CUDA kernel blur_v<<<>>> (GPU temp buffer → GPU)
  │    │         └─ cudaStreamSynchronize(0)
  │    │
  │    └─ Return to pipeline
  │
  └─ Pipeline continues (NVMM buffer still on GPU, now with blurred ROI)
```

**What makes it NVMM-compliant:**
1. `gst_buffer_map()` only accesses the `NvBufSurface` **metadata struct** (surface list, dimensions, GPU pointer address) — it does NOT copy pixel data to CPU.
2. `sp->dataPtr` is a **GPU device pointer** (`NVBUF_MEM_CUDA_DEVICE` / `NVBUF_MEM_DEFAULT`). The CUDA kernels read/write this pointer directly.
3. No `NvBufSurfaceMap()` (which would map pixels to CPU) is ever called.
4. No `cudaMemcpy` to/from host is ever performed.
5. The temporary buffer (`g_tmp`) is allocated with `cudaMalloc` and lives entirely on GPU.

## Requirements

- NVIDIA GPU (dGPU with CUDA support)
- NVIDIA DeepStream SDK 6.x / 7.x
- CUDA Toolkit (nvcc)
- GStreamer 1.0 development headers
- Python 3.x with `pyds` (DeepStream Python bindings)

## Build

```bash
cd src
make
```

This produces `libroi_blur.so`. The build uses split compilation:
- `roi_blur_kernels.cu` → compiled with `nvcc` (no GLib headers)
- `roi_blur_gst.c` → compiled with `gcc` (GLib/GStreamer headers)
- Linked together into a single shared library

## Usage

### Python Integration (ctypes)

```python
import ctypes

# Load the library
blur_lib = ctypes.CDLL("./src/libroi_blur.so")
blur_lib.roi_blur_apply.argtypes = [
    ctypes.c_ulonglong,  # gst_buffer address (hash(gst_buffer))
    ctypes.c_int,        # batch_id
    ctypes.c_int,        # roi_x
    ctypes.c_int,        # roi_y
    ctypes.c_int,        # roi_w
    ctypes.c_int,        # roi_h
    ctypes.c_int,        # kernel_size (e.g., 21)
    ctypes.c_int,        # passes (e.g., 3 for Gaussian approximation)
]
blur_lib.roi_blur_apply.restype = ctypes.c_int
```

### DeepStream Pad Probe Example

```python
def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    buf_addr = hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buf_addr)

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        batch_id = frame_meta.batch_id

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            rect = obj_meta.rect_params

            # Apply blur with padding
            pad_ratio = 0.2
            roi_x = int(rect.left - rect.width * pad_ratio / 2)
            roi_y = int(rect.top - rect.height * pad_ratio / 2)
            roi_w = int(rect.width * (1 + pad_ratio))
            roi_h = int(rect.height * (1 + pad_ratio))

            blur_lib.roi_blur_apply(
                ctypes.c_ulonglong(buf_addr), batch_id,
                roi_x, roi_y, roi_w, roi_h,
                21,  # kernel_size
                3,   # passes (3-pass box blur ≈ Gaussian)
            )

            l_obj = l_obj.next
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK
```

### Pipeline Requirements

The buffer must be in **RGBA format** when the probe is called. Insert a `capsfilter` before `nvdsosd`:

```
... → nvvideoconvert → capsfilter(video/x-raw(memory:NVMM),format=RGBA) → nvdsosd → ...
```

## API Reference

### `roi_blur_apply(gst_buf_addr, batch_id, roi_x, roi_y, roi_w, roi_h, kernel_size, passes)`

Applies blur to a rectangular ROI on the GPU buffer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gst_buf_addr` | `uint64` | GstBuffer address from `hash(gst_buffer)` |
| `batch_id` | `int` | Frame index within the batch |
| `roi_x` | `int` | ROI left coordinate (can be negative, will be clamped) |
| `roi_y` | `int` | ROI top coordinate (can be negative, will be clamped) |
| `roi_w` | `int` | ROI width |
| `roi_h` | `int` | ROI height |
| `kernel_size` | `int` | Blur kernel size (e.g., 21). Larger = stronger blur |
| `passes` | `int` | Number of box blur passes. 3 passes ≈ Gaussian blur |

**Returns:** `0` on success, `-1` on error.

### `roi_blur_free()`

Releases the internal GPU temporary buffer. Call on shutdown if needed.

## How It Works

The blur uses a **3-pass separable box filter**, which is a well-known approximation of Gaussian blur:

1. **Horizontal pass**: Each pixel is replaced by the average of its horizontal neighbors within the kernel radius
2. **Vertical pass**: The result is averaged vertically

This is repeated `passes` times (default 3). By the Central Limit Theorem, repeated convolution with a box kernel converges to a Gaussian kernel.

The separable approach means the complexity is O(1) per pixel regardless of kernel size, since we compute running averages.

## Performance

- **No CPU ↔ GPU data transfer** — the entire operation stays on GPU
- **ROI-only computation** — only the detected bounding box area is processed
- **Efficient memory use** — single persistent temp buffer, lazily allocated and reused across frames
- **Kernel launch overhead** is minimal (~2 microseconds per launch × 2 kernels × passes)

## License

MIT License. See [LICENSE](LICENSE) for details.
