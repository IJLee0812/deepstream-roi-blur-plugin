/*
 * roi_blur_kernels.cu
 *
 * CUDA separable box-blur kernels.
 * No GLib/GStreamer headers - compiles standalone with nvcc.
 */

#include <cuda_runtime.h>
#include <stdio.h>

/* ================================================================
 *  Persistent temp buffer (lazy-allocated, grows as needed)
 * ================================================================ */
static unsigned char *g_tmp    = NULL;
static size_t         g_tmp_sz = 0;

static int ensure_tmp(size_t need)
{
    if (g_tmp && g_tmp_sz >= need) return 0;
    if (g_tmp) { cudaFree(g_tmp); g_tmp = NULL; g_tmp_sz = 0; }
    cudaError_t e = cudaMalloc(&g_tmp, need);
    if (e != cudaSuccess) {
        fprintf(stderr, "[roi_blur] cudaMalloc(%zu): %s\n",
                need, cudaGetErrorString(e));
        return -1;
    }
    g_tmp_sz = need;
    return 0;
}

/* ================================================================
 *  CUDA Kernels – separable box blur (horizontal / vertical)
 * ================================================================ */

__global__ void blur_h(const unsigned char * __restrict__ src,
                       unsigned char       * __restrict__ dst,
                       int pitch, int roi_x, int roi_y,
                       int roi_w, int roi_h, int radius)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= roi_w || ly >= roi_h) return;

    int x = lx + roi_x;
    int y = ly + roi_y;

    int x0 = max(roi_x, x - radius);
    int x1 = min(roi_x + roi_w - 1, x + radius);
    int n  = x1 - x0 + 1;

    float r = 0.f, g = 0.f, b = 0.f;
    const unsigned char *row = src + y * pitch;
    for (int i = x0; i <= x1; ++i) {
        int o = i * 4;
        r += row[o]; g += row[o + 1]; b += row[o + 2];
    }

    unsigned char *out = dst + y * pitch + x * 4;
    out[0] = (unsigned char)(r / n);
    out[1] = (unsigned char)(g / n);
    out[2] = (unsigned char)(b / n);
    out[3] = src[y * pitch + x * 4 + 3];
}

__global__ void blur_v(const unsigned char * __restrict__ src,
                       unsigned char       * __restrict__ dst,
                       int pitch, int roi_x, int roi_y,
                       int roi_w, int roi_h, int radius)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= roi_w || ly >= roi_h) return;

    int x = lx + roi_x;
    int y = ly + roi_y;

    int y0 = max(roi_y, y - radius);
    int y1 = min(roi_y + roi_h - 1, y + radius);
    int n  = y1 - y0 + 1;

    float r = 0.f, g = 0.f, b = 0.f;
    for (int j = y0; j <= y1; ++j) {
        const unsigned char *p = src + j * pitch + x * 4;
        r += p[0]; g += p[1]; b += p[2];
    }

    unsigned char *out = dst + y * pitch + x * 4;
    out[0] = (unsigned char)(r / n);
    out[1] = (unsigned char)(g / n);
    out[2] = (unsigned char)(b / n);
    out[3] = src[y * pitch + x * 4 + 3];
}

/* ================================================================
 *  Internal C API – called from roi_blur_gst.c
 * ================================================================ */
extern "C" {

int roi_blur_cuda(unsigned char *gpu, int P, int W, int H,
                  int roi_x, int roi_y, int roi_w, int roi_h,
                  int kernel_size, int passes)
{
    /* Clamp ROI to frame bounds */
    if (roi_x < 0) { roi_w += roi_x; roi_x = 0; }
    if (roi_y < 0) { roi_h += roi_y; roi_y = 0; }
    if (roi_x + roi_w > W) roi_w = W - roi_x;
    if (roi_y + roi_h > H) roi_h = H - roi_y;
    if (roi_w <= 0 || roi_h <= 0) return 0;

    if (ensure_tmp((size_t)H * P)) return -1;

    int radius = kernel_size / 2;
    dim3 blk(16, 16);
    dim3 grd((roi_w + 15) / 16, (roi_h + 15) / 16);

    for (int p = 0; p < passes; ++p) {
        blur_h<<<grd, blk>>>(gpu,   g_tmp, P, roi_x, roi_y, roi_w, roi_h, radius);
        blur_v<<<grd, blk>>>(g_tmp, gpu,   P, roi_x, roi_y, roi_w, roi_h, radius);
    }

    cudaError_t err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[roi_blur] sync: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

void roi_blur_free(void)
{
    if (g_tmp) { cudaFree(g_tmp); g_tmp = NULL; g_tmp_sz = 0; }
}

}  /* extern "C" */
