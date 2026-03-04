/*
 * roi_blur_gst.c
 *
 * Extracts NvBufSurface GPU pointer from a GstBuffer and
 * invokes the CUDA blur function defined in roi_blur_kernels.cu.
 *
 * Compiled with gcc (separate from nvcc) to avoid GLib/CUDA header conflicts.
 */

#include <stdio.h>
#include <string.h>

#include <gst/gst.h>
#include "nvbufsurface.h"

int  roi_blur_cuda(unsigned char *gpu, int pitch, int W, int H,
                   int roi_x, int roi_y, int roi_w, int roi_h,
                   int kernel_size, int passes);
void roi_blur_free(void);

/* ================================================================
 *  Public API - called from Python via ctypes
 * ================================================================ */

int roi_blur_apply(unsigned long long gst_buf_addr, int batch_id,
                   int roi_x, int roi_y, int roi_w, int roi_h,
                   int kernel_size, int passes)
{
    GstBuffer  *buffer = (GstBuffer *)(void *)(uintptr_t)gst_buf_addr;
    GstMapInfo  map;
    memset(&map, 0, sizeof(map));

    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        fprintf(stderr, "[roi_blur] gst_buffer_map failed\n");
        return -1;
    }

    NvBufSurface *surface = (NvBufSurface *)map.data;

    if (batch_id < 0 || batch_id >= (int)surface->numFilled) {
        gst_buffer_unmap(buffer, &map);
        return -1;
    }

    NvBufSurfaceParams *sp = &surface->surfaceList[batch_id];

    if (sp->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
        fprintf(stderr, "[roi_blur] expected RGBA, got %d\n", sp->colorFormat);
        gst_buffer_unmap(buffer, &map);
        return -1;
    }

    NvBufSurfaceMemType mt = surface->memType;
    if (mt != NVBUF_MEM_CUDA_DEVICE &&
        mt != NVBUF_MEM_CUDA_UNIFIED &&
        mt != NVBUF_MEM_DEFAULT) {
        fprintf(stderr, "[roi_blur] unsupported memType %d\n", mt);
        gst_buffer_unmap(buffer, &map);
        return -1;
    }

    unsigned char *gpu = (unsigned char *)sp->dataPtr;
    int W = (int)sp->width;
    int H = (int)sp->height;
    int P = (int)sp->pitch;

    gst_buffer_unmap(buffer, &map);

    if (!gpu) return -1;

    return roi_blur_cuda(gpu, P, W, H,
                         roi_x, roi_y, roi_w, roi_h,
                         kernel_size, passes);
}
