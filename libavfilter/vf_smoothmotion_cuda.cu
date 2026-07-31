/*
 * Copyright (C) 2026 Philip Langdale <philipl@overt.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/*
 * Pack/unpack kernels for native-YUV support in vf_smoothmotion_cuda.
 *
 * The Smooth Motion network consumes two packed UNORM_INT8X4 input textures and
 * writes a packed UNORM_INT8X4 output surface.  To run it on YUV without any RGB
 * colorspace conversion we exploit that the inputs are read in only two kernels:
 *   - downscale (feeds the optical-flow backbone) gets a LUMA-grey texture
 *     (Y,Y,Y,255) so the RGB-trained net estimates flow from true luminance;
 *   - warp_coarse (renders the output by a per-channel flow-guided blend, which
 *     is colorspace-agnostic) gets the actual packed (Y,U,V,255) texture.
 * The output surface is then packed (Y,U,V,A) and we de-interleave it to planar
 * YUV444P.  4:2:0 inputs are chroma-upsampled here (YUV-internal bilinear, no RGB
 * matrix); the filter advertises a YUV444P output so there is no output-side
 * chroma downsample.
 *
 * These write to linear scratch buffers (the network I/O lives in CUDA arrays,
 * which the filter fills via cuMemcpy2D); pitches are in BYTES.
 */

#include "cuda/vector_helpers.cuh"

static inline __device__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline __device__ unsigned char to_u8(float v)
{
    int i = (int)(v + 0.5f);
    return (unsigned char)clampi(i, 0, 255);
}

/* bilinear sample of a single-byte plane (element stride 1 byte) */
static inline __device__ float bilin1(const unsigned char *p, int pitch,
                                      int cw, int ch, float fx, float fy)
{
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - x0, ay = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = clampi(x0, 0, cw - 1); x1 = clampi(x1, 0, cw - 1);
    y0 = clampi(y0, 0, ch - 1); y1 = clampi(y1, 0, ch - 1);
    float c00 = p[y0 * pitch + x0], c10 = p[y0 * pitch + x1];
    float c01 = p[y1 * pitch + x0], c11 = p[y1 * pitch + x1];
    return (c00 * (1 - ax) + c10 * ax) * (1 - ay) + (c01 * (1 - ax) + c11 * ax) * ay;
}

/* bilinear sample of an interleaved 2-byte plane (NV12 UV), byte offset off */
static inline __device__ float bilin2(const unsigned char *p, int pitch,
                                      int cw, int ch, float fx, float fy, int off)
{
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - x0, ay = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = clampi(x0, 0, cw - 1); x1 = clampi(x1, 0, cw - 1);
    y0 = clampi(y0, 0, ch - 1); y1 = clampi(y1, 0, ch - 1);
    float c00 = p[y0 * pitch + x0 * 2 + off], c10 = p[y0 * pitch + x1 * 2 + off];
    float c01 = p[y1 * pitch + x0 * 2 + off], c11 = p[y1 * pitch + x1 * 2 + off];
    return (c00 * (1 - ax) + c10 * ax) * (1 - ay) + (c01 * (1 - ax) + c11 * ax) * ay;
}

/* The warp texture is stored in VUYX byte order (V,U,Y,X), not Y,U,V - the warp
 * applies one per-pixel flow to all channels identically, so the channel order
 * is free, and matching a real packed layout lets `packed` output be a plain
 * array->frame copy (no repack).  The flow texture must keep luma in the R,G,B
 * slots (the downscale derives luminance from them), so it stays (Y,Y,Y,X). */
static inline __device__ void store(uchar4 *warp, int pw, uchar4 *flow, int pf,
                                    int x, int y, int yy, int u, int v)
{
    uchar4 *wr = (uchar4 *)((char *)warp + (long)y * pw);
    wr[x] = make_uchar4(v, u, yy, 255);
    if (flow) {
        uchar4 *fr = (uchar4 *)((char *)flow + (long)y * pf);
        fr[x] = make_uchar4(yy, yy, yy, 255);
    }
}

/* ---- 16-bit (P010/P016) variants: same logic on uint16 samples ---- */

static inline __device__ unsigned short to_u16(float v)
{
    int i = (int)(v + 0.5f);
    return (unsigned short)clampi(i, 0, 65535);
}

/* bilinear sample of an interleaved 2-component uint16 plane (P0xx UV),
 * short offset off (0=U, 1=V); pitch in bytes */
static inline __device__ float bilin2_16(const unsigned char *p, int pitch,
                                         int cw, int ch, float fx, float fy, int off)
{
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - x0, ay = fy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = clampi(x0, 0, cw - 1); x1 = clampi(x1, 0, cw - 1);
    y0 = clampi(y0, 0, ch - 1); y1 = clampi(y1, 0, ch - 1);
    const unsigned short *r0 = (const unsigned short *)(p + (long)y0 * pitch);
    const unsigned short *r1 = (const unsigned short *)(p + (long)y1 * pitch);
    float c00 = r0[x0 * 2 + off], c10 = r0[x1 * 2 + off];
    float c01 = r1[x0 * 2 + off], c11 = r1[x1 * 2 + off];
    return (c00 * (1 - ax) + c10 * ax) * (1 - ay) + (c01 * (1 - ax) + c11 * ax) * ay;
}

/* 16-bit warp texture is stored in XV48LE byte order (U,Y,V,X) - see store();
 * the flow texture keeps luma in the R,G,B slots as (Y,Y,Y,X). */
static inline __device__ void store16(ushort4 *warp, int pw, ushort4 *flow, int pf,
                                      int x, int y, int yy, int u, int v)
{
    ushort4 *wr = (ushort4 *)((char *)warp + (long)y * pw);
    wr[x] = make_ushort4(u, yy, v, 0xffff);
    if (flow) {
        ushort4 *fr = (ushort4 *)((char *)flow + (long)y * pf);
        fr[x] = make_ushort4(yy, yy, yy, 0xffff);
    }
}

extern "C" {

/* YUV420P (3 planes, 4:2:0) -> packed (Y,U,V,255) warp + (Y,Y,Y,255) flow */
__global__ void Pack_yuv420p(const unsigned char *Y, int pY,
                             const unsigned char *U, int pU,
                             const unsigned char *V, int pV,
                             uchar4 *warp, int pw, uchar4 *flow, int pf,
                             int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int cw = (W + 1) / 2, ch = (H + 1) / 2;
    float fx = (x - 0.5f) * 0.5f, fy = (y - 0.5f) * 0.5f;
    int yy = Y[(long)y * pY + x];
    int u  = to_u8(bilin1(U, pU, cw, ch, fx, fy));
    int v  = to_u8(bilin1(V, pV, cw, ch, fx, fy));
    store(warp, pw, flow, pf, x, y, yy, u, v);
}

/* NV12 (Y plane + interleaved UV, 4:2:0) -> packed warp + flow */
__global__ void Pack_nv12(const unsigned char *Y, int pY,
                          const unsigned char *UV, int pUV,
                          uchar4 *warp, int pw, uchar4 *flow, int pf,
                          int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int cw = (W + 1) / 2, ch = (H + 1) / 2;
    float fx = (x - 0.5f) * 0.5f, fy = (y - 0.5f) * 0.5f;
    int yy = Y[(long)y * pY + x];
    int u  = to_u8(bilin2(UV, pUV, cw, ch, fx, fy, 0));
    int v  = to_u8(bilin2(UV, pUV, cw, ch, fx, fy, 1));
    store(warp, pw, flow, pf, x, y, yy, u, v);
}

/* NV16 (Y plane + interleaved UV, 4:2:2) -> packed warp + flow.  Chroma is
 * subsampled horizontally only (full vertical resolution), so fy == y. */
__global__ void Pack_nv16(const unsigned char *Y, int pY,
                          const unsigned char *UV, int pUV,
                          uchar4 *warp, int pw, uchar4 *flow, int pf,
                          int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int cw = (W + 1) / 2, ch = H;
    float fx = (x - 0.5f) * 0.5f, fy = y;
    int yy = Y[(long)y * pY + x];
    int u  = to_u8(bilin2(UV, pUV, cw, ch, fx, fy, 0));
    int v  = to_u8(bilin2(UV, pUV, cw, ch, fx, fy, 1));
    store(warp, pw, flow, pf, x, y, yy, u, v);
}

/* YUV444P (3 full-res planes) -> packed warp + flow (no chroma resample) */
__global__ void Pack_yuv444p(const unsigned char *Y, int pY,
                             const unsigned char *U, int pU,
                             const unsigned char *V, int pV,
                             uchar4 *warp, int pw, uchar4 *flow, int pf,
                             int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int yy = Y[(long)y * pY + x];
    int u  = U[(long)y * pU + x];
    int v  = V[(long)y * pV + x];
    store(warp, pw, flow, pf, x, y, yy, u, v);
}

/* packed (Y,U,V,A) -> planar YUV444P (3 planes) */
__global__ void Unpack_yuv444p(const uchar4 *src, int ps,
                               unsigned char *Y, int pY,
                               unsigned char *U, int pU,
                               unsigned char *V, int pV,
                               int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const uchar4 *sr = (const uchar4 *)((const char *)src + (long)y * ps);
    uchar4 p = sr[x];   /* VUYX: V,U,Y,X */
    Y[(long)y * pY + x] = p.z;
    U[(long)y * pU + x] = p.y;
    V[(long)y * pV + x] = p.x;
}

/* P010/P016 (uint16 Y plane + interleaved uint16 UV, 4:2:0) -> packed ushort4
 * warp (Y,U,V,0xffff) + grey flow (Y,Y,Y,0xffff).  One kernel serves both: the
 * UNORM_INT16 array normalize absorbs P010's 10-bits-in-MSB scaling. */
__global__ void Pack_p016(const unsigned char *Y, int pY,
                          const unsigned char *UV, int pUV,
                          ushort4 *warp, int pw, ushort4 *flow, int pf,
                          int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int cw = (W + 1) / 2, ch = (H + 1) / 2;
    float fx = (x - 0.5f) * 0.5f, fy = (y - 0.5f) * 0.5f;
    int yy = ((const unsigned short *)(Y + (long)y * pY))[x];
    int u  = to_u16(bilin2_16(UV, pUV, cw, ch, fx, fy, 0));
    int v  = to_u16(bilin2_16(UV, pUV, cw, ch, fx, fy, 1));
    store16(warp, pw, flow, pf, x, y, yy, u, v);
}

/* P210/P212 (uint16 Y plane + interleaved uint16 UV, 4:2:2) -> packed ushort4
 * warp + grey flow.  Like Pack_p016 but chroma is full-height (fy == y); the
 * UNORM_INT16 normalize absorbs the 10/12-bits-in-MSB scaling for both. */
__global__ void Pack_p216(const unsigned char *Y, int pY,
                          const unsigned char *UV, int pUV,
                          ushort4 *warp, int pw, ushort4 *flow, int pf,
                          int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int cw = (W + 1) / 2, ch = H;
    float fx = (x - 0.5f) * 0.5f, fy = y;
    int yy = ((const unsigned short *)(Y + (long)y * pY))[x];
    int u  = to_u16(bilin2_16(UV, pUV, cw, ch, fx, fy, 0));
    int v  = to_u16(bilin2_16(UV, pUV, cw, ch, fx, fy, 1));
    store16(warp, pw, flow, pf, x, y, yy, u, v);
}

/* YUV444P16 (3 full-res uint16 planes) -> packed ushort4 warp + grey flow
 * (no chroma resample) */
__global__ void Pack_yuv444p16(const unsigned char *Y, int pY,
                               const unsigned char *U, int pU,
                               const unsigned char *V, int pV,
                               ushort4 *warp, int pw, ushort4 *flow, int pf,
                               int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    int yy = ((const unsigned short *)(Y + (long)y * pY))[x];
    int u  = ((const unsigned short *)(U + (long)y * pU))[x];
    int v  = ((const unsigned short *)(V + (long)y * pV))[x];
    store16(warp, pw, flow, pf, x, y, yy, u, v);
}

/* packed ushort4 (Y,U,V,A) -> planar YUV444P16 (3 uint16 planes) */
__global__ void Unpack_yuv444p16(const ushort4 *src, int ps,
                                 unsigned char *Y, int pY,
                                 unsigned char *U, int pU,
                                 unsigned char *V, int pV,
                                 int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const ushort4 *sr = (const ushort4 *)((const char *)src + (long)y * ps);
    ushort4 p = sr[x];   /* XV48: U,Y,V,X */
    ((unsigned short *)(Y + (long)y * pY))[x] = p.y;
    ((unsigned short *)(U + (long)y * pU))[x] = p.x;
    ((unsigned short *)(V + (long)y * pV))[x] = p.z;
}

/* ---- packed 4:4:4 YUV input: one interleaved plane -> warp + luma-grey flow.
 * These read the packed source, extract Y,U,V (full-res, no chroma resample)
 * and hand them to store()/store16(), which lay the warp buffer out in the
 * internal VUYX / XV48 order and the flow buffer as luma-grey. ---- */

/* VUYX / VUYA (byte order V,U,Y,X|A) -> warp + flow */
__global__ void Pack_vuyx(const unsigned char *src, int ps,
                          uchar4 *warp, int pw, uchar4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    uchar4 p = ((const uchar4 *)(src + (long)y * ps))[x];   /* V,U,Y,* */
    store(warp, pw, flow, pf, x, y, p.z, p.y, p.x);
}

/* UYVA (byte order U,Y,V,A) -> warp + flow */
__global__ void Pack_uyva(const unsigned char *src, int ps,
                          uchar4 *warp, int pw, uchar4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    uchar4 p = ((const uchar4 *)(src + (long)y * ps))[x];   /* U,Y,V,A */
    store(warp, pw, flow, pf, x, y, p.y, p.x, p.z);
}

/* AYUV (byte order A,Y,U,V) -> warp + flow */
__global__ void Pack_ayuv(const unsigned char *src, int ps,
                          uchar4 *warp, int pw, uchar4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    uchar4 p = ((const uchar4 *)(src + (long)y * ps))[x];   /* A,Y,U,V */
    store(warp, pw, flow, pf, x, y, p.y, p.z, p.w);
}

/* XV48LE (short order U,Y,V,X) -> warp + flow */
__global__ void Pack_xv48(const unsigned char *src, int ps,
                          ushort4 *warp, int pw, ushort4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    ushort4 p = ((const ushort4 *)(src + (long)y * ps))[x];   /* U,Y,V,X */
    store16(warp, pw, flow, pf, x, y, p.y, p.x, p.z);
}

/* AYUV64LE (short order A,Y,U,V) -> warp + flow */
__global__ void Pack_ayuv64(const unsigned char *src, int ps,
                            ushort4 *warp, int pw, ushort4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    ushort4 p = ((const ushort4 *)(src + (long)y * ps))[x];   /* A,Y,U,V */
    store16(warp, pw, flow, pf, x, y, p.y, p.z, p.w);
}

/* XV30LE (2:10:10:10 LE word: X[31:30] V[29:20] Y[19:10] U[9:0]) -> warp + flow.
 * The 10-bit channels are bit-replicated to 16-bit to match the UNORM_INT16
 * normalize (as for P010 / x2rgb10). */
__global__ void Pack_xv30(const unsigned char *src, int ps,
                          ushort4 *warp, int pw, ushort4 *flow, int pf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    unsigned int w = ((const unsigned int *)(src + (long)y * ps))[x];
    unsigned int u = (w        & 0x3ff), yy = (w >> 10) & 0x3ff, v = (w >> 20) & 0x3ff;
    store16(warp, pw, flow, pf, x, y, (yy << 6) | (yy >> 4),
                                      (u  << 6) | (u  >> 4),
                                      (v  << 6) | (v  >> 4));
}

/* x2rgb10 (packed 2:10:10:10 in a 32-bit LE word: X[31:30] R[29:20] G[19:10]
 * B[9:0]) -> packed ushort4 RGBA warp buffer.  The three 10-bit channels are
 * bit-replicated to 16-bit so the UNORM_INT16 texture normalize maps them to
 * the same [0,1] as the native 8-bit RGB path.  RGB needs no separate luma-grey
 * flow texture (the net derives luma internally, as for RGB0/RGBA), so the
 * filter aliases the flow texture to this warp buffer and no flow store is
 * emitted here. */
__global__ void Pack_x2rgb10(const unsigned char *src, int ps,
                             ushort4 *warp, int pw,
                             int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    unsigned int w = ((const unsigned int *)(src + (long)y * ps))[x];
    unsigned int r = (w >> 20) & 0x3ff;
    unsigned int g = (w >> 10) & 0x3ff;
    unsigned int b =  w        & 0x3ff;
    ushort4 *wr = (ushort4 *)((char *)warp + (long)y * pw);
    wr[x] = make_ushort4((r << 6) | (r >> 4),
                         (g << 6) | (g >> 4),
                         (b << 6) | (b >> 4), 0xffff);
}

/* packed ushort4 RGBA (network output) -> x2rgb10 (2:10:10:10 LE word).  Each
 * 16-bit channel is truncated to its top 10 bits; the 2 pad bits are 0. */
__global__ void Unpack_x2rgb10(const ushort4 *src, int ps,
                               unsigned char *dst, int pd,
                               int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const ushort4 *sr = (const ushort4 *)((const char *)src + (long)y * ps);
    ushort4 p = sr[x];
    unsigned int r = p.x >> 6, g = p.y >> 6, b = p.z >> 6;
    ((unsigned int *)(dst + (long)y * pd))[x] = (r << 20) | (g << 10) | b;
}

/* x2bgr10: the R/B-swapped sibling of x2rgb10 (LE word: X[31:30] B[29:20]
 * G[19:10] R[9:0]).  Warp buffer stays R,G,B order like the native RGB path. */
__global__ void Pack_x2bgr10(const unsigned char *src, int ps,
                             ushort4 *warp, int pw,
                             int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    unsigned int w = ((const unsigned int *)(src + (long)y * ps))[x];
    unsigned int b = (w >> 20) & 0x3ff;
    unsigned int g = (w >> 10) & 0x3ff;
    unsigned int r =  w        & 0x3ff;
    ushort4 *wr = (ushort4 *)((char *)warp + (long)y * pw);
    wr[x] = make_ushort4((r << 6) | (r >> 4),
                         (g << 6) | (g >> 4),
                         (b << 6) | (b >> 4), 0xffff);
}

/* packed ushort4 RGBA (network output) -> x2bgr10 (R in the low 10 bits) */
__global__ void Unpack_x2bgr10(const ushort4 *src, int ps,
                               unsigned char *dst, int pd,
                               int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const ushort4 *sr = (const ushort4 *)((const char *)src + (long)y * ps);
    ushort4 p = sr[x];
    unsigned int r = p.x >> 6, g = p.y >> 6, b = p.z >> 6;
    ((unsigned int *)(dst + (long)y * pd))[x] = (b << 20) | (g << 10) | r;
}

}
