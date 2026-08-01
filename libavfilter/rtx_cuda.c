/*
 * Shared core for the NVIDIA RTX Video CUDA filters.
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

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/eval.h"
#include "libavutil/file.h"
#include "libavutil/macros.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "libavutil/thread.h"

#include "filters.h"
#include "rtx_cuda.h"
#include "video.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, r->hwctx->internal->cuda_dl, x)

/* Arena sub-buffer alignment (>= cuMemAlloc's own guarantee, which the
 * per-buffer allocations used to rely on) and a trailing guard covering the
 * tile/halo over-read past the final buffer described in the header. */
#define RTX_ALLOC_ALIGN 512
#define RTX_ALLOC_GUARD (1 << 20)

const FFRtxPixFmt ff_rtx_packed_rgb_fmts[5] = {
    { AV_PIX_FMT_RGB0,     CU_AD_FORMAT_UNSIGNED_INT8, 4, 0 },
    { AV_PIX_FMT_RGBA,     CU_AD_FORMAT_UNSIGNED_INT8, 4, 0 },
    { AV_PIX_FMT_BGR0,     CU_AD_FORMAT_UNSIGNED_INT8, 4, 1 },
    { AV_PIX_FMT_BGRA,     CU_AD_FORMAT_UNSIGNED_INT8, 4, 1 },
    { AV_PIX_FMT_RGBA64LE, CU_AD_FORMAT_UNORM_INT16X4, 8, 2 },
};

const FFRtxPixFmt *ff_rtx_find_fmt(const FFRtxPixFmt *tbl, int n,
                                   enum AVPixelFormat f)
{
    for (int i = 0; i < n; i++)
        if (tbl[i].f == f)
            return &tbl[i];
    return NULL;
}

/* ------------------------------------------------------------------------- *
 * cuSurfObjectCreate/Destroy
 *
 * The only pair ffnvcodec's dynlink loader does not export, so it comes
 * straight out of libcuda -- once per process.  libcuda is already loaded (the
 * hwcontext holds it) and lives for the process, so this neither dlcloses nor
 * refcounts.
 * ------------------------------------------------------------------------- */
typedef CUresult (*tcuSurfObjectCreate)(FFCUsurfObject *, const CUDA_RESOURCE_DESC *);
typedef CUresult (*tcuSurfObjectDestroy)(FFCUsurfObject);

static tcuSurfObjectCreate  rtx_surf_create;
static tcuSurfObjectDestroy rtx_surf_destroy;

static void rtx_load_surf_fns(void)
{
    void *libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!libcuda)
        return;
    rtx_surf_create  = (tcuSurfObjectCreate)dlsym(libcuda, "cuSurfObjectCreate");
    rtx_surf_destroy = (tcuSurfObjectDestroy)dlsym(libcuda, "cuSurfObjectDestroy");
}

static int rtx_surf_fns(AVFilterContext *ctx)
{
    static AVOnce once = AV_ONCE_INIT;
    ff_thread_once(&once, rtx_load_surf_fns);
    if (!rtx_surf_create || !rtx_surf_destroy) {
        av_log(ctx, AV_LOG_ERROR, "cuSurfObjectCreate unavailable\n");
        return AVERROR_EXTERNAL;
    }
    return 0;
}

/* ------------------------------------------------------------------------- *
 * Device binding and output plumbing
 * ------------------------------------------------------------------------- */
void ff_rtx_uninit(AVFilterContext *ctx)
{
    FFRtxPriv *p = ctx->priv;
    ff_rtx_free_graph(ctx, &p->r);
}

int ff_rtx_config_formats(AVFilterContext *ctx, AVFilterLink *inlink,
                          const FFRtxFormats *f,
                          AVHWFramesContext **in_frames_ctx,
                          const FFRtxPixFmt **inpf, const FFRtxPixFmt **outpf)
{
    FilterLink *il = ff_filter_link(inlink);
    const char *hint = f->hint ? f->hint : "";
    const char *open = f->hint ? " (" : "", *close = f->hint ? ")" : "";
    enum AVPixelFormat fmt;

    if (!il->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    *in_frames_ctx = (AVHWFramesContext *)il->hw_frames_ctx->data;

    fmt   = (*in_frames_ctx)->sw_format;
    *inpf = ff_rtx_find_fmt(f->in_tbl, f->n_in, fmt);
    if (!*inpf) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format %s%s%s%s\n",
               av_get_pix_fmt_name(fmt), open, hint, close);
        return AVERROR(ENOSYS);
    }
    if (!outpf)
        return 0;

    /* av_get_pix_fmt() strcmps its argument, so an option cleared to NULL --
     * av_opt_set(..., "format", NULL, 0) is legal for a string option -- must
     * not reach it. */
    if (f->out_format && *f->out_format) {
        fmt = av_get_pix_fmt(f->out_format);
        if (fmt == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "invalid output format '%s'\n", f->out_format);
            return AVERROR(EINVAL);
        }
    }
    *outpf = ff_rtx_find_fmt(f->out_tbl ? f->out_tbl : f->in_tbl,
                             f->out_tbl ? f->n_out : f->n_in, fmt);
    if (!*outpf) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format %s%s%s%s\n",
               av_get_pix_fmt_name(fmt), open, hint, close);
        return AVERROR(ENOSYS);
    }
    return 0;
}

int ff_rtx_bind_device(AVFilterContext *ctx, FFRtxCuda *r,
                       AVHWFramesContext *in_frames_ctx)
{
    r->device_ref = av_buffer_ref(in_frames_ctx->device_ref);
    if (!r->device_ref)
        return AVERROR(ENOMEM);
    r->hwctx  = ((AVHWDeviceContext *)r->device_ref->data)->hwctx;
    r->cu_ctx = r->hwctx->cuda_ctx;
    r->stream = r->hwctx->stream;
    return 0;
}

int ff_rtx_config_hwframes(AVFilterContext *ctx, AVFilterLink *outlink,
                           FFRtxCuda *r, int oW, int oH,
                           enum AVPixelFormat sw_format)
{
    FilterLink *ol = ff_filter_link(outlink);
    AVHWFramesContext *out_frames_ctx;
    int ret;

    outlink->w = oW;
    outlink->h = oH;

    av_buffer_unref(&ol->hw_frames_ctx);
    ol->hw_frames_ctx = av_hwframe_ctx_alloc(r->device_ref);
    if (!ol->hw_frames_ctx)
        return AVERROR(ENOMEM);
    out_frames_ctx = (AVHWFramesContext *)ol->hw_frames_ctx->data;
    out_frames_ctx->format            = AV_PIX_FMT_CUDA;
    out_frames_ctx->sw_format         = sw_format;
    out_frames_ctx->width             = oW;
    out_frames_ctx->height            = oH;
    out_frames_ctx->initial_pool_size = 4;

    if ((ret = ff_filter_init_hw_frames(ctx, outlink, 4)) < 0)
        return ret;
    ret = av_hwframe_ctx_init(ol->hw_frames_ctx);
    if (ret < 0)
        av_log(ctx, AV_LOG_ERROR, "Failed to init CUDA frame context: %d\n", ret);
    return ret;
}

int ff_rtx_setup(AVFilterContext *ctx, FFRtxCuda *r, const char *what,
                 int (*setup_graph)(AVFilterContext *ctx))
{
    CUcontext dummy;
    int ret;

    if ((ret = CHECK_CU(r->hwctx->internal->cuda_dl->cuCtxPushCurrent(r->cu_ctx))) < 0)
        return ret;
    ret = setup_graph(ctx);
    CHECK_CU(r->hwctx->internal->cuda_dl->cuCtxPopCurrent(&dummy));
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "%s graph setup failed (%d)\n", what, ret);
        return ret;
    }
    r->ready = 1;
    return 0;
}

/* ------------------------------------------------------------------------- *
 * Graph setup
 * ------------------------------------------------------------------------- */
int ff_rtx_arch_gate(AVFilterContext *ctx, FFRtxCuda *r,
                     const FFRtxArchGate *gate, int experimental)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    CUdevice dev = 0;
    int cc_major = 0, cc_minor = 0, ret;

    if ((ret = CHECK_CU(cu->cuCtxGetDevice(&dev))) < 0)
        return ret;
    if ((ret = CHECK_CU(cu->cuDeviceGetAttribute(&cc_major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))) < 0)
        return ret;
    if ((ret = CHECK_CU(cu->cuDeviceGetAttribute(&cc_minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))) < 0)
        return ret;

    if (gate->hard_min_major && cc_major < gate->hard_min_major) {
        av_log(ctx, AV_LOG_ERROR, gate->hard_msg, cc_major, cc_minor);
        return AVERROR(ENOSYS);
    }
    /* Blackwell (cc 12.x) and Ada (cc 8.9) are the two the cubins were verified
     * byte-exact on; everything else needs the opt-in. */
    if (cc_major >= 12 || (cc_major == 8 && cc_minor == 9))
        return 0;
    if (!experimental) {
        av_log(ctx, AV_LOG_ERROR, gate->gate_msg, cc_major, cc_minor);
        return AVERROR(ENOSYS);
    }
    av_log(ctx, AV_LOG_WARNING, gate->warn_msg, cc_major, cc_minor);
    return 0;
}

int ff_rtx_load_modules(AVFilterContext *ctx, FFRtxCuda *r, const char *dir,
                        const FFRtxModule *mods, int nmod, int max_mid,
                        const FFRtxFunc *funcs, int nfunc, int max_fid,
                        const char *load_hint)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    char path[1024];
    int ret;

    r->max_mid = max_mid;
    r->max_fid = max_fid;
    r->mod = av_calloc(max_mid + 1, sizeof(*r->mod));
    r->fn  = av_calloc(max_fid + 1, sizeof(*r->fn));
    if (!r->mod || !r->fn)
        return AVERROR(ENOMEM);

    for (int i = 0; i < nmod; i++) {
        uint8_t *buf = NULL;
        size_t bsz = 0;

        if (mods[i].mid < 0 || mods[i].mid > max_mid) {
            av_log(ctx, AV_LOG_ERROR, "%s has module id %d, past the %d these "
                   "tables were sized for\n", mods[i].file, mods[i].mid, max_mid);
            return AVERROR_BUG;
        }
        snprintf(path, sizeof(path), "%s/%s", dir, mods[i].file);
        ret = av_file_map(path, &buf, &bsz, 0, ctx);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "cannot read cubin %s\n", path);
            return ret;
        }
        /* Every cubin is a multi-arch fatbin; cuModuleLoadData picks the image
         * for the running GPU, so a failure here means this data dir carries
         * none. */
        ret = CHECK_CU(cu->cuModuleLoadData(&r->mod[mods[i].mid], buf));
        av_file_unmap(buf, bsz);
        if (ret < 0) {
            if (load_hint) {
                CUdevice dev = 0;
                int cc_major = 0, cc_minor = 0;
                cu->cuCtxGetDevice(&dev);
                cu->cuDeviceGetAttribute(&cc_major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                cu->cuDeviceGetAttribute(&cc_minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
                av_log(ctx, AV_LOG_ERROR, "%s has no image for this GPU (cc %d.%d).  %s\n",
                       mods[i].file, cc_major, cc_minor, load_hint);
            }
            return ret;
        }
    }
    for (int i = 0; i < nfunc; i++) {
        if (funcs[i].fid < 0 || funcs[i].fid > max_fid ||
            funcs[i].mid < 0 || funcs[i].mid > max_mid) {
            av_log(ctx, AV_LOG_ERROR, "kernel %s has ids %d/%d, past the %d/%d "
                   "these tables were sized for\n", funcs[i].name,
                   funcs[i].fid, funcs[i].mid, max_fid, max_mid);
            return AVERROR_BUG;
        }
        ret = CHECK_CU(cu->cuModuleGetFunction(&r->fn[funcs[i].fid],
                                               r->mod[funcs[i].mid], funcs[i].name));
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "missing kernel %s\n", funcs[i].name);
            return ret;
        }
    }
    return 0;
}

int ff_rtx_alloc_arena(AVFilterContext *ctx, FFRtxCuda *r, int nalloc,
                       void (*fill_sizes)(AVFilterContext *ctx, long long *sz),
                       unsigned flags)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    long long *sz;
    size_t total = 0;
    int ret;

    r->nalloc = nalloc;
    r->alloc = av_calloc(nalloc, sizeof(*r->alloc));
    sz       = av_calloc(nalloc, sizeof(*sz));
    if (!r->alloc || !sz) {
        av_freep(&sz);
        return AVERROR(ENOMEM);
    }

    fill_sizes(ctx, sz);
    /* Lay the arena out in one pass, parking each ordinal's offset in alloc[]
     * until there is a base address to add it to. */
    for (int a = 0; a < nalloc; a++) {
        r->alloc[a] = total;
        total += FFALIGN(sz[a] > 0 ? (size_t)sz[a] : 1, RTX_ALLOC_ALIGN);
    }
    av_freep(&sz);
    total += RTX_ALLOC_GUARD;

    if ((ret = CHECK_CU(cu->cuMemAlloc(&r->arena, total))) < 0)
        return ret;
    r->arena_size = total;
    for (int a = 0; a < nalloc; a++)
        r->alloc[a] += r->arena;

    if (flags & FF_RTX_ARENA_ZERO) {
        /* cuMemAlloc does not zero.  Start from a known-zero arena so any
         * scratch a kernel reads before writing is deterministically 0, as in a
         * fresh loader process; the weight uploads then fill their buffers. */
        if ((ret = CHECK_CU(cu->cuMemsetD8Async(r->arena, 0, total, r->stream))) < 0)
            return ret;
        if ((ret = CHECK_CU(cu->cuStreamSynchronize(r->stream))) < 0)
            return ret;
    }
    return 0;
}

int ff_rtx_snapshot_arena(AVFilterContext *ctx, FFRtxCuda *r)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    int ret;

    if (!r->arena_uploaded)     /* nothing to preserve; the reset is a pure memset */
        return 0;
    if ((ret = CHECK_CU(cu->cuMemAlloc(&r->arena_template, r->arena_uploaded))) < 0)
        return ret;
    return CHECK_CU(cu->cuMemcpyDtoDAsync(r->arena_template, r->arena,
                                          r->arena_uploaded, r->stream));
}

int ff_rtx_reset_arena(AVFilterContext *ctx, FFRtxCuda *r)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    int ret;

    if (r->arena_uploaded) {
        ret = CHECK_CU(cu->cuMemcpyDtoDAsync(r->arena, r->arena_template,
                                             r->arena_uploaded, r->stream));
        if (ret < 0)
            return ret;
    }
    return CHECK_CU(cu->cuMemsetD8Async(r->arena + r->arena_uploaded, 0,
                                        r->arena_size - r->arena_uploaded,
                                        r->stream));
}

int ff_rtx_upload_weights(AVFilterContext *ctx, FFRtxCuda *r, const char *dir,
                          const char *file, const FFRtxUpload *up, int nup)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    uint8_t *weights = NULL;
    size_t wsz = 0, end;
    char path[1024];
    int ret;

    snprintf(path, sizeof(path), "%s/%s", dir, file);
    if ((ret = av_file_map(path, &weights, &wsz, 0, ctx)) < 0) {
        av_log(ctx, AV_LOG_ERROR, "cannot read weights %s\n", path);
        return ret;
    }
    for (int i = 0; i < nup; i++) {
        if (up[i].file_off + up[i].size > (long long)wsz) {
            av_log(ctx, AV_LOG_ERROR, "%s is too small\n", path);
            ret = AVERROR_INVALIDDATA;
            break;
        }
        /* Async: a graph carries hundreds of small uploads (dlpp_drv: 525
         * averaging 8.8 KiB) and the blocking form pays its round trip on every
         * one of them.  The source is the mapping below, which has to stay put
         * until the copies land -- hence the synchronize before it is dropped. */
        ret = CHECK_CU(cu->cuMemcpyHtoDAsync((CUdeviceptr)up[i].dst,
                                             weights + up[i].file_off, up[i].size,
                                             r->stream));
        if (ret < 0)
            break;
        /* Track how far into the arena the uploads reach, so a later
         * ff_rtx_snapshot_arena() only has to preserve that much. */
        end = (size_t)((CUdeviceptr)up[i].dst + up[i].size - r->arena);
        if (end > r->arena_uploaded)
            r->arena_uploaded = end;
    }
    /* The copies read from the mapping, so they must complete before it goes. */
    if (ret >= 0)
        ret = CHECK_CU(cu->cuStreamSynchronize(r->stream));
    else
        cu->cuStreamSynchronize(r->stream);
    av_file_unmap(weights, wsz);
    return ret < 0 ? ret : 0;
}

int ff_rtx_alloc_launches(AVFilterContext *ctx, FFRtxCuda *r,
                          int nlaunch, size_t launch_size)
{
    r->launches = av_calloc(nlaunch, launch_size);
    if (!r->launches)
        return AVERROR(ENOMEM);
    r->nlaunch     = nlaunch;
    r->launch_size = launch_size;
    return 0;
}

/* ------------------------------------------------------------------------- *
 * Image binding
 * ------------------------------------------------------------------------- */
static FFRtxImage *rtx_image_slot(AVFilterContext *ctx, FFRtxCuda *r)
{
    if (r->nimage >= FF_RTX_MAX_IMAGES) {
        av_log(ctx, AV_LOG_ERROR, "too many graph images\n");
        return NULL;
    }
    return &r->image[r->nimage++];
}

/* The descriptor every captured graph samples with: linear filtering over
 * normalized coordinates.  Only the address mode varies. */
static CUDA_TEXTURE_DESC rtx_tex_desc(unsigned flags)
{
    CUDA_TEXTURE_DESC td = { 0 };
    if (flags & FF_RTX_CLAMP)
        td.addressMode[0] = td.addressMode[1] = td.addressMode[2] =
            CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode = CU_TR_FILTER_MODE_LINEAR;
    td.flags      = CU_TRSF_NORMALIZED_COORDINATES;
    return td;
}

static int rtx_bind_handles(AVFilterContext *ctx, FFRtxCuda *r, FFRtxImage *img,
                            const CUDA_RESOURCE_DESC *rd, unsigned flags)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    int ret;

    if (flags & FF_RTX_TEX) {
        CUDA_TEXTURE_DESC td = rtx_tex_desc(flags);
        if ((ret = CHECK_CU(cu->cuTexObjectCreate(&img->tex, rd, &td, NULL))) < 0)
            return ret;
    }
    if (flags & FF_RTX_SURF) {
        if ((ret = rtx_surf_fns(ctx)) < 0)
            return ret;
        if (rtx_surf_create(&img->surf, rd) != CUDA_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "cuSurfObjectCreate failed\n");
            return AVERROR_EXTERNAL;
        }
    }
    return 0;
}

FFRtxImage *ff_rtx_image_array(AVFilterContext *ctx, FFRtxCuda *r, int W, int H,
                               CUarray_format cufmt, unsigned flags)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    FFRtxImage *img = rtx_image_slot(ctx, r);
    CUDA_ARRAY3D_DESCRIPTOR ad = { 0 };
    CUDA_RESOURCE_DESC rd = { 0 };

    if (!img)
        return NULL;

    ad.Width = W; ad.Height = H; ad.Depth = 0;
    ad.Format = cufmt;
    ad.NumChannels = 4;
    ad.Flags = (flags & FF_RTX_LDST) ? CUDA_ARRAY3D_SURFACE_LDST : 0;
    if (CHECK_CU(cu->cuArray3DCreate(&img->arr, &ad)) < 0)
        return NULL;

    rd.resType = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = img->arr;
    return rtx_bind_handles(ctx, r, img, &rd, flags) < 0 ? NULL : img;
}

FFRtxImage *ff_rtx_image_pitch(AVFilterContext *ctx, FFRtxCuda *r, int W, int H,
                               CUarray_format cufmt, int bpp, unsigned flags)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    FFRtxImage *img = rtx_image_slot(ctx, r);
    CUDA_RESOURCE_DESC rd = { 0 };

    if (!img)
        return NULL;

    if (CHECK_CU(cu->cuMemAllocPitch(&img->ptr, &img->pitch,
                                     (size_t)W * bpp, H, 16)) < 0)
        return NULL;
    if (flags & FF_RTX_ZERO) {
        if (CHECK_CU(cu->cuMemsetD8Async(img->ptr, 0, img->pitch * H, r->stream)) < 0)
            return NULL;
    }

    rd.resType = CU_RESOURCE_TYPE_PITCH2D;
    rd.res.pitch2D.devPtr       = img->ptr;
    rd.res.pitch2D.format       = cufmt;
    rd.res.pitch2D.numChannels  = 4;
    rd.res.pitch2D.width        = W;
    rd.res.pitch2D.height       = H;
    rd.res.pitch2D.pitchInBytes = img->pitch;
    return rtx_bind_handles(ctx, r, img, &rd, flags) < 0 ? NULL : img;
}

int ff_rtx_tex_over_pitch(AVFilterContext *ctx, FFRtxCuda *r, CUdeviceptr ptr,
                          size_t pitch, int W, int H, CUarray_format cufmt,
                          unsigned flags, CUtexObject *tex)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    CUDA_TEXTURE_DESC td = rtx_tex_desc(flags);
    CUDA_RESOURCE_DESC rd = { 0 };

    rd.resType = CU_RESOURCE_TYPE_PITCH2D;
    rd.res.pitch2D.devPtr       = ptr;
    rd.res.pitch2D.format       = cufmt;
    rd.res.pitch2D.numChannels  = 4;
    rd.res.pitch2D.width        = W;
    rd.res.pitch2D.height       = H;
    rd.res.pitch2D.pitchInBytes = pitch;
    return CHECK_CU(cu->cuTexObjectCreate(tex, &rd, &td, NULL));
}

FFRtxImage *ff_rtx_image_linear(AVFilterContext *ctx, FFRtxCuda *r,
                                size_t size, size_t pitch)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    FFRtxImage *img = rtx_image_slot(ctx, r);

    if (!img)
        return NULL;
    if (CHECK_CU(cu->cuMemAlloc(&img->ptr, size)) < 0)
        return NULL;
    img->pitch = pitch;
    return img;
}

/* Resolve each launch's fnid back to a kernel name rather than the name to a
 * single fid: one kernel name can appear under several fids (the per-layer
 * k_conv modules all export k_conv_fp16_nhwc), so only the launch list says
 * which one the graph actually runs.  @p n limits the comparison to a prefix. */
static int rtx_find_launch(const FFRtxCuda *r, const FFRtxFunc *funcs, int nfunc,
                           const char *name, size_t n)
{
    for (int li = 0; li < r->nlaunch; li++) {
        int fnid = ff_rtx_launch_at(r, li)->fnid;
        for (int i = 0; i < nfunc; i++)
            if (funcs[i].fid == fnid && !strncmp(funcs[i].name, name, n))
                return li;
    }
    return -1;
}

int ff_rtx_find_launch(const FFRtxCuda *r, const FFRtxFunc *funcs, int nfunc,
                       const char *name)
{
    return rtx_find_launch(r, funcs, nfunc, name, strlen(name) + 1);
}

int ff_rtx_find_launch_prefix(const FFRtxCuda *r, const FFRtxFunc *funcs, int nfunc,
                              const char *prefix)
{
    return rtx_find_launch(r, funcs, nfunc, prefix, strlen(prefix));
}

/* ------------------------------------------------------------------------- *
 * Per-frame replay
 *
 * None of this synchronizes.  Every op -- the input copy, all the launches, the
 * output copy -- is issued on the shared device stream (hwctx->stream), and
 * every consumer runs on it too: a downstream CUDA filter, or hwcontext_cuda's
 * transfer path, which copies on that same stream and syncs itself.  So stream
 * issue-order already orders our output before any read of it, and orders the
 * next producer's reuse of the freed input buffer after our read.  Blocking per
 * frame would only bound errors to this frame, at the cost of all CPU/GPU
 * overlap.  This relies on the single-shared-stream contract: a consumer on its
 * own context/stream would need an event at that boundary.
 * ------------------------------------------------------------------------- */
int ff_rtx_filter_frame(AVFilterLink *inlink, AVFrame *in, FFRtxCuda *r,
                        const FFRtxFrameOp *op,
                        void (*retag)(AVFilterContext *ctx, AVFrame *out))
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    CudaFunctions *cu = r->hwctx ? r->hwctx->internal->cuda_dl : NULL;
    CUcontext dummy;
    AVFrame *out;
    int ret;

    if (!r->ready) {
        av_frame_free(&in);
        return AVERROR(EINVAL);
    }

    out = ff_get_video_buffer(outlink, op->oW, op->oH);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);
    if (retag)
        retag(ctx, out);

    ret = FF_CUDA_CHECK_DL(ctx, cu, cu->cuCtxPushCurrent(r->cu_ctx));
    if (ret < 0)
        goto fail;

    /* Restore the arena to its post-upload state for a graph that reads scratch
     * before writing it: a fresh process gets zeroed pages, a long-running host
     * recycles dirty memory. */
    if (op->flags & FF_RTX_OP_RESET_ARENA) {
        ret = ff_rtx_reset_arena(ctx, r);
        if (ret < 0)
            goto fail_pop;
    }
    ret = ff_rtx_frame_to_image(ctx, r, in, op->in_img, op->iW, op->iH, op->ibpp);
    if (ret < 0)
        goto fail_pop;
    ret = op->run ? op->run(ctx)
                  : ff_rtx_launch_all(ctx, r, !!(op->flags & FF_RTX_OP_PSIZE));
    if (ret < 0)
        goto fail_pop;
    ret = ff_rtx_image_to_frame(ctx, r, op->out_img, out, op->oW, op->oH, op->obpp);
    if (ret < 0)
        goto fail_pop;

    if (op->flags & FF_RTX_OP_OPAQUE_ALPHA)
        ret = ff_rtx_fill_opaque_alpha(ctx, r, out, op->oW, op->oH, op->obpp);

fail_pop:
    FF_CUDA_CHECK_DL(ctx, cu, cu->cuCtxPopCurrent(&dummy));
fail:
    av_frame_free(&in);
    if (ret < 0) {
        av_frame_free(&out);
        return ret;
    }
    return ff_filter_frame(outlink, out);
}

int ff_rtx_launch(AVFilterContext *ctx, FFRtxCuda *r, int fnid,
                  const unsigned grid[3], const unsigned block[3], unsigned smem,
                  void *params, size_t psize)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    void *extra[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, params,
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &psize, CU_LAUNCH_PARAM_END };

    /* The tables are the generator's, but the sizes they are indexed against
     * are the caller's -- ISR has to state them by hand, its header carrying no
     * MAX_FID -- so an out-of-range id is a bug to report, not to dereference. */
    if (fnid < 0 || fnid > r->max_fid || !r->fn[fnid]) {
        av_log(ctx, AV_LOG_ERROR, "launch of unresolved kernel id %d (max %d)\n",
               fnid, r->max_fid);
        return AVERROR_BUG;
    }
    return CHECK_CU(cu->cuLaunchKernel(r->fn[fnid], grid[0], grid[1], grid[2],
                                       block[0], block[1], block[2],
                                       smem, r->stream, NULL, extra));
}

int ff_rtx_launch_all(AVFilterContext *ctx, FFRtxCuda *r, int use_psize)
{
    for (int li = 0; li < r->nlaunch; li++) {
        FFRtxLaunch *l = ff_rtx_launch_at(r, li);
        int ret = ff_rtx_launch(ctx, r, l->fnid, l->grid, l->block, l->smem,
                                l->params, use_psize ? l->psize : l->argsize);
        if (ret < 0)
            return ret;
    }
    return 0;
}

int ff_rtx_frame_to_image(AVFilterContext *ctx, FFRtxCuda *r, const AVFrame *in,
                          const FFRtxImage *img, int W, int H, int bpp)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    CUDA_MEMCPY2D c = { 0 };

    c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    c.srcDevice     = (CUdeviceptr)in->data[0];
    c.srcPitch      = in->linesize[0];
    if (img->arr) {
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        c.dstArray      = img->arr;
    } else {
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.dstDevice     = img->ptr;
        c.dstPitch      = img->pitch;
    }
    c.WidthInBytes = (size_t)W * bpp;
    c.Height       = H;
    return CHECK_CU(cu->cuMemcpy2DAsync(&c, r->stream));
}

int ff_rtx_image_to_frame(AVFilterContext *ctx, FFRtxCuda *r,
                          const FFRtxImage *img, AVFrame *out,
                          int W, int H, int bpp)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    CUDA_MEMCPY2D c = { 0 };

    if (img->arr) {
        c.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        c.srcArray      = img->arr;
    } else {
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice     = img->ptr;
        c.srcPitch      = img->pitch;
    }
    c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c.dstDevice     = (CUdeviceptr)out->data[0];
    c.dstPitch      = out->linesize[0];
    c.WidthInBytes  = (size_t)W * bpp;
    c.Height        = H;
    return CHECK_CU(cu->cuMemcpy2DAsync(&c, r->stream));
}

int ff_rtx_fill_opaque_alpha(AVFilterContext *ctx, FFRtxCuda *r, AVFrame *out,
                             int oW, int oH, int bpp)
{
    CudaFunctions *cu = r->hwctx->internal->cuda_dl;
    size_t px = bpp;                                       /* 8 for rgba64le */
    CUdeviceptr a0 = (CUdeviceptr)out->data[0] + (px - 2); /* last u16 = alpha */
    int ret = 0;

    /* Set just the alpha u16 of each pixel, stride = bpp.  A padded row is
     * covered by running over the padding too: it is inside the frame
     * allocation and nothing reads it (hwframe transfers copy oW*bpp per row),
     * so one memset does the whole plane instead of one per row -- which at 4K
     * was over two thousand launches on the critical stream.  The pitch comes
     * from cuMemAllocPitch and is a multiple of 512, hence of bpp, but fall
     * back to the row loop rather than assume it. */
    if (out->linesize[0] % (int)px == 0) {
        size_t stride_px = (size_t)out->linesize[0] / px;
        return CHECK_CU(cu->cuMemsetD2D16Async(a0, px, 0xFFFF, 1,
                                               stride_px * oH, r->stream));
    }
    for (int y = 0; y < oH && ret >= 0; y++)
        ret = CHECK_CU(cu->cuMemsetD2D16Async(a0 + (size_t)y * out->linesize[0],
                                              px, 0xFFFF, 1, oW, r->stream));
    return ret;
}

/* ------------------------------------------------------------------------- *
 * Teardown and helpers
 * ------------------------------------------------------------------------- */
void ff_rtx_free_graph(AVFilterContext *ctx, FFRtxCuda *r)
{
    if (r->hwctx) {
        CudaFunctions *cu = r->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(r->cu_ctx));
        for (int i = 0; i < r->nimage; i++) {
            FFRtxImage *img = &r->image[i];
            if (img->tex)  CHECK_CU(cu->cuTexObjectDestroy(img->tex));
            if (img->surf) rtx_surf_destroy(img->surf);
            if (img->arr)  CHECK_CU(cu->cuArrayDestroy(img->arr));
            if (img->ptr)  CHECK_CU(cu->cuMemFree(img->ptr));
        }
        if (r->arena)          CHECK_CU(cu->cuMemFree(r->arena));
        if (r->arena_template) CHECK_CU(cu->cuMemFree(r->arena_template));
        for (int i = 0; r->mod && i <= r->max_mid; i++)
            if (r->mod[i]) CHECK_CU(cu->cuModuleUnload(r->mod[i]));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_freep(&r->mod);
    av_freep(&r->fn);
    av_freep(&r->alloc);
    av_freep(&r->launches);
    av_buffer_unref(&r->device_ref);
    memset(r, 0, sizeof(*r));
}

enum { VAR_IN_W, VAR_IW, VAR_IN_H, VAR_IH, VAR_VARS_NB };
static const char *const rtx_var_names[] = { "in_w", "iw", "in_h", "ih", NULL };

int ff_rtx_eval_dims(AVFilterContext *ctx, AVFilterLink *inlink,
                     const char *w_expr, const char *h_expr, int defscale,
                     int *oW, int *oH)
{
    double var_values[VAR_VARS_NB], res;
    int ret;

    var_values[VAR_IN_W] = var_values[VAR_IW] = inlink->w;
    var_values[VAR_IN_H] = var_values[VAR_IH] = inlink->h;

    if (w_expr && *w_expr) {
        if ((ret = av_expr_parse_and_eval(&res, w_expr, rtx_var_names, var_values,
                                          NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
            return ret;
        *oW = (int)(res + 0.5);
    } else {
        *oW = inlink->w * defscale;
    }
    if (h_expr && *h_expr) {
        if ((ret = av_expr_parse_and_eval(&res, h_expr, rtx_var_names, var_values,
                                          NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
            return ret;
        *oH = (int)(res + 0.5);
    } else {
        *oH = inlink->h * defscale;
    }
    if (*oW < 1 || *oH < 1) {
        av_log(ctx, AV_LOG_ERROR, "invalid output size %dx%d\n", *oW, *oH);
        return AVERROR(EINVAL);
    }
    return 0;
}

/*   scale = f32(544)/f32(min(W,H));  q(v) = 32*floor(((double)(f32)(v*scale) + 24)/32) */
static int rtx_nn_q(int v, float scale)
{
    float t = (float)v * scale;
    double u = (double)t + 24.0;
    return 32 * (int)floor(u / 32.0);
}

void ff_rtx_nn_dims(int W, int H, int *NW, int *NH)
{
    int mn = W < H ? W : H;
    float scale = 544.0f / (float)mn;

    *NW = rtx_nn_q(W, scale);
    *NH = rtx_nn_q(H, scale);
}
