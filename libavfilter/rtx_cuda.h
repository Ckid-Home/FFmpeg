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

/**
 * @file
 * Shared machinery for the filters that replay a captured NVIDIA kernel graph:
 * vf_vsr_cuda, vf_vsr_drv_cuda, vf_dlpp_drv_cuda, vf_deepdvc_drv_cuda,
 * vf_truehdr_cuda, vf_truehdr_drv_cuda and vf_isr_cuda.
 *
 * All seven are 1:1 filters built the same way.  rtx-video-re emits a
 * <feature>_cuda_gen.h describing one feature's graph -- which cubins to load,
 * how big each scratch buffer is at a given frame size, where the weights go,
 * and every launch's grid/block/argument block -- and the filter's job is to
 * turn that into CUDA calls.  That job is the same every time: gate the GPU
 * architecture, load the modules, lay out one contiguous arena, upload the
 * weights, bind the input/output images, then replay the launch list once per
 * frame.  This header is that job, written once.
 *
 * What stays in each vf_*.c is what genuinely differs: its AVOptions, which
 * generated config the options select, the shape of its I/O binding, and the
 * named tunable offsets it patches into the argument blocks.
 *
 * The generated headers are per-feature and expose everything as `static`, so
 * this core never includes them.  Instead each filter passes its tables in
 * through the layout-compatible views below, and calls its own generated fills
 * behind the small callbacks these functions take.
 */

#ifndef AVFILTER_RTX_CUDA_H
#define AVFILTER_RTX_CUDA_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/pixfmt.h"

#include "avfilter.h"

/* Constants ffnvcodec's dynlink headers do not carry.  Everything else these
 * filters need -- cuTexObjectCreate, cuArray3DCreate, cuMemAllocPitch,
 * cuMemsetD2D16Async, cuMemcpyDtoDAsync, the CU_AD_FORMAT_* enumerators for the
 * standard types -- is already in CudaFunctions / dynlink_cuda.h. */
#ifndef CU_TRSF_NORMALIZED_COORDINATES
#define CU_TRSF_NORMALIZED_COORDINATES 0x02
#endif
#ifndef CU_AD_FORMAT_UNORM_INT16X4
#define CU_AD_FORMAT_UNORM_INT16X4 ((CUarray_format)0xc5)
#endif
#ifndef CU_AD_FORMAT_UNORM_INT_101010_2
#define CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#endif
/* cuLaunchKernel packed-argument sentinels. */
#ifndef CU_LAUNCH_PARAM_END
#define CU_LAUNCH_PARAM_END            ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#endif

/* cuSurfObjectCreate/Destroy are the one pair ffnvcodec's loader does not
 * export, so they are resolved out of libcuda directly -- once per process,
 * inside this core, rather than once per filter instance. */
typedef unsigned long long FFCUsurfObject;

/* ------------------------------------------------------------------------- *
 * Layout-compatible views of the generated tables.
 *
 * Every <feature>_cuda_gen.h emits its module, function and upload records with
 * the same layout under a per-feature struct name, and its launch record with a
 * common leading sequence (the trailing params[] array is sized per feature).
 * The core works through these views; each filter casts its own tables and
 * proves the cast with FF_RTX_ASSERT_*_LAYOUT, so a generator change that broke
 * the assumption would fail the build rather than corrupt a launch.
 * ------------------------------------------------------------------------- */
typedef struct FFRtxModule { int mid; const char *file; } FFRtxModule;
typedef struct FFRtxFunc   { int fid, mid; const char *name; } FFRtxFunc;
typedef struct FFRtxUpload { long long file_off, size; uint64_t dst; } FFRtxUpload;

typedef struct FFRtxLaunch {
    int      fnid, argsize, psize;
    unsigned grid[3], block[3], smem;
    uint8_t  params[];
} FFRtxLaunch;

#define FF_RTX_ASSERT_FIELD(T, U, f) \
    static_assert(offsetof(T, f) == offsetof(U, f) && \
                  sizeof(((T *)0)->f) == sizeof(((U *)0)->f), \
                  #T "." #f " does not match " #U)

#define FF_RTX_ASSERT_MODULE_LAYOUT(T) \
    static_assert(sizeof(T) == sizeof(FFRtxModule), #T " is not FFRtxModule-shaped")
#define FF_RTX_ASSERT_FUNC_LAYOUT(T) \
    static_assert(sizeof(T) == sizeof(FFRtxFunc), #T " is not FFRtxFunc-shaped"); \
    FF_RTX_ASSERT_FIELD(T, FFRtxFunc, fid); \
    FF_RTX_ASSERT_FIELD(T, FFRtxFunc, mid); \
    FF_RTX_ASSERT_FIELD(T, FFRtxFunc, name)
#define FF_RTX_ASSERT_UPLOAD_LAYOUT(T) \
    static_assert(sizeof(T) == sizeof(FFRtxUpload), #T " is not FFRtxUpload-shaped"); \
    FF_RTX_ASSERT_FIELD(T, FFRtxUpload, file_off); \
    FF_RTX_ASSERT_FIELD(T, FFRtxUpload, size); \
    FF_RTX_ASSERT_FIELD(T, FFRtxUpload, dst)
/* The launch view is a prefix, not the whole struct -- params[] is sized per
 * feature -- so this checks the leading sequence plus where params[] starts. */
#define FF_RTX_ASSERT_LAUNCH_LAYOUT(T) \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, fnid); \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, argsize); \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, psize); \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, grid); \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, block); \
    FF_RTX_ASSERT_FIELD(T, FFRtxLaunch, smem); \
    static_assert(offsetof(T, params) == sizeof(FFRtxLaunch), \
                  #T ".params does not follow the FFRtxLaunch prefix")

/* ------------------------------------------------------------------------- *
 * Pixel formats
 * ------------------------------------------------------------------------- */
/**
 * A frame format the graph can be bound to.  @p sel is the kernel's own format
 * selector where the network has one (VSR/DLPP: 0 = raw 8-bit RGB order,
 * 1 = raw 8-bit with an R<->B swap, 2 = the format-agnostic tex.f32 read /
 * sust.p store that packs any UNORM array in its native order); features
 * without a selector leave it 0, or reuse the field for their own flag.
 */
typedef struct FFRtxPixFmt {
    enum AVPixelFormat f;
    CUarray_format     cufmt;
    int                bpp;
    int                sel;
} FFRtxPixFmt;

/* The packed-RGB formats the VSR-family networks accept.  The R<->B swap only
 * exists on the raw 8-bit path, so B-first formats are 8-bit only; higher bit
 * depths go through the native-order sel-2 path.  A feature whose selector is
 * not mapped cannot honour sel at all, so it takes only the R-first 8-bit rows
 * (FF_RTX_N_RGB8_R_FIRST): feeding it a B-first frame would drive the network
 * with red and blue transposed. */
extern const FFRtxPixFmt ff_rtx_packed_rgb_fmts[5];
#define FF_RTX_N_RGB8_R_FIRST 2

const FFRtxPixFmt *ff_rtx_find_fmt(const FFRtxPixFmt *tbl, int n,
                                   enum AVPixelFormat f);

/* ------------------------------------------------------------------------- *
 * Runtime state
 * ------------------------------------------------------------------------- */
/**
 * One image the graph binds: a CUDA array or a linear/pitched allocation, with
 * the bindless texture and/or surface handle over it.  Only the members the
 * requested binding needs are set; the rest stay zero.
 */
typedef struct FFRtxImage {
    CUarray        arr;    ///< set for array-backed images
    CUdeviceptr    ptr;    ///< set for pitched/linear images
    size_t         pitch;  ///< row stride of @ref ptr
    CUtexObject    tex;
    FFCUsurfObject surf;
} FFRtxImage;

#define FF_RTX_MAX_IMAGES 8

/**
 * Everything the core allocates for one configured graph.  Embed this in the
 * filter's private context and pass its address to every ff_rtx_* call;
 * ff_rtx_free_graph() releases all of it.
 */
typedef struct FFRtxCuda {
    AVCUDADeviceContext *hwctx;
    AVBufferRef         *device_ref;
    CUcontext            cu_ctx;
    CUstream             stream;

    CUmodule            *mod;            ///< [max_mid + 1], indexed by mid
    CUfunction          *fn;             ///< [max_fid + 1], indexed by fid
    int                  max_mid, max_fid;

    CUdeviceptr          arena;          ///< one contiguous block, sub-allocated
    CUdeviceptr          arena_template; ///< pristine copy of the uploaded prefix
    size_t               arena_size;
    size_t               arena_uploaded;  ///< bytes from arena start covered by uploads
    CUdeviceptr         *alloc;          ///< [nalloc] pointers into @ref arena
    int                  nalloc;

    void                *launches;       ///< the feature's own launch array
    int                  nlaunch;
    size_t               launch_size;    ///< sizeof one element of @ref launches

    FFRtxImage           image[FF_RTX_MAX_IMAGES];
    int                  nimage;

    int                  ready;          ///< the graph is built and replayable
} FFRtxCuda;

/** The launch at index @p i of a graph built by ff_rtx_alloc_launches(). */
static inline FFRtxLaunch *ff_rtx_launch_at(const FFRtxCuda *r, int i)
{
    return (FFRtxLaunch *)((uint8_t *)r->launches + (size_t)i * r->launch_size);
}

/**
 * Every filter in the family embeds FFRtxCuda directly after its AVClass
 * pointer, so one uninit serves them all.  FF_RTX_ASSERT_PRIV_LAYOUT proves the
 * layout per filter, so a context that grew a member in front would fail the
 * build rather than free the wrong bytes.
 */
typedef struct FFRtxPriv {
    const AVClass *class;
    FFRtxCuda      r;
} FFRtxPriv;

#define FF_RTX_ASSERT_PRIV_LAYOUT(T) \
    static_assert(offsetof(T, r) == offsetof(FFRtxPriv, r), \
                  #T ".r is not where FFRtxPriv.r is")

void ff_rtx_uninit(AVFilterContext *ctx);

/* ------------------------------------------------------------------------- *
 * Device binding and output plumbing
 * ------------------------------------------------------------------------- */
/**
 * The formats a filter accepts, for ff_rtx_config_formats().
 */
typedef struct FFRtxFormats {
    const FFRtxPixFmt *in_tbl;
    int                n_in;
    const FFRtxPixFmt *out_tbl;    ///< NULL: the output is always the input format
    int                n_out;
    const char        *out_format; ///< the `format` option; NULL/empty = same as input
    const char        *hint;       ///< appended to a rejection, e.g. "use rgb0/rgba"
} FFRtxFormats;

/**
 * The config_output prologue every filter shares: require a CUDA hwframe input,
 * look its sw_format up in the input table, and resolve the output format --
 * the `format` option when set, else the input format -- in the output table.
 * @p outpf may be NULL for a filter whose output format is its input format.
 */
int ff_rtx_config_formats(AVFilterContext *ctx, AVFilterLink *inlink,
                          const FFRtxFormats *f,
                          AVHWFramesContext **in_frames_ctx,
                          const FFRtxPixFmt **inpf, const FFRtxPixFmt **outpf);

/**
 * Take a reference on the input frames context's device and cache the CUDA
 * context and stream.  Must be called before anything else touches @p r.
 */
int ff_rtx_bind_device(AVFilterContext *ctx, FFRtxCuda *r,
                       AVHWFramesContext *in_frames_ctx);

/**
 * Set the output link's size and build its CUDA frames context.  Call after
 * ff_rtx_bind_device() and before ff_rtx_setup().
 */
int ff_rtx_config_hwframes(AVFilterContext *ctx, AVFilterLink *outlink,
                           FFRtxCuda *r, int oW, int oH,
                           enum AVPixelFormat sw_format);

/**
 * Push the CUDA context, run @p setup_graph, pop it again.  @p what names the
 * graph in the failure message.
 */
int ff_rtx_setup(AVFilterContext *ctx, FFRtxCuda *r, const char *what,
                 int (*setup_graph)(AVFilterContext *ctx));

/* ------------------------------------------------------------------------- *
 * Graph setup (all of these need the CUDA context current)
 * ------------------------------------------------------------------------- */
/**
 * How far a feature's cubins have been validated.  The shared policy is that
 * Blackwell (cc 12.x) and Ada (cc 8.9) run ungated -- those are the two the
 * cubins were checked byte-exact on -- and every other architecture needs
 * experimental_arch, because its images were matched statically rather than
 * exercised.  What differs per feature is the wording and whether there is a
 * floor below which no image exists at all.
 */
typedef struct FFRtxArchGate {
    int         hard_min_major; ///< refuse cc_major below this outright; 0 = no floor
    const char *hard_msg;       ///< printf'd with cc_major, cc_minor
    const char *gate_msg;       ///< refusal when unvalidated and not opted in
    const char *warn_msg;       ///< warning when running the opted-in path
} FFRtxArchGate;

int ff_rtx_arch_gate(AVFilterContext *ctx, FFRtxCuda *r,
                     const FFRtxArchGate *gate, int experimental);

/**
 * Load every cubin named by @p mods out of @p dir and resolve every kernel in
 * @p funcs, into r->mod[]/r->fn[] sized for @p max_mid / @p max_fid.
 * @p load_hint, if set, is appended to a module-load failure (which is nearly
 * always "this data dir has no image for the running GPU").
 */
int ff_rtx_load_modules(AVFilterContext *ctx, FFRtxCuda *r, const char *dir,
                        const FFRtxModule *mods, int nmod, int max_mid,
                        const FFRtxFunc *funcs, int nfunc, int max_fid,
                        const char *load_hint);

#define FF_RTX_ARENA_ZERO 1  ///< memset the arena before the weights land in it

/**
 * Allocate the graph's scratch and weight buffers as ONE contiguous arena and
 * hand out r->alloc[0..nalloc-1] into it.
 *
 * @p fill_sizes is the feature's generated allocation model: it writes the
 * largest size asked for each ordinal, as the driver's own allocator does.
 *
 * Contiguity is load-bearing, not tidiness: several kernels do a tile/halo read
 * a little past the logical end of their input buffer.  That is harmless while
 * the following bytes are mapped, which inside one arena they always are (an
 * adjacent buffer, or the trailing guard).  With a separate allocation per
 * buffer they scatter, and after a filter-graph rebuild (an mpv seek, say) the
 * heap fragments until the bytes past a buffer are an unmapped hole -- turning
 * the benign over-read into a CUDA_ERROR_ILLEGAL_ADDRESS that poisons the
 * context.
 */
int ff_rtx_alloc_arena(AVFilterContext *ctx, FFRtxCuda *r, int nalloc,
                       void (*fill_sizes)(AVFilterContext *ctx, long long *sz),
                       unsigned flags);

/**
 * Arrange for ff_rtx_reset_arena() to restore the arena to its post-upload
 * state.  For graphs that read scratch before writing it: a fresh process gets
 * zeroed pages from cuMemAlloc and is byte-exact, but a long-running host
 * recycles dirty memory, so the arena has to be put back between frames.
 *
 * Only the uploaded prefix is snapshotted.  ff_rtx_alloc_arena() zeroed the
 * whole arena and the uploads then wrote a prefix of it, so everything past the
 * last uploaded byte is known to be zero -- the reset can memset it instead of
 * copying it back, which is bit-identical and much cheaper (a device-to-device
 * copy reads and writes, a memset only writes).  Call after the uploads.
 */
int ff_rtx_snapshot_arena(AVFilterContext *ctx, FFRtxCuda *r);
int ff_rtx_reset_arena(AVFilterContext *ctx, FFRtxCuda *r);

/**
 * Map @p dir/@p file and run every upload in @p up into the arena.  One
 * weights blob per data dir holds each distinct payload exactly once -- the
 * qualities of a feature share most layers, and the driver plugins share all of
 * them -- so a config's uploads index into it by the generated file offset.
 */
int ff_rtx_upload_weights(AVFilterContext *ctx, FFRtxCuda *r, const char *dir,
                          const char *file, const FFRtxUpload *up, int nup);

/** Allocate the launch array the feature's fill_graph() will populate. */
int ff_rtx_alloc_launches(AVFilterContext *ctx, FFRtxCuda *r,
                          int nlaunch, size_t launch_size);

/* Image binding flags. */
#define FF_RTX_TEX   (1 << 0)  ///< create a bindless texture over the image
#define FF_RTX_SURF  (1 << 1)  ///< create a bindless surface over the image
#define FF_RTX_LDST  (1 << 2)  ///< array is SURFACE_LDST capable
#define FF_RTX_CLAMP (1 << 3)  ///< texture address mode CLAMP (else the default WRAP)
#define FF_RTX_ZERO  (1 << 4)  ///< zero the backing store (pitched images only)

/**
 * Bind a W x H image the graph can read and/or write.  Textures are always
 * created linear-filtered with normalized coordinates, which is what the
 * captured graphs sample with.
 *
 * ff_rtx_image_array()  -- a CUDA array, the usual input texture / output surface
 * ff_rtx_image_pitch()  -- pitched linear memory bound as a PITCH2D texture
 * ff_rtx_image_linear() -- a plain packed buffer, no texture or surface
 *
 * The returned pointer is owned by @p r and stays valid until
 * ff_rtx_free_graph(); NULL means the image could not be created (the reason is
 * already logged).
 */
FFRtxImage *ff_rtx_image_array(AVFilterContext *ctx, FFRtxCuda *r, int W, int H,
                               CUarray_format cufmt, unsigned flags);
FFRtxImage *ff_rtx_image_pitch(AVFilterContext *ctx, FFRtxCuda *r, int W, int H,
                               CUarray_format cufmt, int bpp, unsigned flags);
FFRtxImage *ff_rtx_image_linear(AVFilterContext *ctx, FFRtxCuda *r,
                                size_t size, size_t pitch);

/**
 * Bind a texture over pitched memory the caller owns -- an input frame's own
 * plane, say -- rather than over an image this core allocated.  Same descriptor
 * as ff_rtx_image_pitch() gives, so a filter that binds both ways samples both
 * the same; the caller owns the handle and destroys it.
 */
int ff_rtx_tex_over_pitch(AVFilterContext *ctx, FFRtxCuda *r, CUdeviceptr ptr,
                          size_t pitch, int W, int H, CUarray_format cufmt,
                          unsigned flags, CUtexObject *tex);

/**
 * Index of the first launch running kernel @p name, or -1.  For the features
 * whose generated config does not yet carry the launch index of a tunable's
 * kernel the way VSR's sel_launch does.
 */
int ff_rtx_find_launch(const FFRtxCuda *r, const FFRtxFunc *funcs, int nfunc,
                       const char *name);
/** As ff_rtx_find_launch(), matching a kernel-name prefix. */
int ff_rtx_find_launch_prefix(const FFRtxCuda *r, const FFRtxFunc *funcs, int nfunc,
                              const char *prefix);

/* ------------------------------------------------------------------------- *
 * Per-frame replay
 * ------------------------------------------------------------------------- */
/* ff_rtx_filter_frame() flags. */
#define FF_RTX_OP_PSIZE        (1 << 0)  ///< launch with the kernel's own cbank size
#define FF_RTX_OP_RESET_ARENA  (1 << 1)  ///< restore the pristine arena before each frame
#define FF_RTX_OP_OPAQUE_ALPHA (1 << 2)  ///< force opaque alpha over the output

/**
 * What one frame through a configured graph consists of.  in_img and out_img
 * are the same image for a filter that works in place.
 */
typedef struct FFRtxFrameOp {
    const FFRtxImage *in_img, *out_img;
    int iW, iH, ibpp;
    int oW, oH, obpp;
    unsigned flags;
    /** Replace ff_rtx_launch_all() -- for a graph the launch list cannot
     *  describe on its own, like ISR's per-tile pointer cursor. */
    int (*run)(AVFilterContext *ctx);
} FFRtxFrameOp;

/**
 * One whole frame: take the output buffer, copy the input frame's properties,
 * push the CUDA context, replay the graph over the frame, pop, and forward the
 * result.  @p retag, if set, adjusts the output frame's properties (the TrueHDR
 * filters retag SDR input as HDR) before the graph runs.  Consumes @p in.
 */
int ff_rtx_filter_frame(AVFilterLink *inlink, AVFrame *in, FFRtxCuda *r,
                        const FFRtxFrameOp *op,
                        void (*retag)(AVFilterContext *ctx, AVFrame *out));

/** Issue one launch.  @p params is its argument block, @p psize its cbank size. */
int ff_rtx_launch(AVFilterContext *ctx, FFRtxCuda *r, int fnid,
                  const unsigned grid[3], const unsigned block[3], unsigned smem,
                  void *params, size_t psize);

/**
 * Issue the whole launch list in order.  @p use_psize selects the kernel's own
 * EIATTR_CBANK_PARAM_SIZE rather than the captured driver argsize -- the driver
 * over-reports for some DLPP tex/surf kernels, which makes cuLaunchKernel fail
 * with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES.
 */
int ff_rtx_launch_all(AVFilterContext *ctx, FFRtxCuda *r, int use_psize);

/** Copy a pitched input frame into the graph's input image. */
int ff_rtx_frame_to_image(AVFilterContext *ctx, FFRtxCuda *r, const AVFrame *in,
                          const FFRtxImage *img, int W, int H, int bpp);
/** Copy the graph's output image back out into a pitched frame. */
int ff_rtx_image_to_frame(AVFilterContext *ctx, FFRtxCuda *r,
                          const FFRtxImage *img, AVFrame *out,
                          int W, int H, int bpp);

/**
 * Force opaque alpha over @p out.  The resample store kernel
 * (dlpp_ResampleAndComposeFP16) omits the alpha write in its formatted path --
 * unlike postProcess, which stores 1.0 -- so >= 10-bit output on the resample
 * path would come out fully transparent (RGB is correct; verified in SASS, and
 * the SDK DLL has the same omission).  These networks always produce opaque
 * output, so this runs for any sel-2 output: it fixes the resample case and is
 * a harmless no-op on the fast path.
 */
int ff_rtx_fill_opaque_alpha(AVFilterContext *ctx, FFRtxCuda *r, AVFrame *out,
                             int oW, int oH, int bpp);

/* ------------------------------------------------------------------------- *
 * Teardown and helpers
 * ------------------------------------------------------------------------- */
/**
 * Release everything ff_rtx_* built, against the CUDA context it was built on,
 * and reset @p r so a graph can be built again.  Safe when nothing is
 * configured.  config_output() may run more than once -- a mid-stream
 * reconfigure, or a media player rebuilding its filter graph on seek -- so this
 * must leave no leaked allocation and no stale device pointer baked into a
 * launch argument block.
 */
void ff_rtx_free_graph(AVFilterContext *ctx, FFRtxCuda *r);

/**
 * Evaluate the `w`/`h` output-size expressions over in_w/iw/in_h/ih.  An unset
 * or empty expression means @p defscale x the input.
 */
int ff_rtx_eval_dims(AVFilterContext *ctx, AVFilterLink *inlink,
                     const char *w_expr, const char *h_expr, int defscale,
                     int *oW, int *oH);

/**
 * TrueHDR's internal network resolution: shorter side -> 544, longer side
 * aspect-scaled and quantized to a multiple of 32.  Must bit-match the float32
 * arithmetic of rtxv.fit.truehdr.nn_dims (verified byte-exact across 28
 * resolutions).
 */
void ff_rtx_nn_dims(int W, int H, int *NW, int *NH);

#endif /* AVFILTER_RTX_CUDA_H */
