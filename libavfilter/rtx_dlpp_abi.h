/*
 * The DLPP kernel ABI, shared by the two filters that drive those kernels.
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
 * vf_vsr_drv_cuda and vf_dlpp_drv_cuda replay two different driver plugins, but
 * the glue kernels around their networks are literally the same kernels -- the
 * DLPP pre/post-process and resample-and-compose -- so the argument-block
 * offsets of the format selectors are one ABI, not two.  They were captured
 * once; this is where they live, so a re-capture cannot update one filter and
 * leave the other patching a stale offset.
 *
 * What stays per filter is what genuinely differs: which tunables that filter
 * exposes and where they sit (vsr_drv's detail/smooth on preProcess, dlpp_drv's
 * wipe on the SR head).
 */

#ifndef AVFILTER_RTX_DLPP_ABI_H
#define AVFILTER_RTX_DLPP_ABI_H

#include <stdint.h>
#include <string.h>

#include "avfilter.h"
#include "rtx_cuda.h"

/* The input kernel, and the two alternative output-store kernels: the fast path
 * ends in postProcess, the resample path in ResampleAndComposeFP16. */
#define FF_DLPP_PRE_KERNEL       "dlpp_preProcess"
#define FF_DLPP_POST_KERNEL      "dlpp_postProcess"
#define FF_DLPP_RESAMPLE_KERNEL  "dlpp_ResampleAndComposeFP16"

/* Format-selector offsets in each kernel's params block. */
#define FF_DLPP_PRE_FMT_OFF      0x30
#define FF_DLPP_POST_FMT_OFF     0x40
#define FF_DLPP_RESAMPLE_FMT_OFF 0x50

/**
 * Patch the input and output format selectors of a DLPP-family graph.
 *
 * The captured argbufs carry sel 0 (RGBA8); a chosen format that is not sel 0
 * needs the kernel told, so a selector that cannot be found is a bug in the
 * generated tables rather than something to skip -- silently leaving sel 0 in
 * place would emit a whole encode with red and blue transposed.
 *
 * @param pre_out receives the preProcess launch index, which is also where both
 *                filters' own preProcess tunables live; -1 when there is none
 *                and the format did not need one.
 */
static inline int ff_dlpp_patch_selectors(AVFilterContext *ctx, FFRtxCuda *r,
                                          const FFRtxFunc *funcs, int nfunc,
                                          const FFRtxPixFmt *inpf,
                                          const FFRtxPixFmt *outpf,
                                          const char *tag, int *pre_out)
{
    int pre   = ff_rtx_find_launch(r, funcs, nfunc, FF_DLPP_PRE_KERNEL);
    int store = ff_rtx_find_launch(r, funcs, nfunc, FF_DLPP_POST_KERNEL);
    int store_fmt_off = FF_DLPP_POST_FMT_OFF;

    if (store < 0) {
        store = ff_rtx_find_launch(r, funcs, nfunc, FF_DLPP_RESAMPLE_KERNEL);
        store_fmt_off = FF_DLPP_RESAMPLE_FMT_OFF;
    }
    *pre_out = pre;

    if (inpf->sel) {
        uint32_t sel = inpf->sel;
        if (pre < 0) {
            av_log(ctx, AV_LOG_ERROR, "no input pre-process kernel for config %s\n", tag);
            return AVERROR_BUG;
        }
        memcpy(ff_rtx_launch_at(r, pre)->params + FF_DLPP_PRE_FMT_OFF, &sel, 4);
    }
    if (outpf->sel) {
        uint32_t sel = outpf->sel;
        if (store < 0) {
            av_log(ctx, AV_LOG_ERROR, "no output-store kernel for config %s\n", tag);
            return AVERROR_BUG;
        }
        memcpy(ff_rtx_launch_at(r, store)->params + store_fmt_off, &sel, 4);
    }
    return 0;
}

#endif /* AVFILTER_RTX_DLPP_ABI_H */
