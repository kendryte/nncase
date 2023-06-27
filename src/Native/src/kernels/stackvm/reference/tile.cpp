/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

template <typename T> static void copy_data(T *dst, const T *src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
result<void> tile_impl(const T *input, T *output,
                       gsl::span<const size_t> in_shape,
                       gsl::span<const size_t> out_shape,
                       [[maybe_unused]] gsl::span<const size_t> in_strides,
                       [[maybe_unused]] gsl::span<const size_t> out_strides,
                       [[maybe_unused]] gsl::span<const size_t> &repeats) {
    result<void> ret_v = ok();

    int shape_size_in[4] = {1, 1, 1, 1};
    int shape_size_out[4] = {1, 1, 1, 1};
    int repeat_size[4] = {1, 1, 1, 1};
    for (int i = in_shape.size() - 1, j = 0; i >= 0; --i, ++j) {
        shape_size_in[j] = in_shape[i];
        shape_size_out[j] = out_shape[i];
    }

    for (int i = repeats.size() - 1, j = 0; i >= 0; --i, ++j) {
        repeat_size[j] = repeats[i];
    }

    int w = shape_size_in[0];
    int h = shape_size_in[1];
    int d = shape_size_in[2];
    int c = shape_size_in[3];

    int wd = shape_size_out[0];
    int hd = shape_size_out[1];
    int dd = shape_size_out[2];
    // int cd = shape_size_out[3];

    int repeat_w = repeat_size[0];
    int repeat_h = repeat_size[1];
    int repeat_d = repeat_size[2];
    int repeat_c = repeat_size[3];

    for (int ci = 0; ci < c; ++ci) {
        // channel_step = ci * h * d * w;
        // dst_channel_step = ci * hd * dd * wd;
        for (int di = 0; di < d; ++di) {
            for (int hi = 0; hi < h; ++hi) {
                const T *src = input + ci * h * d * w + di * h * w + hi * w;
                T *dst = output + ci * hd * dd * wd + di * hd * wd + hi * wd;
                for (int i = 0; i < repeat_w; ++i) {
                    copy_data(dst, src, w);
                    dst += w;
                }
            }
        }
        for (int di = 0; di < d; ++di) {
            T *dst = output + ci * hd * dd * wd + di * hd * wd;
            int size_x = h * wd;
            T *dst1 = dst + size_x;
            for (int i = 1; i < repeat_h; ++i) {
                copy_data(dst1, dst, size_x);
                dst1 += size_x;
            }
        }
        {
            T *dst = output + ci * hd * dd * wd;
            int size_x = d * hd * wd;
            T *dst1 = dst + size_x;
            for (int i = 1; i < repeat_d; ++i) {
                copy_data(dst1, dst, size_x);
                dst1 += size_x;
            }
        }
    }

    {
        T *dst = output;
        int size_x = c * dd * hd * wd;
        T *dst1 = dst + size_x;
        for (int i = 1; i < repeat_c; ++i) {
            copy_data(dst1, dst, size_x);
            dst1 += size_x;
        }
    }

    return ret_v;
}

#define TILE_IMPL(_ty)                                                         \
    return tile_impl(IN_CAST(_ty, input), OUT_CAST(_ty, output), in_shape,     \
                     out_shape, in_strides, out_strides, repeats);

result<void> nncase::kernels::stackvm::reference::tile(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    gsl::span<const size_t> repeats) {
    if (in_shape.size() > 4) {
        return stackvm::reference::tile(dt, input, output, in_shape, out_shape,
                                        in_strides, out_strides, repeats);
    }
    try_var(tycode, to_typecode(dt));
    TYPE_SELECT(tycode, TILE_IMPL);
}