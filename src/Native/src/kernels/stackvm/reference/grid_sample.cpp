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
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T>
result<void> grid_sample_impl(
    const T *input, const T *grid, T *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> grid_shape,
    gsl::span<const size_t> grid_strides, gsl::span<const size_t> out_strides,
    grid_sample_align_corners_t align_corners,
    NNCASE_UNUSED grid_sample_mode_t mode,
    NNCASE_UNUSED grid_sample_padding_mode_t padding_mode,
    NNCASE_UNUSED kernel_context &context) noexcept {
    dims_t in_index(4), grid_index(4), out_index(4);
    const int32_t H_in = in_shape[2];
    const int32_t W_in = in_shape[3];

    auto get_input = [&](int32_t in_y, int32_t in_x) {
        if (padding_mode == grid_sample_padding_mode_t::border) {
            in_x = std::max(0, std::min(in_x, W_in - 1));
            in_y = std::max(0, std::min(in_y, H_in - 1));
        } else if (in_x < 0 || in_x >= W_in || in_y < 0 || in_y >= H_in) {
            return T(0);
        }

        in_index[2] = in_y;
        in_index[3] = in_x;
        return input[offset(in_strides, in_index)];
    };

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        grid_index[0] = batch;
        out_index[0] = batch;
        for (size_t oh = 0; oh < grid_shape[1]; oh++) {
            grid_index[1] = oh;
            out_index[2] = oh;
            for (size_t ow = 0; ow < grid_shape[2]; ow++) {
                grid_index[2] = ow;
                out_index[3] = ow;

                grid_index[3] = 0;
                auto x_norm = grid[offset(grid_strides, grid_index)];
                grid_index[3] = 1;
                auto y_norm = grid[offset(grid_strides, grid_index)];

                float x_src, y_src;
                int32_t x_w, x_e, y_n, y_s;
                if (align_corners ==
                    grid_sample_align_corners_t::align_corners) {
                    x_src = (x_norm + 1) / 2 * (in_shape[3] - 1);
                    y_src = (y_norm + 1) / 2 * (in_shape[2] - 1);
                } else {
                    x_src = ((x_norm + 1) * in_shape[3] - 1) / 2;
                    y_src = ((y_norm + 1) * in_shape[2] - 1) / 2;
                }
                x_w = (int32_t)std::floor(x_src);
                y_n = (int32_t)std::floor(y_src);
                x_e = x_w + 1;
                y_s = y_n + 1;

                float lw = x_src - x_w;
                float rw = 1.0 - lw;
                float nw = y_src - y_n;
                float sw = 1.0 - nw;

                float w_nw = rw * sw;
                float w_ne = lw * sw;
                float w_sw = rw * nw;
                float w_se = lw * nw;

                for (size_t oc = 0; oc < in_shape[1]; oc++) {
                    {
                        in_index[1] = oc;
                        out_index[1] = oc;
                        auto val_nw = get_input(y_n, x_w) * w_nw;
                        auto val_ne = get_input(y_n, x_e) * w_ne;
                        auto val_sw = get_input(y_s, x_w) * w_sw;
                        auto val_se = get_input(y_s, x_e) * w_se;
                        output[offset(out_strides, out_index)] =
                            T(val_nw + val_ne + val_sw + val_se);
                    }
                }
            }
        }
    }
    return ok();
}
} // namespace

#define FP_OR_Q_IMPL(type, KERNEL)                                             \
    switch (type) {                                                            \
    case dt_float32:                                                           \
        return KERNEL(float);                                                  \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

#define GRID_SAMPLE_IMPL(type)                                                 \
    grid_sample_impl(reinterpret_cast<const type *>(input),                    \
                     reinterpret_cast<const type *>(grid),                     \
                     reinterpret_cast<type *>(output), in_shape, in_strides,   \
                     grid_shape, grid_strides, out_strides, align_corners,     \
                     mode, padding_mode, context);

result<void> nncase::kernels::stackvm::reference::grid_sample(
    typecode_t type, const gsl::byte *input, const gsl::byte *grid,
    gsl::byte *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> grid_shape,
    gsl::span<const size_t> grid_strides, gsl::span<const size_t> out_strides,
    grid_sample_align_corners_t align_corners, grid_sample_mode_t mode,
    grid_sample_padding_mode_t padding_mode, kernel_context &context) noexcept {
    FP_OR_Q_IMPL(type, GRID_SAMPLE_IMPL);
}
