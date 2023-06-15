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
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

namespace {
template <class T>
result<void> resize_bilinear_impl(
    const T *input, T *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    int32_t out_h, int32_t out_w, bool align_corners,
    NNCASE_UNUSED bool half_pixel_centers,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto scales = kernels::detail::get_resize_scales(in_shape, out_h, out_w,
                                                     align_corners);
    auto height_scale = scales.first;
    auto width_scale = scales.second;

    const float rounding_offset =
        std::numeric_limits<T>::is_integer ? .5f : .0f;
    dims_t in_index(4), out_index(4);

    auto get_input = [&](int32_t in_y, int32_t in_x) {
        in_index[2] = in_y;
        in_index[3] = in_x;
        return input[offset(in_strides, in_index)];
    };

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++) {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++) {
                out_index[2] = oy;
                float in_y;
                int32_t in_y0, in_y1;
                kernels::detail::set_resize_bilinear(
                    oy, height_scale, half_pixel_centers, in_shape[2], in_y,
                    in_y0, in_y1);

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    out_index[3] = ox;
                    float in_x;
                    int32_t in_x0, in_x1;
                    kernels::detail::set_resize_bilinear(
                        ox, width_scale, half_pixel_centers, in_shape[3], in_x,
                        in_x0, in_x1);

                    auto v0 = get_input(in_y0, in_x0);
                    auto v1 = get_input(in_y1, in_x0);
                    auto v2 = get_input(in_y0, in_x1);
                    auto v3 = get_input(in_y1, in_x1);

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);
                    output[offset(out_strides, out_index)] =
                        T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 +
                          rounding_offset);
                }
            }
        }
    }
    return ok();
}

template <class T>
result<void> resize_nearest_neighbor_impl(
    const T *input, T *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto scales = kernels::detail::get_resize_scales(in_shape, out_h, out_w,
                                                     align_corners);
    auto height_scale = scales.first;
    auto width_scale = scales.second;

    dims_t in_index(4), out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++) {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++) {
                auto in_y = kernels::detail::get_nearest_neighbor(
                    oy, in_shape[2], height_scale, align_corners,
                    half_pixel_centers);
                in_index[2] = in_y;
                out_index[2] = oy;

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    auto in_x = kernels::detail::get_nearest_neighbor(
                        ox, in_shape[3], width_scale, align_corners,
                        half_pixel_centers);
                    in_index[3] = in_x;
                    out_index[3] = ox;
                    output[offset(out_strides, out_index)] =
                        input[offset(in_strides, in_index)];
                }
            }
        }
    }
    return ok();
}

#define FP_OR_Q_IMPL(type, KERNEL)                                             \
    switch (type) {                                                            \
    case dt_float32:                                                           \
        return KERNEL(float);                                                  \
    case dt_int8:                                                              \
    case dt_uint8:                                                             \
        return KERNEL(uint8_t);                                                \
    case dt_int16:                                                             \
    case dt_uint16:                                                            \
        return KERNEL(uint16_t);                                               \
    case dt_int32:                                                             \
    case dt_uint32:                                                            \
        return KERNEL(uint32_t);                                               \
    case dt_int64:                                                             \
    case dt_uint64:                                                            \
        return KERNEL(uint64_t);                                               \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

#define RESIZE_BILINEAR_IMPL(type)                                             \
    resize_bilinear_impl(reinterpret_cast<const type *>(input),                \
                         reinterpret_cast<type *>(output), in_shape,           \
                         in_strides, out_strides, out_h, out_w, align_corners, \
                         half_pixel_centers, context);

#define RESIZE_NEAREST_NEIGHBOR_IMPL(type)                                     \
    resize_nearest_neighbor_impl(reinterpret_cast<const type *>(input),        \
                                 reinterpret_cast<type *>(output), in_shape,   \
                                 in_strides, out_strides, out_h, out_w,        \
                                 align_corners, half_pixel_centers, context);
} // namespace

result<void> nncase::kernels::stackvm::reference::resize_bilinear(
    typecode_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept {
    FP_OR_Q_IMPL(type, RESIZE_BILINEAR_IMPL);
}

result<void> nncase::kernels::stackvm::reference::resize_nearest_neighbor(
    typecode_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept {
    FP_OR_Q_IMPL(type, RESIZE_NEAREST_NEIGHBOR_IMPL);
}