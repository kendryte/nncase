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
#include "opt_ops.h"
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {

template <class T>
result<void> resize_bilinear_impl(
    const T *input, T *output, gsl::span<const size_t> in_shape,
    NNCASE_UNUSED gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides, int32_t out_h,
    int32_t out_w, bool align_corners, NNCASE_UNUSED bool half_pixel_centers,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto scales = kernels::detail::get_resize_scales(in_shape, out_h, out_w,
                                                     align_corners);
    auto height_scale = scales.first;
    auto width_scale = scales.second;
    const float rounding_offset =
        std::numeric_limits<T>::is_integer ? .5f : .0f;
    dims_t in_index(4), out_index(4);

    const auto in_img_size = in_shape[2] * in_shape[3];
    const auto out_img_size = out_w * out_h;

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        auto in_batch = input + (size_t)batch * in_shape[1] * in_img_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_w * out_h;
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(                                          \
    kernels::default_kernel_context().num_threads)
#endif
        for (int oc = 0; oc < in_shape[1]; oc++) {
            auto in_c = in_batch + (size_t)oc * in_img_size;
            auto *output_ptr = begin_output_ptr + oc * out_img_size;
            for (int oy = 0; oy < out_h; oy++) {
                float in_y;
                int32_t in_y0, in_y1;
                kernels::detail::set_resize_bilinear(
                    oy, height_scale, half_pixel_centers, in_shape[2], in_y,
                    in_y0, in_y1);

                for (int ox = 0; ox < out_w; ox++) {
                    float in_x;
                    int32_t in_x0, in_x1;
                    kernels::detail::set_resize_bilinear(
                        ox, width_scale, half_pixel_centers, in_shape[3], in_x,
                        in_x0, in_x1);

                    auto v0 = in_c[in_y0 * in_shape[3] + in_x0];
                    auto v1 = in_c[in_y1 * in_shape[3] + in_x0];
                    auto v2 = in_c[in_y0 * in_shape[3] + in_x1];
                    auto v3 = in_c[in_y1 * in_shape[3] + in_x1];

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);

                    *output_ptr++ = T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 +
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
    NNCASE_UNUSED gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides, int32_t out_h,
    int32_t out_w, NNCASE_UNUSED bool align_corners,
    NNCASE_UNUSED bool half_pixel_centers,
    get_coordinate_func_t get_coordinate_func,
    get_nearest_pixel_func_t get_nearset_func,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto scales = kernels::detail::get_resize_scales(in_shape, out_h, out_w,
                                                     align_corners);
    auto height_scale = scales.first;
    auto width_scale = scales.second;

    const auto in_image_size = in_shape[2] * in_shape[3];
    const auto out_image_size = out_h * out_w;
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        auto *begin_input_ptr = input + batch * in_shape[1] * in_image_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_image_size;
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(                                          \
    kernels::default_kernel_context().num_threads)
#endif
        for (int oc = 0; oc < in_shape[1]; oc++) {
            auto *input_ptr = begin_input_ptr + oc * in_image_size;
            auto *output_ptr = begin_output_ptr + oc * out_image_size;

            for (int oy = 0; oy < out_h; oy++) {
                auto iy = get_coordinate_func(oy, height_scale, out_h, 0, 0, 0);
                int64_t in_y = get_nearset_func(iy);
                auto *in_row = input_ptr + in_y * in_shape[3];

                for (int ox = 0; ox < out_w; ox++) {
                    auto ix =
                        get_coordinate_func(ox, width_scale, out_w, 0, 0, 0);
                    int64_t in_x = get_nearset_func(ix);
                    *output_ptr++ = in_row[in_x];
                }
            }
        }
    }
    return ok();
}

inline result<void> gnne_resize_nearest_neighbor(
    const bfloat16 *input, bfloat16 *output, gsl::span<const size_t> in_shape,
    NNCASE_UNUSED gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides, int32_t out_h,
    int32_t out_w, NNCASE_UNUSED bool align_corners,
    NNCASE_UNUSED bool half_pixel_centers,
    NNCASE_UNUSED kernel_context &context) {
    if (align_corners || half_pixel_centers) {
        return err(std::errc::not_supported);
    }

    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;

    const auto in_image_size = in_shape[2] * in_shape[3];
    const auto out_image_size = out_h * out_w;
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        auto *begin_input_ptr = input + batch * in_shape[1] * in_image_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_image_size;
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(                                          \
    kernels::default_kernel_context().num_threads)
#endif
        for (int oc = 0; oc < in_shape[1]; oc++) {
            auto *input_ptr = begin_input_ptr + oc * in_image_size;
            auto *output_ptr = begin_output_ptr + oc * out_image_size;

            for (int oy = 0; oy < out_h; oy++) {
                auto in_y = std::min((int32_t)floorf(oy * height_scale),
                                     (int32_t)in_shape[2] - 1);
                auto *in_row = input_ptr + in_y * in_shape[3];

                for (int ox = 0; ox < out_w; ox++) {
                    auto in_x = std::min((int32_t)floorf(ox * width_scale),
                                         (int32_t)in_shape[3] - 1);
                    *output_ptr++ = in_row[in_x];
                }
            }
        }
    }
    return ok();
}

inline result<void> resize_bilinear_impl(
    const bfloat16 *input, bfloat16 *output, gsl::span<const size_t> in_shape,
    NNCASE_UNUSED gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides, int32_t out_h,
    int32_t out_w, bool align_corners, NNCASE_UNUSED bool half_pixel_centers,
    NNCASE_UNUSED kernel_context &context) {
    if (half_pixel_centers) {
        return err(std::errc::not_supported);
    }

    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;
    if (align_corners && out_h > 1)
        height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
    if (align_corners && out_w > 1)
        width_scale = (float)(in_shape[3] - 1) / (out_w - 1);

    const auto in_img_size = in_shape[2] * in_shape[3];
    const auto out_img_size = out_w * out_h;

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        auto in_batch = input + (size_t)batch * in_shape[1] * in_img_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_w * out_h;
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(                                          \
    kernels::default_kernel_context().num_threads)
#endif
        for (int oc = 0; oc < in_shape[1]; oc++) {
            auto in_c = in_batch + (size_t)oc * in_img_size;
            auto *output_ptr = begin_output_ptr + oc * out_img_size;
            for (int oy = 0; oy < out_h; oy++) {
                auto in_y = oy * height_scale;
                auto in_y0 = (int)floorf(in_y);
                auto in_y1 = std::min(in_y0 + 1, (int32_t)in_shape[2] - 1);

                for (int ox = 0; ox < out_w; ox++) {
                    auto in_x = ox * width_scale;
                    auto in_x0 = (int)floorf(in_x);
                    auto in_x1 = std::min(in_x0 + 1, (int32_t)in_shape[3] - 1);

                    auto v0 = in_c[in_y0 * in_shape[3] + in_x0];
                    auto v1 = in_c[in_y1 * in_shape[3] + in_x0];
                    auto v2 = in_c[in_y0 * in_shape[3] + in_x1];
                    auto v3 = in_c[in_y1 * in_shape[3] + in_x1];

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);

                    *output_ptr = bfloat16::round_to_bfloat16(
                        v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3);
                    ++output_ptr;
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
    case dt_bfloat16:                                                          \
        return KERNEL(bfloat16);                                               \
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
    resize_nearest_neighbor_impl(                                              \
        reinterpret_cast<const type *>(input),                                 \
        reinterpret_cast<type *>(output), in_shape, in_strides, out_strides,   \
        out_h, out_w, align_corners, half_pixel_centers, get_coordinate_func,  \
        get_nearset_func, context);

result<void> optimized::resize_bilinear(
    typecode_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept {
    FP_OR_Q_IMPL(type, RESIZE_BILINEAR_IMPL);
}

result<void> optimized::resize_nearest_neighbor(
    typecode_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    get_coordinate_func_t get_coordinate_func,
    get_nearest_pixel_func_t get_nearset_func,
    kernel_context &context) noexcept {
    FP_OR_Q_IMPL(type, RESIZE_NEAREST_NEIGHBOR_IMPL);
}