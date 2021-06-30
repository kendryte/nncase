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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{

template <class T>
result<void> resize_bilinear_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
                                  const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, NNCASE_UNUSED bool half_pixel_centers, kernel_context &context) noexcept
{
    auto [height_scale, width_scale] = kernels::detail::get_resize_scales(in_shape, out_h, out_w, align_corners);
    const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;
    runtime_shape_t in_index(4), out_index(4);

    const auto in_img_size = in_shape[2] * in_shape[3];
    const auto out_img_size = out_w * out_h;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        auto in_batch = input + (size_t)batch * in_shape[1] * in_img_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_w * out_h;
#pragma omp parallel for num_threads(kernels::default_kernel_context().num_threads)
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            auto in_c = in_batch + (size_t)oc * in_img_size;
            auto *output_ptr = begin_output_ptr + oc * out_img_size;
            for (int oy = 0; oy < out_h; oy++)
            {
                float in_y;
                int32_t in_y0, in_y1;
                kernels::detail::set_resize_bilinear(oy, height_scale, half_pixel_centers, in_shape[2], in_y, in_y0, in_y1);

                for (int ox = 0; ox < out_w; ox++)
                {
                    float in_x;
                    int32_t in_x0, in_x1;
                    kernels::detail::set_resize_bilinear(ox, width_scale, half_pixel_centers, in_shape[3], in_x, in_x0, in_x1);

                    auto v0 = in_c[in_y0 * in_shape[3] + in_x0];
                    auto v1 = in_c[in_y1 * in_shape[3] + in_x0];
                    auto v2 = in_c[in_y0 * in_shape[3] + in_x1];
                    auto v3 = in_c[in_y1 * in_shape[3] + in_x1];

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);

                    *output_ptr++ = T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 + rounding_offset);
                }
            }
        }
    }
    return ok();
}

template <class T>
result<void> resize_nearest_neighbor_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
                                          const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers, kernel_context &context) noexcept
{
    auto [height_scale, width_scale] = kernels::detail::get_resize_scales(in_shape, out_h, out_w, align_corners);

    const auto in_image_size = in_shape[2] * in_shape[3];
    const auto out_image_size = out_h * out_w;
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        auto *begin_input_ptr = input + batch * in_shape[1] * in_image_size;
        auto *begin_output_ptr = output + batch * in_shape[1] * out_image_size;
#pragma omp parallel for num_threads(kernels::default_kernel_context().num_threads)
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            auto *input_ptr = begin_input_ptr + oc * in_image_size;
            auto *output_ptr = begin_output_ptr + oc * out_image_size;

            for (int oy = 0; oy < out_h; oy++)
            {
                auto in_y = kernels::detail::get_nearest_neighbor(oy, in_shape[2], height_scale, align_corners, half_pixel_centers);
                auto *in_row = input_ptr + in_y * in_shape[3];

                for (int ox = 0; ox < out_w; ox++)
                {
                    auto in_x = kernels::detail::get_nearest_neighbor(ox, in_shape[3], width_scale, align_corners, half_pixel_centers);
                    *output_ptr++ = in_row[in_x];
                }
            }
        }
    }
    return ok();
}
}

#define FP_OR_Q_IMPL(type, KERNEL)            \
    switch (type)                             \
    {                                         \
    case dt_float32:                          \
        return KERNEL(float);                 \
    case dt_int8:                             \
    case dt_uint8:                            \
        return KERNEL(uint8_t);               \
    case dt_int16:                            \
    case dt_uint16:                           \
        return KERNEL(uint16_t);              \
    case dt_int32:                            \
    case dt_uint32:                           \
        return KERNEL(uint32_t);              \
    case dt_int64:                            \
    case dt_uint64:                           \
        return KERNEL(uint64_t);              \
    default:                                  \
        return err(std::errc::not_supported); \
    }

#define RESIZE_BILINEAR_IMPL(type) \
    resize_bilinear_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, out_h, out_w, align_corners, half_pixel_centers, context);

#define RESIZE_NEAREST_NEIGHBOR_IMPL(type) \
    resize_nearest_neighbor_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, out_h, out_w, align_corners, half_pixel_centers, context);

result<void> optimized::resize_bilinear(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
                                        const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
                                        kernel_context &context) noexcept
{
    FP_OR_Q_IMPL(type, RESIZE_BILINEAR_IMPL);
}

result<void> optimized::resize_nearest_neighbor(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
                                                const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
                                                kernel_context &context) noexcept
{
    FP_OR_Q_IMPL(type, RESIZE_NEAREST_NEIGHBOR_IMPL);
}