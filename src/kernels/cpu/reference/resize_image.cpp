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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
std::pair<float, float> get_scales(const runtime_shape_t &in_shape, int32_t out_h, int32_t out_w, bool align_corners)
{
    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;
    if (align_corners && out_h > 1)
        height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
    if (align_corners && out_w > 1)
        width_scale = (float)(in_shape[3] - 1) / (out_w - 1);
    return { height_scale, width_scale };
}

void set_resize_bilinear(size_t value, float scale, bool half_pixel_centers, size_t shape_size, float &scaled_value, int32_t &v0, int32_t &v1)
{
    if(half_pixel_centers)
    {
        scaled_value = (value + 0.5f) * scale - 0.5f;
    }
    else
    {
        scaled_value = value * scale;
    }
    float scaled_value_floor = std::floor(scaled_value);
    v0 = std::max(static_cast<int32_t>(scaled_value_floor), 0);
    v1 = std::min(static_cast<int32_t>(std::ceil(scaled_value)), static_cast<int32_t>(shape_size - 1));
}

template <class T>
result<void> resize_bilinear_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, NNCASE_UNUSED bool half_pixel_centers, kernel_context &context) noexcept
{
    auto [height_scale, width_scale] = get_scales(in_shape, out_h, out_w, align_corners);
    const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;
    runtime_shape_t in_index(4), out_index(4);

    auto get_input = [&](int32_t in_y, int32_t in_x) {
        in_index[2] = in_y;
        in_index[3] = in_x;
        return input[offset(in_strides, in_index)];
    };

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++)
            {
                out_index[2] = oy;
                float in_y;
                int32_t in_y0, in_y1;
                set_resize_bilinear(oy, height_scale, half_pixel_centers, in_shape[2], in_y, in_y0, in_y1);

                for (size_t ox = 0; ox < (size_t)out_w; ox++)
                {
                    out_index[3] = ox;
                    float in_x;
                    int32_t in_x0, in_x1;
                    set_resize_bilinear(ox, width_scale, half_pixel_centers, in_shape[3], in_x, in_x0, in_x1);

                    auto v0 = get_input(in_y0, in_x0);
                    auto v1 = get_input(in_y1, in_x0);
                    auto v2 = get_input(in_y0, in_x1);
                    auto v3 = get_input(in_y1, in_x1);

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);
                    output[offset(out_strides, out_index)] = T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 + rounding_offset);
                }
            }
        }
    }
    return ok();
}

template<class T>
size_t get_nearest_neighbor(T input_value,  size_t shape_size, float scale, bool align_corners, bool half_pixel_centers)
{
    const auto offset = half_pixel_centers ? 0.5f : 0.0f;
    const auto after_scale = (static_cast<float>(input_value) + offset) * scale;
    const auto align_corners_val = align_corners ? std::round(after_scale) : std::floor(after_scale);
    int32_t output_value = std::min(static_cast<int32_t>(align_corners_val), static_cast<int32_t>(shape_size - 1));
    if (half_pixel_centers) {
        output_value = std::max(0, output_value);
    }
    return output_value;
}

template <class T>
result<void> resize_nearest_neighbor_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
                                          const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers, kernel_context &context) noexcept
{
    auto [height_scale, width_scale] = get_scales(in_shape, out_h, out_w, align_corners);
    runtime_shape_t in_index(4), out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++)
            {
                auto in_y = get_nearest_neighbor(oy, in_shape[2], height_scale, align_corners, half_pixel_centers);
                in_index[2] = in_y;
                out_index[2] = oy;

                for (size_t ox = 0; ox < (size_t)out_w; ox++)
                {
                    auto in_x = get_nearest_neighbor(ox, in_shape[3], width_scale, align_corners, half_pixel_centers);
                    in_index[3] = in_x;
                    out_index[3] = ox;
                    output[offset(out_strides, out_index)] = input[offset(in_strides, in_index)];
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

result<void> reference::resize_bilinear(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept
{
    FP_OR_Q_IMPL(type, RESIZE_BILINEAR_IMPL);
}

result<void> reference::resize_nearest_neighbor(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept
{
    FP_OR_Q_IMPL(type, RESIZE_NEAREST_NEIGHBOR_IMPL);
}