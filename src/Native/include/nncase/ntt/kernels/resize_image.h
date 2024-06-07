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
#pragma once
#include "../apply.h"
#include "../shape_infer/reduce_axis.h"
#include "../tensor_ops.h"
#include "../utility.h"
#include "binary.h"
#include "unary.h"
#include <algorithm>

namespace nncase::ntt {

enum class image_resize_mode_t : uint8_t {
    bilinear = 0,
    nearest_neighbor = 1,
};

enum class image_resize_nearest_mode_t : int32_t {
    round_prefer_floor = 0,
    round_prefer_ceil = 1,
    floor = 2,
    ceil = 3,
};

enum class image_resize_transformation_mode_t : int32_t {
    half_pixel = 0,
    pytorch_half_pixel = 1,
    align_corners = 2,
    asymmetric = 3,
    tfcrop_and_resize = 4,
};

namespace resize_detail {
using get_coordinate_func_t = float (*)(float, float, float, float, float,
                                        float);
using get_nearest_pixel_func_t = int64_t (*)(float);

get_coordinate_func_t get_coordinate_from_resized(
    image_resize_transformation_mode_t coordinate_transform_mode);

get_nearest_pixel_func_t
get_nearest_pixel_from_origin(image_resize_nearest_mode_t nearest_mode);

inline get_coordinate_func_t get_coordinate_from_resized(
    image_resize_transformation_mode_t coordinate_transform_mode) {
    switch (coordinate_transform_mode) {
    case image_resize_transformation_mode_t::asymmetric:
        return [](float x_resized, float x_scale, float, float, float, float) {
            return x_resized * x_scale;
        };
    case image_resize_transformation_mode_t::pytorch_half_pixel:
        return [](float x_resized, float x_scale, float length_resized, float,
                  float, float) {
            return length_resized > 1 ? (x_resized + 0.5f) * x_scale - 0.5f
                                      : 0.0f;
        };
    case image_resize_transformation_mode_t::align_corners:
        return [](float x_resized, float, float length_resized,
                  float length_original, float, float) {
            return length_resized == 1 ? 0
                                       : x_resized * (length_original - 1) /
                                             (length_resized - 1);
        };
    case image_resize_transformation_mode_t::tfcrop_and_resize:
        return [](float x_resized, float, float length_resized,
                  float length_original, float roi_start, float roi_end) {
            auto orig =
                length_resized > 1
                    ? roi_start * (length_original - 1) +
                          (x_resized * (roi_end - roi_start) *
                           (length_original - 1)) /
                              (length_resized - 1)
                    : 0.5 * (roi_start + roi_end) * (length_original - 1);
            return static_cast<float>(orig);
        };
    default: // "image_resize_transformation_mode_t::half_pixel"
        return [](float x_resized, float x_scale, float, float, float, float) {
            return ((x_resized + 0.5f) * x_scale) - 0.5f;
        };
    }
}

inline get_nearest_pixel_func_t
get_nearest_pixel_from_origin(image_resize_nearest_mode_t nearest_mode) {
    switch (nearest_mode) {
    case image_resize_nearest_mode_t::round_prefer_ceil:
        return [](float x_original) -> int64_t {
            return static_cast<int64_t>(std::round(x_original));
        };
    case image_resize_nearest_mode_t::floor:
        return [](float x_original) -> int64_t {
            return static_cast<int64_t>(std::floor(x_original));
        };
    case image_resize_nearest_mode_t::ceil:
        return [](float x_original) -> int64_t {
            return static_cast<int64_t>(std::ceil(x_original));
        };
    default: // default is round_prefer_floor
        return [](float x_original) -> int64_t {
            // for half way cases prefer floor
            if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
                return static_cast<int64_t>(std::floor(x_original));
            }
            return static_cast<int64_t>(std::round(x_original));
        };
    }
}

template <typename TShape>
inline std::tuple<float, float> get_resize_scales(TShape in_shape,
                                                  int32_t out_h, int32_t out_w,
                                                  bool align_corners) noexcept {
    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;
    if (align_corners && out_h > 1)
        height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
    if (align_corners && out_w > 1)
        width_scale = (float)(in_shape[3] - 1) / (out_w - 1);
    return std::make_tuple(height_scale, width_scale);
}

inline void set_resize_bilinear(size_t value, float scale,
                                bool half_pixel_centers, size_t shape_size,
                                float &scaled_value, int32_t &v0, int32_t &v1) {
    if (half_pixel_centers) {
        scaled_value = (value + 0.5f) * scale - 0.5f;
    } else {
        scaled_value = value * scale;
    }
    float scaled_value_floor = std::floor(scaled_value);
    v0 = std::max(static_cast<int32_t>(scaled_value_floor), 0);
    v1 = std::min(static_cast<int32_t>(std::ceil(scaled_value)),
                  static_cast<int32_t>(shape_size - 1));
}

template <typename T> float get_rounding_offset() {
    return std::is_integral_v<T> ? .5f : .0f;
}

template <IsFixedTensor T> float get_rounding_offset() {
    return std::is_integral_v<typename T::element_type> ? .5f : .0f;
}

template <typename T, typename TInShape, typename TInStrides,
          typename TOutStrides>
void resize_bilinear(const T *input, T *output, const TInShape in_shape,
                     const TInStrides in_strides, const TOutStrides out_strides,
                     int32_t out_h, int32_t out_w, bool align_corners,
                     bool half_pixel_centers) noexcept {
    auto [height_scale, width_scale] =
        get_resize_scales(in_shape, out_h, out_w, align_corners);

    const T rounding_offset = (T)get_rounding_offset<T>();
    ranked_shape<4> in_index, out_index;

    auto get_input = [&](int32_t in_y, int32_t in_x) {
        in_index[2] = in_y;
        in_index[3] = in_x;
        return input[linear_offset(in_index, in_strides)];
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
                set_resize_bilinear(oy, height_scale, half_pixel_centers,
                                    in_shape[2], in_y, in_y0, in_y1);

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    out_index[3] = ox;
                    float in_x;
                    int32_t in_x0, in_x1;
                    set_resize_bilinear(ox, width_scale, half_pixel_centers,
                                        in_shape[3], in_x, in_x0, in_x1);

                    auto v0 = get_input(in_y0, in_x0);
                    auto v1 = get_input(in_y1, in_x0);
                    auto v2 = get_input(in_y0, in_x1);
                    auto v3 = get_input(in_y1, in_x1);

                    auto a0 = (T)((1 - (in_y - in_y0)) * (1 - (in_x - in_x0)));
                    auto a1 = (T)((in_y - in_y0) * (1 - (in_x - in_x0)));
                    auto a2 = (T)((1 - (in_y - in_y0)) * (in_x - in_x0));
                    auto a3 = (T)((in_y - in_y0) * (in_x - in_x0));
                    output[linear_offset(out_index, out_strides)] =
                        (T)(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 +
                            rounding_offset);
                }
            }
        }
    }
    return;
}

template <typename T, typename TInShape, typename TInStrides,
          typename TOutStrides>
void resize_neareast_neighbor(
    const T *input, T *output, const TInShape in_shape,
    const TInStrides in_strides, const TOutStrides out_strides,
    const int32_t out_h, const int32_t out_w, bool align_corners,
    [[maybe_unused]] bool half_pixel_centers,
    get_coordinate_func_t get_coordinate_func,
    get_nearest_pixel_func_t get_nearset_func) noexcept {
    auto [height_scale, width_scale] =
        get_resize_scales(in_shape, out_h, out_w, align_corners);

    ranked_shape<4> in_index, out_index;
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++) {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++) {
                auto iy = get_coordinate_func(oy, height_scale, out_h,
                                              in_shape[2], 0, 0);
                int64_t in_y = get_nearset_func(iy);
                if (in_y < 0)
                    in_y = 0;
                if (in_y >= in_shape[2])
                    in_y = in_shape[2] - 1;
                in_index[2] = in_y;
                out_index[2] = oy;

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    auto ix = get_coordinate_func(ox, width_scale, out_w,
                                                  in_shape[3], 0, 0);
                    int64_t in_x = get_nearset_func(ix);
                    if (in_x < 0)
                        in_x = 0;
                    if (in_x >= in_shape[3])
                        in_x = in_shape[3] - 1;
                    in_index[3] = in_x;
                    out_index[3] = ox;
                    output[linear_offset(out_index, out_strides)] =
                        input[linear_offset(in_index, in_strides)];
                }
            }
        }
    }
}

} // namespace resize_detail

template <typename TIn, typename TOut, IsFixedDims TPackedAxes,
          IsFixedDims TPadedNums, IsFixedDims TNewSize>
void resize(const TIn &input, TOut &&output,
            [[maybe_unused]] const TPackedAxes packedAxes,
            [[maybe_unused]] const TPadedNums padedNums,
            [[maybe_unused]] const TNewSize new_size,
            image_resize_mode_t resize_mode,
            image_resize_transformation_mode_t transformation_mode,
            image_resize_nearest_mode_t nearest_mode) {
    if (resize_mode == image_resize_mode_t::bilinear) {
        resize_detail::resize_bilinear(
            input.elements().data(), output.elements().data(), input.shape(),
            input.strides(), output.strides(), TNewSize::at(2), TNewSize::at(3),
            transformation_mode ==
                image_resize_transformation_mode_t::align_corners,
            transformation_mode ==
                image_resize_transformation_mode_t::half_pixel);
    } else {
        resize_detail::get_coordinate_func_t get_coordinate_func =
            resize_detail::get_coordinate_from_resized(transformation_mode);
        resize_detail::get_nearest_pixel_func_t get_nearset_func =
            resize_detail::get_nearest_pixel_from_origin(nearest_mode);
        resize_detail::resize_neareast_neighbor(
            input.elements().data(), output.elements().data(), input.shape(),
            input.strides(), output.strides(), TNewSize::at(2), TNewSize::at(3),
            transformation_mode ==
                image_resize_transformation_mode_t::align_corners,
            transformation_mode ==
                image_resize_transformation_mode_t::half_pixel,
            get_coordinate_func, get_nearset_func);
    }
}
} // namespace nncase::ntt