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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <nncase/kernels/stackvm/resize_image.h>
#include <nncase/runtime/datatypes.h>
#include <numeric>

#ifdef __GNUC__
#define CXX_RESTRICT __restrict__
#elif _MSC_VER
#define CXX_RESTRICT __restrict
#else
#define CXX_RESTRICT
#endif

#define TYPE_IMPL_SELECT(type, IMPL)                                           \
    switch (runtime::get_bytes(type)) {                                        \
        IMPL(1, uint8_t);                                                      \
        IMPL(2, uint16_t);                                                     \
        IMPL(4, uint32_t);                                                     \
        IMPL(8, uint64_t);                                                     \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

BEGIN_NS_NNCASE_KERNELS

template <class offset_type, class S, class It>
inline offset_type element_offset(const S &strides, It first,
                                  It last) noexcept {
    using difference_type = typename std::iterator_traits<It>::difference_type;
    auto size = static_cast<difference_type>((std::min)(
        static_cast<size_t>(std::distance(first, last)), strides.size()));
    return std::inner_product(last - size, last, strides.cend() - size,
                              offset_type(0));
}

inline size_t offset(gsl::span<const size_t> strides,
                     gsl::span<const size_t> index) {
    // scalar
    if (strides.size() == 0 || index.size() == 0) {
        return 0;
    }
    assert(strides.size() == index.size());
    return kernels::element_offset<size_t>(strides, index.begin(), index.end());
}

template <class TShape>
TShape reshape_linear_index(const TShape &new_shape, size_t index) {
    TShape new_index(new_shape.size());
    size_t i = new_shape.size() - 1;
    for (auto it = new_shape.rbegin(); it != new_shape.rend(); ++it) {
        new_index[i--] = index % *it;
        index /= *it;
    }

    return new_index;
}

template <class TShape>
size_t linear_index(const TShape &shape, const TShape &index) {
    assert(index.size() == shape.size());
    size_t new_index = index[0];
    for (size_t i = 1; i < shape.size(); i++)
        new_index = new_index * shape[i] + index[i];
    return new_index;
}

namespace detail {
inline size_t get_windowed_output_size(size_t size, int32_t filter,
                                       int32_t stride, int32_t dilation,
                                       const padding &padding) {
    auto effective_filter_size = (filter - 1) * dilation + 1;
    return (size_t)((int32_t)size + padding.before + padding.after -
                    effective_filter_size + stride) /
           stride;
}

inline dims_t get_binary_output_shape(gsl::span<const size_t> input_a_shape,
                                      gsl::span<const size_t> input_b_shape) {
    dims_t out_shape;

    const auto dest_dims =
        (int32_t)std::max(input_a_shape.size(), input_b_shape.size());
    const auto in_a_ext = dest_dims - (int32_t)input_a_shape.size();
    const auto in_b_ext = dest_dims - (int32_t)input_b_shape.size();

    for (int32_t i = 0; i < dest_dims; i++) {
        const auto in_a_dim = i - (int32_t)in_a_ext;
        const auto in_b_dim = i - (int32_t)in_b_ext;

        const auto in_a = in_a_dim < 0 ? 1 : input_a_shape[in_a_dim];
        const auto in_b = in_b_dim < 0 ? 1 : input_b_shape[in_b_dim];
        if (in_a == in_b)
            out_shape.push_back(in_a);
        else if (in_a == 1)
            out_shape.push_back(in_b);
        else if (in_b == 1)
            out_shape.push_back(in_a);
        else
            assert(!"inputs are not compatible to broadcast");
    }

    return out_shape;
}

template <class T> inline T clamp(T value, T min, T max) {
    return std::max(std::min(value, max), min);
}

template <class T>
inline T apply_activation(T value, value_range<T> activation) {
    return clamp(value, activation.min, activation.max);
}

inline dims_t get_reduced_offset(gsl::span<const size_t> in_offset,
                                 gsl::span<const size_t> reduced_shape) {
    dims_t off(reduced_shape.size());
    const auto dims_ext = in_offset.size() - reduced_shape.size();
    for (size_t i = 0; i < reduced_shape.size(); i++) {
        if (in_offset[i + dims_ext] >= reduced_shape[i])
            off[i] = 0;
        else
            off[i] = in_offset[i + dims_ext];
    }

    return off;
}

inline dims_t get_reduced_shape(gsl::span<const size_t> in_shape,
                                gsl::span<const size_t> axis, bool keep_dims) {
    dims_t shape;
    shape.reserve(in_shape.size() - (keep_dims ? 0 : axis.size()));
    for (size_t i = 0; i < in_shape.size(); i++) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            shape.push_back(in_shape[i]);
        } else {
            if (keep_dims)
                shape.push_back(1);
        }
    }
    return shape;
}

template <class TShape>
size_t get_reduce_block_size(const TShape &in_shape, const TShape &axis) {
    size_t size = 1;
    for (size_t i = 0; i < in_shape.size(); i++) {
        if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
            size *= in_shape[i];
        }
    }

    return size;
}

inline dims_t get_reduced_offset(gsl::span<const size_t> in_offset,
                                 gsl::span<const size_t> axis, bool keep_dims) {
    if (in_offset.size() == 0) {
        return in_offset;
    }
    dims_t off;
    off.reserve(in_offset.size() - (keep_dims ? 0 : axis.size()));
    for (size_t i = 0; i < in_offset.size(); i++) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            off.push_back(in_offset[i]);
        } else {
            if (keep_dims)
                off.push_back(0);
        }
    }

    return off;
}

template <class T, class TRange> struct default_ptr_getter {
    T *operator()(const TRange &range) const noexcept { return range; }
};

template <int32_t Bits> int32_t to_signed(uint32_t value) {
    auto mask = uint32_t(1) << (Bits - 1);
    if (Bits != 32 && (value & mask) != 0) {
        auto sign = 0xFFFFFFFF << Bits;
        return (int)(value | sign);
    }

    return (int32_t)value;
}

template <int32_t Bits> int64_t to_signed(uint64_t value) {
    auto mask = uint64_t(1) << (Bits - 1);
    if ((value & mask) != 0) {
        auto sign = 0xFFFFFFFFFFFFFFFF << Bits;
        return (int64_t)(value | sign);
    }

    return (int64_t)value;
}

template <class T>
constexpr T quantize(float value, const quant_param_t &param) noexcept {
    return (T)clamp((int32_t)lrintf(value / param.scale + param.zero_point),
                    (int32_t)std::numeric_limits<T>::lowest(),
                    (int32_t)std::numeric_limits<T>::max());
}

inline std::pair<float, float>
get_resize_scales(gsl::span<const size_t> in_shape, int32_t out_h,
                  int32_t out_w, bool align_corners) {
    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;
    if (align_corners && out_h > 1)
        height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
    if (align_corners && out_w > 1)
        width_scale = (float)(in_shape[3] - 1) / (out_w - 1);
    return {height_scale, width_scale};
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

template <class T>
inline size_t get_nearest_neighbor(T input_value, size_t shape_size,
                                   float scale, bool align_corners,
                                   bool half_pixel_centers) {
    const auto offset = half_pixel_centers ? 0.5f : 0.0f;
    const auto after_scale = (static_cast<float>(input_value) + offset) * scale;
    const auto align_corners_val =
        align_corners ? roundf(after_scale) : std::floor(after_scale);
    int32_t output_value = std::min(static_cast<int32_t>(align_corners_val),
                                    static_cast<int32_t>(shape_size - 1));
    if (half_pixel_centers) {
        output_value = std::max(0, output_value);
    }
    return output_value;
}

} // namespace detail
END_NS_NNCASE_KERNELS
