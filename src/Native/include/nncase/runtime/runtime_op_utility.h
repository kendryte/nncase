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
#include "datatypes.h"
#include "result.h"
#include <numeric>

BEGIN_NS_NNCASE_RUNTIME

inline size_t get_bytes(const datatype_t &type) { return type->size_bytes(); }

template <class TShape> inline size_t compute_size(const TShape &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
}

template <class TShape>
inline size_t get_bytes(const datatype_t &type, const TShape &shape) {
    return compute_size(shape) * get_bytes(type);
}

template <class TShape>
inline size_t compute_size(const TShape &shape, const TShape &strides) {
    size_t max_stride = 1, max_shape = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if ((shape[i] == 1 ? 0 : strides[i]) >= max_stride) {
            max_stride = strides[i];
            max_shape = shape[i];
        }
    }
    size_t size = max_stride * max_shape;
    return size;
}

template <class TShape>
inline size_t get_bytes(const datatype_t &type, const TShape &shape,
                        const TShape &strides) {
    return compute_size(shape, strides) * get_bytes(type);
}

namespace detail {
template <class shape_type, class strides_type, class bs_ptr>
inline std::size_t compute_strides(const shape_type &shape,
                                   strides_type &strides,
                                   [[maybe_unused]] bs_ptr bs) {
    using strides_value_type = typename std::decay_t<strides_type>::value_type;
    strides_value_type data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i) {
        strides[i - 1] = data_size;
        data_size =
            strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
    }
    return static_cast<std::size_t>(data_size);
}
} // namespace detail

template <class shape_type, class strides_type>
inline std::size_t compute_strides(const shape_type &shape,
                                   strides_type &strides) {
    return detail::compute_strides(shape, strides, nullptr);
}

inline strides_t get_default_strides(gsl::span<const size_t> shape) {
    strides_t strides(shape.size());
    compute_strides(shape, strides);
    return strides;
}

template <class TShape>
TShape convert_shape_type(const TShape &shape, datatype_t src,
                          datatype_t dest) {
    const auto src_size = get_bytes(src);
    const auto dest_size = get_bytes(dest);

    TShape new_shape = shape;
    if (!new_shape.empty()) {
        auto &v = new_shape.back();
        v = new_shape.back() * src_size / dest_size;
    }

    return new_shape;
}

template <class TShape>
result<TShape> convert_strides_type(const TShape &strides, datatype_t src,
                                    datatype_t dest) {
    const auto src_size = get_bytes(src);
    const auto dest_size = get_bytes(dest);

    if (src_size == dest_size)
        return ok(strides);

    TShape new_strides = strides;
    // 1. Except last dim
    for (size_t i = 0; i < new_strides.size() - 1; i++) {
        auto &v = new_strides[i];
        if (v == 0)
            v = 1;
        v = v * src_size / dest_size;
    }

    // 2. Last dim
    if (!new_strides.empty()) {
        // 2.1. If last dim is not 0 or 1, unsupported
        auto last_dim = new_strides.back();
        if (last_dim != 0 || last_dim != 1)
            return err(std::errc::not_supported);
    }

    return ok(new_strides);
}

template <int32_t Bits, class T> uint8_t count_leading_zeros(T value) {
    uint8_t num_zeroes = 0;
    for (int32_t i = Bits - 1; i >= 0; i--) {
        if ((value & (1ULL << i)) == 0)
            ++num_zeroes;
        else
            break;
    }

    return num_zeroes;
}

template <class T = uint64_t> inline T bit_mask(uint8_t shift) {
    return (T(1) << shift) - 1;
}

template <class T, bool Banker = false> T carry_shift(T value, int32_t shift) {
    if (shift > 0) {
        if (Banker) {
            T result;
            // Sign |  Int (T - shift - 1 bits) | Frac (shift bits)
            //  S      IIII                       FFF
            auto integral = value >> shift;
            auto fractional = value & bit_mask(shift);
            auto sign = value < 0 ? -1 : 1;
            auto half = size_t(1) << (shift - 1);

            // frac < 0.5
            if (fractional < half) {
                return integral;
            }
            // frac > 0.5
            else if (fractional > half) {
                return integral + sign;
            }
            // frac == 0.5
            else {
                // odd
                if (integral & 1)
                    return integral + sign;
                // even
                else
                    return integral;
            }

            return result;
        } else {
            value += T(1) << (shift - 1);
            value >>= shift;
        }
    } else if (shift < 0) {
        value = value << (-shift);
    }

    return value;
}

template <bool Banker = false>
inline int32_t mul_and_carry_shift(int32_t value, int32_t mul, int32_t shift) {
    return (int32_t)carry_shift<int64_t, Banker>((int64_t)value * mul, shift);
}

template <size_t Bits, bool Signed = true, class T = int64_t>
inline bool within_range(T value) noexcept {
    if constexpr (Signed) {
        auto min = -(1LL << (Bits - 1));
        auto max = (1LL << (Bits - 1)) - 1;
        return value >= min && value <= max;
    } else {
        auto min = 0ULL;
        auto max = (1ULL << Bits) - 1;
        return value >= min && value <= max;
    }
}

template <class T> inline T clamp(T value, T min, T max) {
    return std::min(max, std::max(value, min));
}

template <uint8_t Bits> inline int32_t clamp(int32_t value) {
    auto min = std::numeric_limits<int32_t>::lowest() >> (32 - Bits);
    auto max = std::numeric_limits<int32_t>::max() >> (32 - Bits);
    return clamp(value, min, max);
}

inline bool is_contiguous(gsl::span<const size_t> shape,
                          gsl::span<const size_t> strides) {
    size_t data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i) {
        if (strides[i - 1] != data_size) {
            return false;
        }
        data_size *= shape[i - 1];
    }
    return true;
}

inline int
get_last_not_contiguous_index(gsl::span<const size_t> strides,
                              gsl::span<const size_t> default_strides) {
    for (int i = strides.size() - 1; i >= 0; --i) {
        if (strides[i] != default_strides[i]) {
            return i + 1;
        }
    }
    return -1;
}

template <size_t A, size_t B>
constexpr auto is_not_equal =
    std::integral_constant<bool, std::not_equal_to<size_t>{}(A, B)>{};

struct DefaultCallable {};
END_NS_NNCASE_RUNTIME
