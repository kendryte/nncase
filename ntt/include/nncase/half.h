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
#include "ntt/compiler_defs.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <float.h>
#include <functional>
#include <limits>

namespace nncase {
struct fp16_from_raw_t {
    explicit fp16_from_raw_t() = default;
};

inline constexpr fp16_from_raw_t fp16_from_raw{};

struct half {
  private:
    static constexpr uint16_t ZERO_VALUE = 0;

    // this is quiet NaN, sNaN only used for send signal
    static constexpr uint16_t NAN_VALUE = 0x7e00;

  public:
    constexpr half() noexcept = default;
    constexpr half(_Float16 v) noexcept : value_(v) {}

    explicit half(float v) noexcept : value_(round_to_half(v).value_) {}

    template <class T,
              class = std::enable_if_t<std::is_integral<T>::value ||
                                       std::is_floating_point<T>::value>>
    explicit half(const T &val) noexcept : half(static_cast<float>(val)) {}

    constexpr half(fp16_from_raw_t, uint16_t value) noexcept
        : value_(std::bit_cast<_Float16>(value)) {}

    constexpr operator _Float16() const noexcept { return value_; }
    constexpr operator float() const noexcept { return (float)value_; }

    constexpr uint16_t raw() const noexcept {
        return std::bit_cast<uint16_t>(value_);
    }

    static constexpr half from_raw(uint16_t v) noexcept {
        return half(nncase::fp16_from_raw, v);
    }

    static half round_to_half(float v) { return (_Float16)v; }

    static constexpr half epsilon() noexcept { return from_raw(0x0800); }

    static constexpr half highest() noexcept { return from_raw(0x7bff); }

    static constexpr half min() noexcept { return from_raw(0x0400); }

    static constexpr half lowest() noexcept { return from_raw(0xfbff); }

    static constexpr half quiet_NaN() noexcept { return from_raw(0x7e00); }

    static constexpr half signaling_NaN() noexcept { return from_raw(0x7d00); }

    static constexpr half infinity() noexcept { return from_raw(0x7c00); }

    constexpr bool zero() const noexcept {
        return (raw() & 0x7FFF) == ZERO_VALUE;
    }

    void operator=(const float &v) noexcept {
        value_ = (round_to_half(v).value_);
    }

  private:
    _Float16 value_;
};

#define DEFINE_FP16_BINARY_FP16RET(x)                                          \
    inline half operator x(half a, half b) noexcept {                          \
        return half::round_to_half(float(a) x float(b));                       \
    }

#define DEFINE_FP16_BINARY_BOOLRET(x)                                          \
    inline bool operator x(half a, half b) noexcept {                          \
        return float(a) x float(b);                                            \
    }

DEFINE_FP16_BINARY_FP16RET(+)
DEFINE_FP16_BINARY_FP16RET(-)
DEFINE_FP16_BINARY_FP16RET(*)
DEFINE_FP16_BINARY_FP16RET(/)
DEFINE_FP16_BINARY_BOOLRET(<)
DEFINE_FP16_BINARY_BOOLRET(<=)
DEFINE_FP16_BINARY_BOOLRET(>=)
DEFINE_FP16_BINARY_BOOLRET(>)

#define DEFINE_FP16_BINARY_SELF_MOD(x, op)                                     \
    inline half &operator x(half & a, half b) noexcept {                       \
        a = a op b;                                                            \
        return a;                                                              \
    }

DEFINE_FP16_BINARY_SELF_MOD(+=, +)
DEFINE_FP16_BINARY_SELF_MOD(-=, -)
DEFINE_FP16_BINARY_SELF_MOD(*=, *)
DEFINE_FP16_BINARY_SELF_MOD(/=, /)

inline half operator-(half a) noexcept {
    return half::round_to_half(-float(a));
}

inline bool operator==(const half &lhs, const half &rhs) noexcept {
    return lhs.raw() == rhs.raw();
}

inline bool operator!=(const half &lhs, const half &rhs) noexcept {
    return lhs.raw() != rhs.raw();
}
} // namespace nncase

namespace std {
template <> struct hash<nncase::half> {
    size_t operator()(const nncase::half &v) const {
        return hash<float>()(static_cast<float>(v));
    }
};

template <> struct numeric_limits<nncase::half> {
    static constexpr float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr float_round_style round_style = std::round_to_nearest;
    static constexpr int radix = FLT_RADIX;

    static constexpr nncase::half(min)() noexcept {
        return nncase::half::min();
    }

    static constexpr nncase::half(max)() noexcept {
        return nncase::half::highest();
    }

    static constexpr nncase::half lowest() noexcept {
        return nncase::half::lowest();
    }

    static constexpr nncase::half epsilon() noexcept {
        return nncase::half::epsilon();
    }

    static nncase::half round_error() noexcept {
        return nncase::half((double)0.5);
    }

    static constexpr nncase::half denorm_min() noexcept {
        return nncase::half::min();
    }

    static constexpr nncase::half infinity() noexcept {
        return nncase::half::infinity();
    }

    static constexpr nncase::half quiet_NaN() noexcept {
        return nncase::half::quiet_NaN();
    }

    static constexpr nncase::half signaling_NaN() noexcept {
        return nncase::half::signaling_NaN();
    }

    static constexpr int digits = 11;
    static const int min_exponent = -13;
    static const int min_exponent10 = -4;
    static const int max_exponent = 16;
    static const int max_exponent10 = 4;
};

using nncase::half;
inline bool isinf(const half &a) { return std::isinf(float(a)); }
inline bool isnan(const half &a) { return std::isnan(float(a)); }
inline bool isfinite(const half &a) { return std::isfinite(float(a)); }
inline half abs(const half &a) { return half::round_to_half(fabsf(float(a))); }
inline half exp(const half &a) { return half::round_to_half(expf(float(a))); }
inline half log(const half &a) { return half::round_to_half(logf(float(a))); }
inline half log10(const half &a) {
    return half::round_to_half(log10f(float(a)));
}
inline half sqrt(const half &a) { return half::round_to_half(sqrtf(float(a))); }
inline half pow(const half &a, const half &b) {
    return half::round_to_half(powf(float(a), float(b)));
}

inline half sin(const half &a) { return half::round_to_half(sinf(float(a))); }
inline half cos(const half &a) { return half::round_to_half(cosf(float(a))); }
inline half tan(const half &a) { return half::round_to_half(tanf(float(a))); }
inline half tanh(const half &a) { return half::round_to_half(tanhf(float(a))); }
inline half floor(const half &a) {
    return half::round_to_half(floorf(float(a)));
}
inline half ceil(const half &a) { return half::round_to_half(ceilf(float(a))); }
inline half round(const half &a) {
    return half::round_to_half(roundf(float(a)));
}
inline half nearbyint(const half &a) {
    return half::round_to_half(nearbyintf(float(a)));
}
inline long lrint(const half &a) { return lrintf(float(a)); }

template <> struct is_floating_point<half> : public std::true_type {};
} // namespace std