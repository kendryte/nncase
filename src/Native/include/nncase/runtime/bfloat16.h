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
#include "../compiler_defs.h"
#include <cmath>
#include <cstdint>
#include <float.h>
#include <functional>
#include <limits>

namespace nncase {
struct from_raw_t {
    explicit from_raw_t() = default;
};

NNCASE_INLINE_VAR constexpr from_raw_t from_raw{};

struct bfloat16 {
  private:
    union fp32 {
        uint32_t u32;
        float f32;

        uint16_t u16() const noexcept {
            constexpr size_t index = NNCASE_LITTLE_ENDIAN ? 1 : 0;
            return reinterpret_cast<const uint16_t *>(&u32)[index];
        }

        uint16_t &u16() noexcept {
            constexpr size_t index = NNCASE_LITTLE_ENDIAN ? 1 : 0;
            return reinterpret_cast<uint16_t *>(&u32)[index];
        }
    };

    // A value that represents "zero".
    static constexpr uint16_t ZERO_VALUE = 0;

    // A value that represents "not a number".
    static constexpr uint16_t NAN_VALUE = 0x7FC0;

  public:
    bfloat16() noexcept = default;

    explicit bfloat16(float v) noexcept : value_(round_to_bfloat16(v).value_) {}

    template <class T,
              class = std::enable_if_t<std::is_integral<T>::value ||
                                       std::is_floating_point<T>::value>>
    explicit bfloat16(const T &val) noexcept
        : bfloat16(static_cast<float>(val)) {}

    bfloat16(int &&val) noexcept : bfloat16(static_cast<float>(val)) {}

    constexpr bfloat16(from_raw_t, uint16_t value) noexcept : value_(value) {}

    operator float() const noexcept {
        fp32 result;
        result.u32 = 0;
        result.u16() = value_;
        return result.f32;
    }

    const uint16_t &raw() const noexcept { return value_; }
    uint16_t &raw() noexcept { return value_; }

    static constexpr bfloat16 from_raw(uint16_t v) noexcept {
        return bfloat16(nncase::from_raw, v);
    }

    static bfloat16 truncate_to_bfloat16(const float v) noexcept {
        bfloat16 output;

        if (!std::isnan(v)) {
            fp32 f;
            f.f32 = v;
            output.value_ = f.u16();
        } else {
            output.value_ = NAN_VALUE;
        }

        return output;
    }

    // Converts a float point to bfloat16, with round-nearest-to-even as
    // rounding method.
    static bfloat16 round_to_bfloat16(float v) {
        uint32_t input;
        fp32 f;
        f.f32 = v;
        input = f.u32;
        bfloat16 output;

        if (!std::isnan(v)) {
            // Least significant bit of resulting bfloat.
            uint32_t lsb = (input >> 16) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            input += rounding_bias;
            output.value_ = static_cast<uint16_t>(input >> 16);
        } else {
            // If the value is a NaN, squash it to a qNaN with msb of fraction
            // set, this makes sure after truncation we don't end up with an
            // inf.
            //
            // qNaN magic: All exponent bits set + most significant bit of
            // fraction set.
            output.value_ = NAN_VALUE;
        }

        return output;
    }

    static constexpr bfloat16 epsilon() noexcept {
        // 0x1.0p-7
        return from_raw(0x3c00);
    }

    static constexpr bfloat16 highest() noexcept {
        // 0x1.FEp127
        return from_raw(0x7F7F);
    }

    static constexpr bfloat16 min() noexcept {
        // 0x1p-126
        return from_raw(0x0080);
    }

    static constexpr bfloat16 lowest() noexcept {
        // -0x1.FEp127
        return from_raw(0xFF7F);
    }

    static constexpr bfloat16 nan() noexcept { return from_raw(0x7fc0); }

    static constexpr bfloat16 quiet_NaN() noexcept { return from_raw(0x7fc0); }

    static constexpr bfloat16 signaling_NaN() noexcept {
        return from_raw(0x7f81);
    }

    static constexpr bfloat16 infinity() noexcept { return from_raw(0x7f80); }

    constexpr bool zero() const noexcept {
        return (value_ & 0x7FFF) == ZERO_VALUE;
    }

    void operator=(const float &v) noexcept {
        value_ = (round_to_bfloat16(v).value_);
    }

  private:
    uint16_t value_;
};

#define DEFINE_BF16_BINARY_BF16RET(x)                                          \
    inline bfloat16 operator x(bfloat16 a, bfloat16 b) noexcept {              \
        return bfloat16::round_to_bfloat16(float(a) x float(b));               \
    }

#define DEFINE_BF16_BINARY_BOOLRET(x)                                          \
    inline bool operator x(bfloat16 a, bfloat16 b) noexcept {                  \
        return float(a) x float(b);                                            \
    }

DEFINE_BF16_BINARY_BF16RET(+)
DEFINE_BF16_BINARY_BF16RET(-)
DEFINE_BF16_BINARY_BF16RET(*)
DEFINE_BF16_BINARY_BF16RET(/)
DEFINE_BF16_BINARY_BOOLRET(<)
DEFINE_BF16_BINARY_BOOLRET(<=)
DEFINE_BF16_BINARY_BOOLRET(>=)
DEFINE_BF16_BINARY_BOOLRET(>)

#define DEFINE_BF16_BINARY_SELF_MOD(x, op)                                     \
    inline bfloat16 &operator x(bfloat16 &a, bfloat16 b) noexcept {            \
        a = a op b;                                                            \
        return a;                                                              \
    }

DEFINE_BF16_BINARY_SELF_MOD(+=, +)
DEFINE_BF16_BINARY_SELF_MOD(-=, -)
DEFINE_BF16_BINARY_SELF_MOD(*=, *)
DEFINE_BF16_BINARY_SELF_MOD(/=, /)

inline bfloat16 operator-(bfloat16 a) noexcept {
    return bfloat16::round_to_bfloat16(-float(a));
}

inline bool operator==(const bfloat16 &lhs, const bfloat16 &rhs) noexcept {
    return lhs.raw() == rhs.raw();
}

inline bool operator!=(const bfloat16 &lhs, const bfloat16 &rhs) noexcept {
    return lhs.raw() != rhs.raw();
}
} // namespace nncase

namespace std {
template <> struct hash<nncase::bfloat16> {
    size_t operator()(const nncase::bfloat16 &v) const {
        return hash<float>()(static_cast<float>(v));
    }
};

template <> struct numeric_limits<nncase::bfloat16> {
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr int radix = FLT_RADIX;

    NNCASE_UNUSED static constexpr nncase::bfloat16(min)() noexcept {
        return nncase::bfloat16::min();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16(max)() noexcept {
        return nncase::bfloat16::highest();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 lowest() noexcept {
        return nncase::bfloat16::lowest();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 epsilon() noexcept {
        return nncase::bfloat16::epsilon();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 round_error() noexcept {
        // 0.5
        return nncase::bfloat16::from_raw(0x3f00);
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 denorm_min() noexcept {
        return nncase::bfloat16::min();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 infinity() noexcept {
        return nncase::bfloat16::infinity();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 quiet_NaN() noexcept {
        return nncase::bfloat16::quiet_NaN();
    }

    NNCASE_UNUSED static constexpr nncase::bfloat16 signaling_NaN() noexcept {
        return nncase::bfloat16::signaling_NaN();
    }

    static constexpr int digits = 8;
    static constexpr int max_exponent = FLT_MAX_EXP;
    static constexpr int min_exponent = FLT_MIN_EXP;
};

using nncase::bfloat16;
inline bool isinf(const bfloat16 &a) { return std::isinf(float(a)); }
inline bool isnan(const bfloat16 &a) { return std::isnan(float(a)); }
inline bool isfinite(const bfloat16 &a) { return std::isfinite(float(a)); }
inline bfloat16 abs(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(fabsf(float(a)));
}
inline bfloat16 exp(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(expf(float(a)));
}
inline bfloat16 log(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(logf(float(a)));
}
inline bfloat16 log10(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(log10f(float(a)));
}
inline bfloat16 sqrt(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(sqrtf(float(a)));
}
inline bfloat16 pow(const bfloat16 &a, const bfloat16 &b) {
    return bfloat16::round_to_bfloat16(powf(float(a), float(b)));
}
inline bfloat16 sin(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(sinf(float(a)));
}
inline bfloat16 cos(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(cosf(float(a)));
}
inline bfloat16 tan(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(tanf(float(a)));
}
inline bfloat16 tanh(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(tanhf(float(a)));
}
inline bfloat16 floor(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(floorf(float(a)));
}
inline bfloat16 ceil(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(ceilf(float(a)));
}
inline bfloat16 round(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(roundf(float(a)));
}
inline bfloat16 nearbyint(const bfloat16 &a) {
    return bfloat16::round_to_bfloat16(nearbyintf(float(a)));
}
inline long lrint(const bfloat16 &a) { return lrintf(float(a)); }
} // namespace std
