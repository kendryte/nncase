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
#include <cmath>
#include <cstdint>
#include <float.h>
#include <functional>
#include <limits>
#include <nncase/compiler_defs.h>

namespace nncase {
struct fp16_from_raw_t {
    explicit fp16_from_raw_t() = default;
};

NNCASE_INLINE_VAR constexpr fp16_from_raw_t fp16_from_raw{};

struct half {
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

    static constexpr uint16_t ZERO_VALUE = 0;

    // this is quiet NaN, sNaN only used for send signal
    static constexpr uint16_t NAN_VALUE = 0x7e00;

  public:
    half() noexcept = default;

    explicit half(float v) noexcept : value_(round_to_half(v).value_) {}

    template <class T,
              class = std::enable_if_t<std::is_integral<T>::value ||
                                       std::is_floating_point<T>::value>>
    explicit half(const T &val) noexcept : half(static_cast<float>(val)) {}

    half(int &&val) noexcept : half(static_cast<float>(val)) {}

    constexpr half(fp16_from_raw_t, uint16_t value) noexcept : value_(value) {}

    operator float() const noexcept {
        const fp32 magic = {113 << 23};
        const unsigned int shifted_exp = 0x7c00
                                         << 13; // exponent mask after shift
        fp32 o;

        o.u32 = (value_ & 0x7fff) << 13;        // exponent/mantissa bits
        unsigned int exp = shifted_exp & o.u32; // just the exponent
        o.u32 += (127 - 15) << 23;              // exponent adjust

        // handle exponent special cases
        if (exp == shifted_exp) {      // Inf/NaN?
            o.u32 += (128 - 16) << 23; // extra exp adjust
        } else if (exp == 0) {         // Zero/Denormal?
            o.u32 += 1 << 23;          // extra exp adjust
            o.f32 -= magic.f32;        // renormalize
        }

        o.u32 |= (value_ & 0x8000) << 16; // sign bit
        return o.f32;
    }

    const uint16_t &raw() const noexcept { return value_; }
    uint16_t &raw() noexcept { return value_; }

    static constexpr half from_raw(uint16_t v) noexcept {
        return half(nncase::fp16_from_raw, v);
    }

    static half round_to_half(float v) {
        fp32 f;
        f.f32 = v;
        const fp32 f32infy = {255 << 23};
        const fp32 f16max = {(127 + 16) << 23};
        const fp32 denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
        unsigned int sign_mask = 0x80000000u;

        unsigned int sign = f.u32 & sign_mask;
        f.u32 ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code
        // (since there's no unsigned PCMPGTD).
        half o;
        if (f.u32 >= f16max.u32) // result is Inf or NaN (all exponent bits set)
        {
            o.value_ = (f.u32 > f32infy.u32) ? 0x7e00
                                             : 0x7c00; // NaN->qNaN and Inf->Inf
        } else {
            if (f.u32 < (113 << 23)) { // resulting FP16 is subnormal or zero
                // use a magic value to align our 10 mantissa bits at the bottom
                // of the float. as long as FP addition is round-to-nearest-even
                // this just works.
                f.f32 += denorm_magic.f32;

                // and one integer subtract of the bias later, we have our final
                // float!
                o.value_ = static_cast<uint16_t>(f.u32 - denorm_magic.u32);
            } else {
                unsigned int mant_odd =
                    (f.u32 >> 13) & 1; // resulting mantissa is odd

                // update exponent, rounding bias part 1
                // Equivalent to `f.u32 += ((unsigned int)(15 - 127) << 23) +
                // 0xfff`, but without arithmetic overflow.
                f.u32 += 0xc8000fffU;
                // rounding bias part 2
                f.u32 += mant_odd;
                // take the bits!
                o.value_ = static_cast<uint16_t>(f.u32 >> 13);
            }
        }
        o.value_ |= static_cast<uint16_t>(sign >> 16);
        return o;
    }

    static constexpr half epsilon() noexcept { return from_raw(0x0800); }

    static constexpr half highest() noexcept { return from_raw(0x7bff); }

    static constexpr half min() noexcept { return from_raw(0x0400); }

    static constexpr half lowest() noexcept { return from_raw(0xfbff); }

    static constexpr half quiet_NaN() noexcept { return from_raw(0x7e00); }

    static constexpr half signaling_NaN() noexcept { return from_raw(0x7d00); }

    static constexpr half infinity() noexcept { return from_raw(0x7c00); }

    constexpr bool zero() const noexcept {
        return (value_ & 0x7FFF) == ZERO_VALUE;
    }

    void operator=(const float &v) noexcept {
        value_ = (round_to_half(v).value_);
    }

  private:
    uint16_t value_;
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
    inline half &operator x(half &a, half b) noexcept {                        \
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

    NNCASE_UNUSED static constexpr nncase::half(min)() noexcept {
        return nncase::half::min();
    }

    NNCASE_UNUSED static constexpr nncase::half(max)() noexcept {
        return nncase::half::highest();
    }

    NNCASE_UNUSED static constexpr nncase::half lowest() noexcept {
        return nncase::half::lowest();
    }

    NNCASE_UNUSED static constexpr nncase::half epsilon() noexcept {
        return nncase::half::epsilon();
    }

    NNCASE_UNUSED static nncase::half round_error() noexcept {
        return nncase::half((double)0.5);
    }

    NNCASE_UNUSED static constexpr nncase::half denorm_min() noexcept {
        return nncase::half::min();
    }

    NNCASE_UNUSED static constexpr nncase::half infinity() noexcept {
        return nncase::half::infinity();
    }

    NNCASE_UNUSED static constexpr nncase::half quiet_NaN() noexcept {
        return nncase::half::quiet_NaN();
    }

    NNCASE_UNUSED static constexpr nncase::half signaling_NaN() noexcept {
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
} // namespace std