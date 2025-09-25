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
#include "../../primitive_ops.h"
#include "../../vector.h"
#include "../../vector_ops.h"
#include "arch_types.h"
#include "arm_math.h"
#include "primitive_ops.float32.h"
#include <arm_neon.h>

namespace nncase::ntt::ops {

// cast_elem
template <> struct cast_elem<ntt::vector<half, 8>, float> {
    ntt::vector<float, 2, 4>
    operator()(const ntt::vector<half, 8> &v) const noexcept {
        float32x4_t fp32_v0 = vcvt_f32_f16(vget_low_f16(v));
        float32x4_t fp32_v1 = vcvt_f32_f16(vget_high_f16(v));
        return std::array<ntt::vector<float, 4>, 2>{fp32_v0, fp32_v1};
    }
};

// cast_elem
template <> struct cast_elem<ntt::vector<float, 2, 4>, half> {
    ntt::vector<half, 8>
    operator()(const ntt::vector<float, 2, 4> &v) const noexcept {
        float16x4_t low_half = vcvt_f16_f32(v(0_dim));
        float16x4_t high_half = vcvt_f16_f32(v(1_dim));
        return vcombine_f16(low_half, high_half);
    }
};

// binary

// add
template <> struct add<float, half> {
    constexpr float operator()(const float &s1, const half &s2) const noexcept {
        return s1 + (float)s2;
    }
};

template <> struct add<ntt::vector<half, 8>, ntt::vector<half, 8>> {
    ntt::vector<half, 8>
    operator()(const ntt::vector<half, 8> &v1,
               const ntt::vector<half, 8> &v2) const noexcept {
        return vaddq_f16(v1, v2);
    }
};

template <> struct add<ntt::vector<float, 2, 4>, ntt::vector<half, 8>> {
    ntt::vector<float, 2, 4>
    operator()(const ntt::vector<float, 2, 4> &v1,
               const ntt::vector<half, 8> &v2) const noexcept {
        const auto fp32_v2 = ntt::cast_elem<float>(v2);
        return ntt::add(v1, fp32_v2);
    }
};
} // namespace nncase::ntt::ops
