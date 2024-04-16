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
#include "arch_types.h"
#include "arm_math.h"

namespace nncase::ntt::ops {

// unary op
template <> struct exp<ntt::vector<float, 4>> {
    ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return exp_ps(v);
    }
};

template <> struct cos<ntt::vector<float, 4>> {
    ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return cos_ps(v);
    }
};

template <> struct sin<ntt::vector<float, 4>> {
    ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return sin_ps(v);
    }
};

template <> struct log<ntt::vector<float, 4>> {
    ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return log_ps(v);
    }
};

template <> struct sqrt<ntt::vector<float, 4>> {
    ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return vsqrtq_f32(v);
    }
};

// binary
template <> struct add<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vaddq_f32(lhs, rhs);
    }
};

template <> struct sub<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vsubq_f32(lhs, rhs);
    }
};

template <> struct mul<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vmulq_f32(lhs, rhs);
    }
};

template <> struct div<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vdivq_f32(lhs, rhs);
    }
};

template <> struct max<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vmaxq_f32(lhs, rhs);
    }
};

template <> struct reduce<ops::add, float, ntt::vector<float, 4>> {
    float operator()(const ntt::vector<float, 4> &tensor) {
        return vaddvq_f32(tensor);
    }
};

template <> struct reduce<ops::max, float, ntt::vector<float, 4>> {
    float operator()(const ntt::vector<float, 4> &tensor) {
        return vmaxvq_f32(tensor);
    }
};
} // namespace nncase::ntt::ops
