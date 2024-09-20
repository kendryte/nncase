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

// outer product
template <> struct outer_product<ntt::vector<float, 4>, ntt::vector<float, 4>> {
    auto operator()(const ntt::vector<float, 4> &v1,
                    const ntt::vector<float, 4> &v2) const noexcept {
        ntt::vector<float32_t, 4, 4> result;
        float32x4_t m0 = vdupq_n_f32(v1(0));
        float32x4_t m1 = vdupq_n_f32(v1(1));
        float32x4_t m2 = vdupq_n_f32(v1(2));
        float32x4_t m3 = vdupq_n_f32(v1(3));

        result(0) = vmulq_f32(m0, v2);
        result(1) = vmulq_f32(m1, v2);
        result(2) = vmulq_f32(m2, v2);
        result(3) = vmulq_f32(m3, v2);
        return result;
    }
};

template <bool Acc>
struct mma<Acc, true, ntt::vector<float, 4, 4>, ntt::vector<float, 4, 4>,
           ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4>
    operator()(const ntt::vector<float, 4, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 4, 4> &out) const noexcept {
        ntt::vector<float, 4, 4> ret;
        
        // c,n,m,lane => c = c + (m[lane] * n)
        if (Acc){
          ret(0) = vfmaq_laneq_f32(out(0), rhs(0), lhs(0), 0); // k = 0
          ret(1) = vfmaq_laneq_f32(out(1), rhs(0), lhs(0), 1);
          ret(2) = vfmaq_laneq_f32(out(2), rhs(0), lhs(0), 2);
          ret(3) = vfmaq_laneq_f32(out(3), rhs(0), lhs(0), 3);
        } else {
          vector<float, 4> zero = vdupq_n_f32(0.f);
          ret(0) = vfmaq_laneq_f32(zero, rhs(0), lhs(0), 0); // k = 0
          ret(1) = vfmaq_laneq_f32(zero, rhs(0), lhs(0), 1);
          ret(2) = vfmaq_laneq_f32(zero, rhs(0), lhs(0), 2);
          ret(3) = vfmaq_laneq_f32(zero, rhs(0), lhs(0), 3);
        }

        ret(0) = vfmaq_laneq_f32(ret(0), rhs(1), lhs(1), 0); // k = 1
        ret(1) = vfmaq_laneq_f32(ret(1), rhs(1), lhs(1), 1);
        ret(2) = vfmaq_laneq_f32(ret(2), rhs(1), lhs(1), 2);
        ret(3) = vfmaq_laneq_f32(ret(3), rhs(1), lhs(1), 3);
        
        ret(0) = vfmaq_laneq_f32(ret(0), rhs(2), lhs(2), 0); // k = 2
        ret(1) = vfmaq_laneq_f32(ret(1), rhs(2), lhs(2), 1);
        ret(2) = vfmaq_laneq_f32(ret(2), rhs(2), lhs(2), 2);
        ret(3) = vfmaq_laneq_f32(ret(3), rhs(2), lhs(2), 3);
        
        ret(0) = vfmaq_laneq_f32(ret(0), rhs(3), lhs(3), 0); // k = 3
        ret(1) = vfmaq_laneq_f32(ret(1), rhs(3), lhs(3), 1);
        ret(2) = vfmaq_laneq_f32(ret(2), rhs(3), lhs(3), 2);
        ret(3) = vfmaq_laneq_f32(ret(3), rhs(3), lhs(3), 3);

        return ret;
    }
};

} // namespace nncase::ntt::ops
