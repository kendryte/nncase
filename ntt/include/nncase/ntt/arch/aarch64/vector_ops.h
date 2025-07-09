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
#include "../../vector_ops.h"
#include "arch_types.h"
#include "arm_math.h"

namespace nncase::ntt::vector_ops {
template <> struct vload_scalar<ntt::vector<float, 4>> {
    ntt::vector<float, 4> operator()(const float &v) const noexcept {
        return vdupq_n_f32(v);
    }
};

template <> struct vload_scalar<ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4> operator()(const float &v) const noexcept {
        ntt::vector<float, 4, 4> ret;
        ret(0) = vdupq_n_f32(v);
        ret(1) = vdupq_n_f32(v);
        ret(2) = vdupq_n_f32(v);
        ret(3) = vdupq_n_f32(v);
        return ret;
    }
};

template <> struct vload_scalar<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(const float &v) const noexcept {
        return float32x4x2_t{vdupq_n_f32(v), vdupq_n_f32(v)};
    }
};

template <bool Acc>
struct vmma<Acc, true, ntt::vector<float, 4, 4>, ntt::vector<float, 4, 4>,
            ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4>
    operator()(const ntt::vector<float, 4, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 4, 4> &out) const noexcept {
        ntt::vector<float, 4, 4> ret;

        // c,n,m,lane => c = c + (m[lane] * n)
        if (Acc) {
            ret(0_dim) = vfmaq_laneq_f32(out(0_dim), rhs(0_dim), lhs(0_dim), 0); // k = 0
            ret(1_dim) = vfmaq_laneq_f32(out(1_dim), rhs(0_dim), lhs(0_dim), 1);
            ret(2_dim) = vfmaq_laneq_f32(out(2_dim), rhs(0_dim), lhs(0_dim), 2);
            ret(3_dim) = vfmaq_laneq_f32(out(3_dim), rhs(0_dim), lhs(0_dim), 3);
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
} // namespace nncase::ntt::vector_ops
