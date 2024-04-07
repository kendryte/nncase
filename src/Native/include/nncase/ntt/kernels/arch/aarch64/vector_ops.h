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
#include <arm_neon.h>

namespace nncase::ntt::vector_ops {
template <> struct reduce_sum<ntt::vector<float, 4>> {
    float operator()(ntt::vector<float, 4> v) const noexcept {
        float32x2_t vec1 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        return vaddv_f32(vec1);
    }
};

template <> struct reduce_sum<ntt::vector<float, 8>> {
    float operator()(ntt::vector<float, 8> v) const noexcept {
        float32x4x2_t val = v;
        float result = 0;
        auto vec = val.val[0];
        float32x2_t vec1 = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
        float32x2_t vec2 = vadd_f32(vec1, vrev64_f32(vec1));
        result += vget_lane_f32(vec2, 0);

        vec = val.val[1];
        vec1 = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
        vec2 = vadd_f32(vec1, vrev64_f32(vec1));
        result += vget_lane_f32(vec2, 0);

        return result;
    }
};

template <> struct reduce_max<ntt::vector<float, 4>> {
    float operator()(ntt::vector<float, 4> v) const noexcept {
        return vmaxvq_f32(v);
    }
};

template <> struct reduce_max<ntt::vector<float, 8>> {
    float operator()(ntt::vector<float, 8> v) const noexcept {
        float32x4x2_t val = v;
        return std::max(vmaxvq_f32(val.val[0]), vmaxvq_f32(val.val[1]));
    }
};

} // namespace nncase::ntt::vector_ops
