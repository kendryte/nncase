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
#include "../../tensor_ops.h"
#include "arch_types.h"
#include "arm_math.h"

namespace nncase::ntt::tensor_ops {
template <> struct load_scalar<ntt::vector<float, 4>> {
    ntt::vector<float, 4> operator()(const float &v) const noexcept {
        return vdupq_n_f32(v);
    }
};

template <> struct reduce<ntt::vector<float, 4>, ops::add> {
    float operator()(const ntt::vector<float, 4> &tensor) {
        return vaddvq_f32(tensor);
    }
};

template <> struct reduce<ntt::vector<float, 8>, ops::add> {
    float operator()(const ntt::vector<float, 8> &tensor) {
        float32x4x2_t val = tensor;
        return vaddvq_f32(val.val[0]) + vaddvq_f32(val.val[1]);
    }
};

template <> struct reduce<ntt::vector<float, 4>, ops::max> {
    float operator()(const ntt::vector<float, 4> &tensor) {
        return vmaxvq_f32(tensor);
    }
};

template <> struct reduce<ntt::vector<float, 8>, ops::max> {
    float operator()(const ntt::vector<float, 8> &tensor) {
        float32x4x2_t val = tensor;
        return std::max(vmaxvq_f32(val.val[0]), vmaxvq_f32(val.val[1]));
    }
};
} // namespace nncase::ntt::tensor_ops
