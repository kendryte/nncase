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
#include "arm_math.h"
#include <nncase/ntt/vector_type.h>

namespace std {
inline nncase::ntt::vector<float32_t, 4>
cos(nncase::ntt::vector<float32_t, 4> v) {
    return cos_ps(v);
}

inline float32x4x2_t exp(float32x4x2_t v) {
    return float32x4x2_t{exp_ps(v.val[0]), exp_ps(v.val[1])};
}
} // namespace std

namespace nncase::ntt {
namespace arch {
template <size_t Extent, class T, class Op>
constexpr void unary(Op &&op, const T *input_p, T *output_p) {
    for (size_t i = 0; i < Extent; i++) {
        output_p[i] = op(input_p[i]);
    }
}

template <class T, class Op>
constexpr void unary(Op &&op, const T *input_p, T *output_p, size_t extent) {
    for (size_t i = 0; i < extent; i++) {
        output_p[i] = op(input_p[i]);
    }
}
} // namespace arch
} // namespace nncase::ntt
