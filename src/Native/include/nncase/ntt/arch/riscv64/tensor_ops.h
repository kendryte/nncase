/* Copyright 2019-2024 Canaan Inc.
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

namespace nncase::ntt::tensor_ops {
#define RVV_LOAD_SCALAR_FLOAT32(vl, lmul)                                      \
    template <> struct tload_scalar<ntt::vector<float, vl>> {                  \
        ntt::vector<float, vl> operator()(float f) const noexcept {            \
            return __riscv_vfmv_v_f_f32m##lmul(f, vl);                         \
        }                                                                      \
    };

RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, 1), 1)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, 2), 2)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, 4), 4)
RVV_LOAD_SCALAR_FLOAT32(NTT_VL(sizeof(float) * 8, 8), 8)

} // namespace nncase::ntt::tensor_ops