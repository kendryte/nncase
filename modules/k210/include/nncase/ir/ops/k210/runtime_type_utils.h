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
#include <nncase/ir/ir_types.h>
#include <nncase/runtime/k210/runtime_types.h>

namespace nncase::ir::k210 {
template <class T>
runtime::k210::kpu_shape_t to_kpu_shape(const xt::dynamic_shape<T> &in_shape,
                                        T default_val = 1) {
    assert(in_shape.size() <= 4);

    runtime::k210::kpu_shape_t r_in_shape{};
    const auto in_ext = 4 - (int32_t)in_shape.size();

    for (int32_t i = 0; i < in_ext; i++)
        r_in_shape[i] = int32_t(default_val);
    for (size_t i = in_ext; i < 4; i++)
        r_in_shape[i] = int32_t(in_shape[i - in_ext]);
    return r_in_shape;
}
} // namespace nncase::ir::k210
