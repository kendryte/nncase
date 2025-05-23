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
#include "../apply.h"
#include "../loop.h"
#include "../shape_infer/reduce_axis.h"
#include "../utility.h"
#include <tuple>

namespace nncase::ntt {
template <size_t Axis, Tensor... TInputs, class TOut>
void stack(const std::tuple<TInputs...> &inputs, TOut &&output) {
    auto domain = output.shape().template replace_at<Axis>(dim_one);
    apply(domain, [&](auto out_index) {
        auto left_shape = slice_dims<Axis>(out_index);
        auto right_shape = slice_dims<output.rank() - Axis - 1>(out_index);
        auto in_index = concat_dims(left_shape, right_shape);
        loop<sizeof...(TInputs)>([&](auto i) {
            auto input = std::get<i>(inputs);
            for (out_index[Axis] = 0; out_index[Axis] < inputs.size();
                 out_index[Axis]++) {
                output(out_index) = input(in_index);
            }
        });
    });
}
} // namespace nncase::ntt
