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

namespace nncase::ntt {

template <Tensor TInput, Tensor TCos, Tensor TSin, class TOut>
void rope(const TInput &input, const TCos &cos, const TSin &sin,
          TOut &&output) {
    const auto domain = input.shape().template slice<0, TInput::rank() - 1>();
    const auto half_dim = input.shape().back() / 2_dim;

    ntt::apply(domain, [&](const auto &index) {
        const auto cos_sin_index = index.template slice<1>();

        // 1st half
        ntt::loop<half_dim>([&](const auto &i) {
            output(index.append(i)) =
                input(index.append(i)) * cos(cos_sin_index.append(i)) +
                -input(index.append(i + half_dim)) *
                    sin(cos_sin_index.append(i));
        });

        // 2nd half
        ntt::loop<half_dim>([&](const auto &i) {
            output(index.append(half_dim + i)) =
                input(index.append(half_dim + i)) *
                    cos(cos_sin_index.append(half_dim + i)) +
                input(index.append(i)) *
                    sin(cos_sin_index.append(half_dim + i));
        });
    });
}
} // namespace nncase::ntt
