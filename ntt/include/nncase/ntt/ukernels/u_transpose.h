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
#include "../loop.h"
#include <cstddef>
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, bool Arch>
class u_transpose {
  public:
    constexpr void operator()(const TIn &input, TOut &output) noexcept {
        constexpr auto domain = typename TIn::shape_type{};
        auto out_index = ranked_shape<domain.rank()>{};
        apply(domain, [&](auto index) {
            loop<domain.rank()>(
                [&](auto i) { out_index[i] = index[TPerm::at(i)]; });
            output(out_index) = input(index);
        });
    }
};
} // namespace ukernels

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut>
constexpr void u_transpose(const TIn &input, TOut &&output) noexcept {
    ukernels::u_transpose<TPerm, std::decay_t<TIn>, std::decay_t<TOut>, true>
        impl;
    impl(input, output);
}

} // namespace nncase::ntt
