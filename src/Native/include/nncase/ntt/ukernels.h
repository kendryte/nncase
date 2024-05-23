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
#include "primitive_ops.h"
#include "tensor.h"
#include "tensor_traits.h"

namespace nncase::ntt::ukernels {
template <size_t M, size_t N, size_t MStrides, bool Arch, class TIn, class TOut>
class upack {
  public:
    constexpr void operator()(const TIn *input, TOut *output) noexcept {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                output[j](i) = input[i * MStrides + j];
            }
        }

        if constexpr (M < TOut::shape_type::length()) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = M; i < TOut::shape_type::length(); i++) {
                    output[j](i) = (TIn)0;
                }
            }
        }
    }
};
} // namespace nncase::ntt::ukernels

namespace nncase::ntt {
template <size_t M, size_t N, size_t MStrides, class TIn, class TOut>
constexpr void upack(const TIn *input, TOut *output) noexcept {
    ukernels::upack<M, N, MStrides, true, std::decay_t<TIn>, std::decay_t<TOut>>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
