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
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../ukernels.h"
#include "../utility.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut> class reshape_impl {
  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        using element_type = element_or_scalar_t<TIn>;
        auto input_conti_dims = contiguous_dims(input.shape(), input.strides());
        auto output_conti_dims =
            contiguous_dims(output.shape(), output.strides());

        auto input_conti_size =
            input.shape().aggregate(1, [&](const auto acc, auto dim, auto i) {
                return i >= input.shape().rank() - input_conti_dims ? acc * dim
                                                                    : acc;
            });
        auto output_conti_size =
            output.shape().aggregate(1, [&](const auto acc, auto dim, auto i) {
                return i >= output.shape().rank() - output_conti_dims
                           ? acc * dim
                           : acc;
            });

        // may need other constraints, just assuming possible reshape is filterd
        // by type-infer.
        auto len = input_conti_size == output_conti_size ? input_conti_size : 1;

        const size_t size = input.size();
        for (size_t i = 0; i < size; i += len) {
            auto in_index = unravel_index(i, input.shape());
            auto out_index = unravel_index(i, output.shape());
            ntt::u_unary(ntt::ops::copy<element_type>{}, &input(in_index), 1,
                         &output(out_index), 1, len);
        }
    }
};
} // namespace detail

template <Tensor TIn, class TOut>
void reshape(const TIn &input, TOut &&output) {
    detail::reshape_impl<TIn, std::decay_t<TOut>>()(input, output);
}
} // namespace nncase::ntt
