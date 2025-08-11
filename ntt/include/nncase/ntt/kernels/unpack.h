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
#include "../shape_infer/devectorize.h"
#include "../ukernels.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut, size_t AxesRank> class unpack_impl {
  public:
    using TVec = typename TIn::element_type;

    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TAxes &axes) {
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto elem_rank = TAxes::rank();
        constexpr auto elem_shape =
            TVec::shape().template slice<0, TAxes::rank()>();

        const auto conti_dims_input =
            contiguous_dims(input.shape(), input.strides());
        const auto conti_dims_output =
            contiguous_dims(output.shape(), output.strides());

        if (conti_dims_input == in_rank && conti_dims_output == out_rank &&
            AxesRank <= 2) {
            ntt::u_unpack(input, output, axes);
        } else {
            const auto domain = input.shape().concat(elem_shape);
            apply(domain, [&](auto index) {
                const auto in_index = index.template slice<0, in_rank>();
                const auto elem_index =
                    index.template slice<in_rank, elem_rank>();
                const auto out_index_template =
                    index.template slice<0, in_rank>();
                const auto out_index = axes.aggregate(
                    out_index_template,
                    [&](const auto cnt_out_index, auto axis, auto i) {
                        return cnt_out_index.template replace_at<axis>(
                            cnt_out_index[axis] * elem_shape[i] +
                            index[in_rank + i]);
                    });
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};
} // namespace detail

template <Tensor TIn, class TOut, FixedDimensions TAxes>
void unpack(const TIn &input, TOut &&output, const TAxes &axes) noexcept {
    detail::unpack_impl<TIn, std::decay_t<TOut>, TAxes::rank()> impl;
    impl(input, output, axes);
}
} // namespace nncase::ntt
