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
#include "../tensor.h"
#include "nncase/ntt/dimension.h"
#include <cstddef>

namespace nncase::ntt {
namespace detail {
template <Tensor TA, Tensor TB, Tensor TC> class gather_impl {
  public:
    inline static constexpr auto rank = TA::rank();
    inline static constexpr auto indices_rank = TB::rank();

    template <FixedDimension TAxis>
    constexpr void operator()(const TA &input, const TB &indices, TC &output,
                              const TAxis &axis) noexcept {
        dynamic_shape_t<rank> in_index{};
        ntt::apply(output.shape(), [&](auto out_index) {
            // indices_index = out_index[axis:]
            const auto indices_index =
                out_index.template slice<axis, indices_rank>();

            // in_index[:axis] = out_index[:axis]
            // in_index[axis] = indices(indices_index)
            // in_index[axis:] = out_index[indices_rank+axis:]
            const auto in_index =
                out_index.template slice<0, axis>()
                    .append((dim_t)indices(indices_index))
                    .concat(out_index.template slice<axis + indices_rank,
                                                     rank - (axis + 1)>());
            output(out_index) = input(in_index);
        });
    }
};

} // namespace detail

template <Tensor TA, Tensor TB, class TC, FixedDimension TAxis>
void gather(const TA &input, const TB &indices, TC &&output,
            const TAxis &axis) noexcept {
    detail::gather_impl<TA, TB, std::decay_t<TC>> impl;
    impl(input, indices, output, axis);
}
} // namespace nncase::ntt
