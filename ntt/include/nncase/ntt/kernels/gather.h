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
#include "../tensor_traits.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TA, Tensor TB, Tensor TC> class gather_impl {
  public:
    inline static constexpr auto rank = TA::rank();
    inline static constexpr auto indices_rank = TB::rank();

    template <FixedDimension TAxis>
    constexpr void operator()(const TA &input, const TB &indices, TC &output,
                              const TAxis &) noexcept {
        constexpr auto axis = TAxis{};
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

template <ShardedTensor TA, Tensor TB, Tensor TC>
class distributed_gather_impl {
  public:
    using mesh_type = typename TA::mesh_type;
    using local_input_tensor_type = typename TA::local_tensor_type;
    using element_type = local_input_tensor_type::value_type;

    inline static constexpr auto rank = TA::rank();
    inline static constexpr auto indices_rank = TB::rank();

    template <FixedDimension TAxis>
    constexpr void operator()(const TA &input, const TB &indices, TC &output,
                              const TAxis &) noexcept {
        constexpr auto axis = TAxis{};

        const auto local_mesh_index = mesh_type::local_index();
        const auto global_offset =
            input.sharding().global_offset(input.shape(), local_mesh_index);
        const auto local_shape = input.local().shape();

        const auto axis_global_start = global_offset[axis];
        const auto axis_global_end = axis_global_start + local_shape[axis];

        ntt::apply(output.shape(), [&](auto out_index) {
            // indices_index = out_index[axis:]
            const auto indices_index =
                out_index.template slice<axis, indices_rank>();
            auto global_idx = indices(indices_index);

            if (global_idx >= axis_global_start &&
                global_idx < axis_global_end) {
                // in_index[:axis] = out_index[:axis]
                // in_index[axis] = global_idx - axis_global_start
                // in_index[axis:] = out_index[indices_rank+axis:]
                const auto in_index =
                    out_index.template slice<0, axis>()
                        .append((dim_t)global_idx - axis_global_start)
                        .concat(out_index.template slice<axis + indices_rank,
                                                         rank - (axis + 1)>());
            } else {
                // Index is outside the local shard's range, fill with zeros
                output(out_index) = element_or_scalar_t<element_type>{0};
            }
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

template <ShardedTensor TA, Tensor TB, class TC, FixedDimension TAxis>
void gather(const TA &input, const TB &indices, TC &&output,
            const TAxis &axis) noexcept {
    detail::distributed_gather_impl<TA, TB, std::decay_t<TC>> impl;
    impl(input, indices, output, axis);
}
} // namespace nncase::ntt
