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
            const auto indices_index =
                out_index.template slice<axis, indices_rank>();

            const auto in_index =
                out_index.template slice<0, axis>()
                    .append((dim_t)indices(indices_index))
                    .concat(out_index.template slice<axis + indices_rank,
                                                     rank - (axis + 1)>());
            output(out_index) = input(in_index);
        });
    }

    template <FixedDimension TAxis>
        requires FixedTensor<TB>
    constexpr void operator()(const TA &input, const TB &indices, TC &output,
                              const TAxis &) noexcept {

        using slice_type = element_or_scalar_t<TB>;

        constexpr size_t indices_len = TB::size();

        segment segments[indices_len];
        size_t count = find_continuous_segments(
            (const slice_type *)(indices.elements().data()), indices_len,
            segments);

        auto addr_output_byte =
            reinterpret_cast<unsigned char *>(output.buffer().data());
        auto addr_output_element = output.buffer().data();

        constexpr auto indices_rank = TB::rank();
        dynamic_shape_t<rank> in_index;
        dynamic_shape_t<indices_rank> indices_index;
        dynamic_shape_t<rank> src_index;
        ntt::loop<rank>([&](auto &i) { src_index[i] = 0; });

        using element_type = element_or_scalar_t<TA>;
        constexpr auto element_size = sizeof(element_type);
        auto input_conti_dims = contiguous_dims(input.shape(), input.strides());
        auto domain_before_axis =
            input.shape().template slice<0, TAxis::value>();
        auto domain_after_axis =
            input.shape()
                .template slice<TAxis::value + 1, rank - TAxis::value - 1>();

        // Original implementation for non-sharded tensors
        if (input_conti_dims == rank && count != indices_len) {
            ntt::apply(domain_before_axis, [&](auto index) {
                for (size_t i = 0; i < count; i++) {
                    auto seq = segments[i];
                    loop<TAxis::value>(
                        [&](auto j) { src_index[j] = index[j]; });
                    src_index[fixed_dim_v<TAxis::value>] =
                        indices.elements()[seq.start];
                    auto len =
                        seq.length * domain_after_axis.length() * element_size;
                    std::memcpy(addr_output_byte, &(input(src_index)), len);
                    addr_output_byte += len;
                }
            });
        } else if (input_conti_dims == rank) {
            ntt::apply(domain_before_axis, [&](auto index) {
                for (size_t i = 0; i < TB::size(); i++) {
                    loop<TAxis::value>(
                        [&](auto j) { src_index[j] = index[j]; });
                    src_index[fixed_dim_v<TAxis::value>] =
                        indices.elements()[i];
                    auto addr_input = reinterpret_cast<const element_type *>(
                        &(input(src_index)));
                    auto len = domain_after_axis.length();

                    ntt::u_unary(ntt::ops::copy<element_type>{}, addr_input, 1,
                                 addr_output_element, 1, len);
                    addr_output_element += len;
                }
            });
        } else {
            constexpr auto axis = TAxis{};
            ntt::apply(output.shape(), [&](auto out_index) {
                const auto indices_index =
                    out_index.template slice<axis, indices_rank>();
                const auto in_index =
                    out_index.template slice<0, axis>()
                        .append((dim_t)indices(indices_index))
                        .concat(out_index.template slice<axis + indices_rank,
                                                         rank - (axis + 1)>());
                output(out_index) = input(in_index);
            });
        }
    }

  private:
    struct segment {
        size_t start;
        size_t length;
    };

    template <typename T>
    size_t find_continuous_segments(const T *arr, size_t arrSize,
                                    segment *segments) {
        if (arrSize == 0)
            return 0;

        size_t segment_count = 0;
        size_t start = 0;
        size_t length = 1;

        for (size_t i = 1; i < arrSize; ++i) {
            if (arr[i] == arr[i - 1] + 1) {
                ++length;
            } else {
                segments[segment_count].start = start;
                segments[segment_count].length = length;
                ++segment_count;
                start = i;
                length = 1;
            }
        }

        segments[segment_count].start = start;
        segments[segment_count].length = length;
        ++segment_count;

        return segment_count;
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
        const auto local_input = input.local();
        const auto local_shape = local_input.shape();

        const auto axis_global_start = global_offset[axis];
        const auto axis_global_end = axis_global_start + local_shape[axis];

        ntt::apply(output.shape(), [&](auto out_index) {
            const auto indices_index =
                out_index.template slice<axis, indices_rank>();
            auto global_idx = indices(indices_index);

            if (global_idx >= axis_global_start &&
                global_idx < axis_global_end) {
                const auto in_index =
                    out_index.template slice<0, axis>()
                        .append((dim_t)global_idx - axis_global_start)
                        .concat(out_index.template slice<axis + indices_rank,
                                                         rank - (axis + 1)>());
                output(out_index) = local_input(in_index);
            } else {
                // Index is outside the local shard's range, fill with zeros
                output(out_index) = element_type{};
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
