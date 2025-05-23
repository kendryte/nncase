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
#include "../ukernels.h"
#include "../utility.h"
#include "../sharded_tensor.h"
#include "../tensor_traits.h"
#include <cstddef>
namespace nncase::ntt {

namespace detail {

template <class Shape, class InStrides, class IndexShape, class IndicesStrides> class gather_impl;

// weights: static
// indices: static
template <size_t... Dims, size_t... InStrides, size_t... IndicesRank, size_t... IndicesStrides>
class gather_impl<fixed_shape<Dims...>, fixed_strides<InStrides...>,
                  fixed_shape<IndicesRank...>,
                  fixed_strides<IndicesStrides...>> {
  public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        constexpr auto rank = TA::shape_type::rank();
        using slice_type = element_or_scalar_t<TB>;

        constexpr size_t indices_len = TB::size();

        segment segments[indices_len];
        size_t count = find_continuous_segments(
            (const slice_type *)(indices.elements().data()), indices_len,
            segments);

        auto addr_output_byte =
            reinterpret_cast<unsigned char *>(output.buffer().data());
        auto addr_output_element = output.buffer().data();

        
        constexpr auto indices_rank = TB::shape_type::rank();
        constexpr auto out_shape = std::decay_t<TC>::shape();
        ranked_shape<rank> in_index;
        ranked_shape<indices_rank> indices_index;
        ranked_shape<rank> src_index;
        for (size_t i = 0; i < rank; i++) {
            src_index[i] = 0;
        }
        
        // Check if input is a sharded tensor
        if constexpr (IsShardedTensor<TA>) {
            using TensorTypeA = typename TA::local_tensor_type;
            using element_type = element_or_scalar_t<TensorTypeA>;
            constexpr auto element_size = sizeof(element_type);

            using mesh_type = typename TA::mesh_type;
            using sharding_type = typename TA::sharding_type;

            auto local_shape = input.local().shape();
            auto local_strides = input.local().strides();
            auto input_conti_dims = contiguous_dims(local_shape, local_strides);
            auto local_mesh_index = mesh_type::local_index();
            auto global_offset =
                sharding_type::global_offset(input.shape(), local_mesh_index);

            auto domain_before_axis = slice_dims<Axis>(local_shape);
            constexpr auto domain_after_axis = slice_dims<rank - Axis - 1, Axis + 1>(local_shape);

            size_t axis_global_start = global_offset[Axis];
            size_t axis_global_end = axis_global_start + local_shape[Axis];

            if (input_conti_dims == rank && count != indices_len) {
                apply(domain_before_axis, [&](auto index) {
                    for (size_t i = 0; i < count; i++) {
                        auto seq = segments[i];
                        for (size_t j = 0; j < Axis; j++) {
                            src_index[j] = index[j];
                        }

                        auto global_idx = indices.elements()[seq.start];
                        size_t len = seq.length * domain_after_axis.length() * element_size;
                        if (global_idx >= axis_global_start && global_idx < axis_global_end) {
                            src_index[Axis] = global_idx - axis_global_start;
                            std::memcpy(addr_output_byte, &(input.local()(src_index)), len);
                        } else {
                            // Index is outside the local shard's range, fill with zeros
                            std::memset(addr_output_byte, 0, len);
                        }
                        addr_output_byte += len;
                    }
                });
            } else if (input_conti_dims == rank) {
                apply(domain_before_axis, [&](auto index) {
                    for (size_t i = 0; i < TB::size(); i++) {
                        for (size_t j = 0; j < Axis; j++) {
                            src_index[j] = index[j];
                        }

                        auto global_idx = indices.elements()[i];
                        constexpr auto len = domain_after_axis.length();
                        if (global_idx >= axis_global_start && global_idx < axis_global_end) {
                            src_index[Axis] = global_idx - axis_global_start;
                            auto addr_input = reinterpret_cast<const element_type *>(
                                &(input.local()(src_index)));

                            u_unary<ntt::ops::copy<element_type>, element_type>(
                                addr_input, 1, addr_output_element, 1, len);
                        } else {
                            // Index is outside the local shard's range, fill with zeros
                            std::memset(addr_output_element, 0, len * element_size);
                        }
                        addr_output_element += len;
                    }
                });
            } else {
                apply(out_shape, [&](auto out_index) {
                    // in_index[:axis] = out_index[:axis]
                    loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

                    // in_index[axis] = indices(indices_index)
                    loop<indices_rank>(
                        [&](auto i) { indices_index[i] = out_index[i + Axis]; });
                    auto global_idx = indices(indices_index);

                    if (global_idx >= axis_global_start && global_idx < axis_global_end) {
                        in_index[Axis] = global_idx - axis_global_start;

                        // in_index[axis:] = out_index[axis:]
                        loop<rank - (Axis + 1)>([&](auto i) {
                            in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
                        });
                        output(out_index) = input.local()(in_index);
                    } else {
                        // Index is outside the local shard's range, fill with zeros
                        output(out_index) = element_type{0};
                    }
                });
            }
        } else {
            using element_type = element_or_scalar_t<TA>;
            constexpr auto element_size = sizeof(element_type);
            auto input_conti_dims = contiguous_dims(input.shape(), input.strides());
            auto domain_before_axis = slice_dims<Axis>(input.shape());
            constexpr auto domain_after_axis = slice_dims<rank - Axis - 1, Axis + 1>(TA::shape());
            // Original implementation for non-sharded tensors
            if (input_conti_dims == rank && count != indices_len) {
                apply(domain_before_axis, [&](auto index) {
                    for (size_t i = 0; i < count; i++) {
                        auto seq = segments[i];
                        for (size_t j = 0; j < Axis; j++) {
                            src_index[j] = index[j];
                        }
                        src_index[Axis] = indices.elements()[seq.start];
                        auto len =
                            seq.length * domain_after_axis.length() * element_size;
                        std::memcpy(addr_output_byte, &(input(src_index)), len);
                        addr_output_byte += len;
                    }
                });
            } else if (input_conti_dims == rank) {
                apply(domain_before_axis, [&](auto index) {
                    for (size_t i = 0; i < TB::size(); i++) {

                        for (size_t j = 0; j < Axis; j++) {
                            src_index[j] = index[j];
                        }
                        src_index[Axis] = indices.elements()[i];
                        auto addr_input = reinterpret_cast<const element_type *>(
                            &(input(src_index)));
                        constexpr auto len = domain_after_axis.length();

                        u_unary<ntt::ops::copy<element_type>, element_type>(
                            addr_input, 1, addr_output_element, 1, len);
                        addr_output_element += len;
                    }
                });
            } else {
                apply(out_shape, [&](auto out_index) {
                    // in_index[:axis] = out_index[:axis]
                    loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

                    // in_index[axis] = indices(indices_index)
                    loop<indices_rank>(
                        [&](auto i) { indices_index[i] = out_index[i + Axis]; });
                    in_index[Axis] = indices(indices_index);

                    // in_index[axis:] = out_index[axis:]
                    loop<rank - (Axis + 1)>([&](auto i) {
                        in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
                    });
                    output(out_index) = input(in_index);
                });
            }
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

// weights: dynamic
// indices: dynamic
template <size_t Rank, class InStrides, size_t IndicesRank, class IndicesStrides>
class gather_impl<ranked_shape<Rank>, InStrides, ranked_shape<IndicesRank>, IndicesStrides> {
  public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        ranked_shape<Rank> in_index;
        constexpr auto indices_rank = TB::shape_type::rank();
        ranked_shape<indices_rank> indices_index;

        // Check if input is a sharded tensor
        if constexpr (IsShardedTensor<TA>) {
            using mesh_type = typename TA::mesh_type;
            using sharding_type = typename TA::sharding_type;

            auto local_mesh_index = mesh_type::local_index();
            auto global_offset = sharding_type::global_offset(input.shape(), local_mesh_index);
            auto local_shape = input.local().shape();

            size_t axis_global_start = global_offset[Axis];
            size_t axis_global_end = axis_global_start + local_shape[Axis];

            apply(output.shape(), [&](auto out_index) {
                // in_index[:axis] = out_index[:axis]
                loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

                // in_index[axis] = indices(indices_index)
                loop<indices_rank>(
                    [&](auto i) { indices_index[i] = out_index[i + Axis]; });
                auto global_idx = indices(indices_index);

                if (global_idx >= axis_global_start && global_idx < axis_global_end) {
                    in_index[Axis] = global_idx - axis_global_start;

                    // in_index[axis:] = out_index[axis:]
                    loop<Rank - (Axis + 1)>([&](auto i) {
                        in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
                    });
                    output(out_index) = input.local()(in_index);
                } else {
                    // Index is outside the local shard's range, fill with zeros
                    output(out_index) = element_or_scalar_t<TA>{0};
                }
            });
        } else {
            apply(output.shape(), [&](auto out_index) {
                // in_index[:axis] = out_index[:axis]
                loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

                // in_index[axis] = indices(indices_index)
                loop<indices_rank>(
                    [&](auto i) { indices_index[i] = out_index[i + Axis]; });
                in_index[Axis] = indices(indices_index);

                // in_index[axis:] = out_index[axis:]
                loop<Rank - (Axis + 1)>([&](auto i) {
                    in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
                });
                output(out_index) = input(in_index);
            });
        }
    }
};
// weights: dynamic indices: static
template <size_t Rank, class InStrides, size_t... IndicesDims, class IndicesStrides>
class gather_impl<ranked_shape<Rank>, InStrides, fixed_shape<IndicesDims...>, IndicesStrides> {
public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        constexpr size_t indices_rank = sizeof...(IndicesDims);

        ranked_shape<Rank> in_index;
        ranked_shape<indices_rank> indices_index;

        apply(output.shape(), [&](auto out_index) {
            loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

            loop<indices_rank>([&](auto i) {
                indices_index[i] = out_index[Axis + i];
            });
            in_index[Axis] = indices(indices_index);

            loop<Rank - Axis - 1>([&](auto i) {
                in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
            });

            output(out_index) = input(in_index);
        });
    }
};

// weights: static
// indices: dynamic
template <size_t... Dims, class InStrides, size_t IndicesRank, class IndicesStrides>
class gather_impl<fixed_shape<Dims...>, InStrides, ranked_shape<IndicesRank>, IndicesStrides> {
public:
    template <size_t Axis, typename TA, typename TB, typename TC>
    constexpr void operator()(const TA &input, const TB &indices, TC &&output) {
        constexpr size_t rank = sizeof...(Dims);
        constexpr size_t indices_rank = IndicesRank;

        ranked_shape<rank> in_index;
        ranked_shape<indices_rank> indices_index;

        apply(output.shape(), [&](auto out_index) {
            loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

            loop<indices_rank>([&](auto i) {
                indices_index[i] = out_index[Axis + i];
            });
            in_index[Axis] = indices(indices_index);

            loop<rank - Axis - 1>([&](auto i) {
                in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
            });

            output(out_index) = input(in_index);
        });
    }
};

} // namespace detail

template <size_t Axis, typename TA, typename TB, typename TC>
void gather(const TA &input, const TB &indices, TC &&output) noexcept {

    if constexpr (IsShardedTensor<TA>) {
        using TensorTypeA = typename TA::local_tensor_type;
        using TensorTypeB = typename TB::local_tensor_type;
        detail::gather_impl<typename TensorTypeA::shape_type, typename TensorTypeA::strides_type,
                        typename TensorTypeB::shape_type, typename TensorTypeB::strides_type>
        impl;
        impl.template operator()<Axis>(input, indices, output);
    } else {
        detail::gather_impl<typename TA::shape_type, typename TA::strides_type,
                        typename TB::shape_type, typename TB::strides_type>
        impl;
        impl.template operator()<Axis>(input, indices, output);
    }
}

} // namespace nncase::ntt
