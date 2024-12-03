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
#include "../sharded_tensor.h"
#include "../tensor_traits.h"
#include "nncase/ntt/kernels/copy.h"
#include "nncase/ntt/primitive_ops.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/sharding.h"
#include "nncase/ntt/tensor.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace tar {
extern uint8_t collective_pool_ptr[];
}

namespace nncase::ntt {
template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept;

namespace detail {
template <class SrcTensor, class DestTensor> struct reshard_impl;

// shard
template <IsTensor SrcTensor, IsShardedTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename DestTensor::mesh_type;
    using sharding_type = typename DestTensor::sharding_type;

    // Make TestGatherReduceScatter happy.
    // static_assert(std::is_same_v<typename
    // sharding_type::implicit_policy_type,
    //                              distributed::shard_policy::B>,
    //               "Cannot shard to a non-Broadcast sharding type.");

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        auto local_mesh_index = mesh_type::local_index();
        auto global_offset =
            sharding_type::global_offset(dest.global_shape(), local_mesh_index);
        auto local = dest.local();
        tensor_copy(src.view(global_offset, local.shape()), local);
    }
};

template <size_t Rank> struct slice_with_global_offset {
    ranked_shape<Rank> global_offset;
    ranked_shape<Rank> local_offset;
    ranked_shape<Rank> shape;
};

// unshard
template <IsShardedTensor SrcTensor, IsTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename SrcTensor::mesh_type;
    using sharding_type = typename SrcTensor::sharding_type;
    using global_shape_type = typename SrcTensor::global_shape_type;
    using local_shape_type = typename SrcTensor::local_shape_type;

    static constexpr size_t rank = global_shape_type::rank();

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        auto local_mesh_index = mesh_type::local_index();
        auto global_shape = src.global_shape();
        auto slice =
            shard_to_slice_with_global_offset(global_shape, local_mesh_index);
        if (slice.shape.length() != 0) {
            // Not empty slice
            auto local = src.local().view(slice.local_offset, slice.shape);
            tensor_copy(local, dest.view(slice.global_offset, slice.shape));
        }
        distributed::topology_synchronize();
    }

  private:
    static constexpr slice_with_global_offset<rank>
    shard_to_slice_with_global_offset(
        const global_shape_type &global_shape,
        const typename mesh_type::index_type &shard_index) {
        slice_with_global_offset<rank> local_slice;

        // 1. Fill split axes.
        {
            auto get_split_axis_slice_impl = [&]<size_t Axis>() {
                auto global_dim = global_shape.at(Axis);
                using policy_t = std::tuple_element_t<
                    Axis, typename sharding_type::axis_policy_type>;
                if constexpr (!std::is_same_v<policy_t,
                                              distributed::shard_policy::I>) {
                    local_slice.global_offset[Axis] =
                        policy_t::template global_offset<mesh_type>(
                            global_dim, shard_index);
                    local_slice.local_offset[Axis] = 0;
                    local_slice.shape[Axis] =
                        policy_t::template local_dim<mesh_type>(global_dim,
                                                                shard_index);
                }
            };
            auto get_split_axis_slice =
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    (get_split_axis_slice_impl.template operator()<Is>(), ...);
                };
            get_split_axis_slice(
                std::make_index_sequence<global_shape_type::rank()>{});
        }

        // 2. Fill non-split axes.
        {
            constexpr auto non_split_mesh_dims = get_non_split_mesh_dims();
            constexpr auto non_split_mesh_strides =
                default_strides(non_split_mesh_dims);
            constexpr auto non_split_tensor_axes = get_non_split_tensor_axes();
#if defined(__GNUC__) && !defined(__clang__)
            constexpr auto split_counts =
                get_non_split_tensor_axes_split_counts(glglobal_shape_type{});
            constexpr auto split_count_strides = default_strides(split_counts);
#else
            auto split_counts =
                get_non_split_tensor_axes_split_counts(global_shape);
            auto split_count_strides = default_strides(split_counts);
#endif

            auto non_split_mesh_indexes =
                get_non_split_mesh_indexes(shard_index);
            auto non_split_mesh_linear_offset =
                linear_offset(non_split_mesh_indexes, non_split_mesh_strides);

            auto local_split_id = unravel_index(non_split_mesh_linear_offset,
                                                split_count_strides);

            for (size_t i = 0; i < non_split_tensor_axes.rank(); i++) {
                auto axis = non_split_tensor_axes[i];
                auto global_dim = global_shape.at(axis);
                auto split_count = split_counts[i];
                auto local_dim = ntt::ceil_div(global_dim, split_count);
                auto global_offset = local_dim * local_split_id[i];
                if (global_offset >= global_dim) {
                    local_slice.global_offset[axis] = 0;
                    local_slice.local_offset[axis] = 0;
                    local_slice.shape[axis] = 0;
                } else {
                    auto global_end =
                        std::min(global_offset + local_dim, global_dim);
                    local_slice.global_offset[axis] = global_offset;
                    local_slice.local_offset[axis] = global_offset;
                    local_slice.shape[axis] = global_end - global_offset;
                }
            }
        }
        return local_slice;
    }

    static constexpr auto get_non_split_mesh_dims() noexcept {
        constexpr auto non_split_mesh_axes = get_non_split_mesh_axes();
        ranked_shape<non_split_mesh_axes.rank()> non_split_mesh_dims{};
        for (size_t i = 0; i < non_split_mesh_dims.rank(); i++) {
            auto axis = non_split_mesh_axes[i];
            non_split_mesh_dims[i] = mesh_type::shape_type::at(axis);
        }
        return non_split_mesh_dims;
    }

    static constexpr auto get_non_split_mesh_indexes(
        const typename mesh_type::index_type &shard_index) noexcept {
        constexpr auto non_split_mesh_axes = get_non_split_mesh_axes();
        ranked_shape<non_split_mesh_axes.rank()> non_split_mesh_indexes{};
        for (size_t i = 0; i < non_split_mesh_axes.rank(); i++) {
            auto axis = non_split_mesh_axes[i];
            non_split_mesh_indexes[i] = shard_index[axis];
        }
        return non_split_mesh_indexes;
    }

    template <class Shape>
    static constexpr auto
    get_non_split_tensor_axes_split_counts(const Shape &shape) noexcept {
        constexpr auto non_split_mesh_dims = get_non_split_mesh_dims();
        constexpr auto expected_split_count =
            (std::ptrdiff_t)non_split_mesh_dims.length();

        // Split non-split axes into split_count groups, based on each size of
        // the tensor dimensions.
        constexpr auto non_split_tensor_axes = get_non_split_tensor_axes();
        ranked_shape<non_split_tensor_axes.rank()> split_counts{};

        // 1. Calculate the initial split counts.
        {
            std::array<float, non_split_tensor_axes.rank()> log_dims;
            for (size_t i = 0; i < non_split_tensor_axes.rank(); i++) {
                auto dim = (float)shape.at(non_split_tensor_axes[i]);
                log_dims[i] = std::log(dim);
            }

            auto total_log_dim =
                std::accumulate(log_dims.begin(), log_dims.end(), 0.f);
            for (size_t i = 0; i < non_split_tensor_axes.rank(); i++) {
                auto split_factor = total_log_dim == 0.f
                                        ? 0.f
                                        : (log_dims[i] / total_log_dim *
                                           std::log(expected_split_count));
                auto dim = shape.at(non_split_tensor_axes[i]);
                split_counts[i] = std::max(
                    size_t(1),
                    std::min(dim, static_cast<size_t>(std::exp(split_factor))));
            }
        }

        // 2. Adjust the split counts to make sure the total count is similar to
        // the expected.
        {
            auto total_splits = (std::ptrdiff_t)std::accumulate(
                split_counts.begin(), split_counts.end(), size_t(1),
                std::multiplies<size_t>());
            auto total_diff = expected_split_count - total_splits;
            bool improved;
            do {
                size_t adjust_axis = 0;
                std::ptrdiff_t adjust_delta = 0;
                std::ptrdiff_t adjust_total_splits = total_splits;
                std::ptrdiff_t adjust_diff = total_diff;

                if (total_diff < 0) {
                    for (size_t i = 0; i < split_counts.rank(); i++) {
                        auto split_count = split_counts[i];
                        if (split_count > 1) {
                            auto new_total_splits =
                                total_splits / split_count * (split_count - 1);
                            auto new_total_diff =
                                expected_split_count - new_total_splits;
                            if ((adjust_diff < 0 &&
                                 new_total_diff > adjust_diff) ||
                                (adjust_diff > 0 && new_total_diff >= 0 &&
                                 new_total_diff < adjust_diff)) {
                                adjust_axis = i;
                                adjust_delta = -1;
                                adjust_total_splits = new_total_splits;
                                adjust_diff = new_total_diff;
                            }
                        }
                    }
                } else if (total_diff > 0) {
                    for (size_t i = 0; i < split_counts.rank(); i++) {
                        auto split_count = split_counts[i];
                        if (split_count < shape.at(non_split_tensor_axes[i])) {
                            auto new_total_splits =
                                total_splits / split_count * (split_count + 1);
                            auto new_total_diff =
                                expected_split_count - new_total_splits;
                            if (new_total_diff >= 0 &&
                                new_total_diff < adjust_diff) {
                                adjust_axis = i;
                                adjust_delta = 1;
                                adjust_total_splits = new_total_splits;
                                adjust_diff = new_total_diff;
                            }
                        }
                    }
                }

                if (adjust_delta) {
                    split_counts[adjust_axis] += adjust_delta;
                    total_splits = adjust_total_splits;
                    total_diff = adjust_diff;
                    improved = true;
                } else {
                    improved = false;
                }
            } while (improved);
        }
        return split_counts;
    }

    static constexpr auto get_non_split_tensor_axes() noexcept {
        constexpr auto split_axes_mask = get_split_tensor_axes_mask();
        constexpr auto non_split_axes_count =
            std::count_if(split_axes_mask.begin(), split_axes_mask.end(),
                          [](auto x) { return x == 0; });
        ranked_shape<non_split_axes_count> non_split_axes{};
        size_t non_split_axis_index = 0;
        for (size_t i = 0; i < split_axes_mask.rank(); i++) {
            if (split_axes_mask[i] == 0) {
                non_split_axes[non_split_axis_index++] = i;
            }
        }
        return non_split_axes;
    }

    static constexpr ranked_shape<rank> get_split_tensor_axes_mask() noexcept {
        ranked_shape<rank> split_axes{};
        auto mark_split_axes_impl = [&]<size_t Axis>() {
            using policy_t =
                std::tuple_element_t<Axis,
                                     typename sharding_type::axis_policy_type>;
            if constexpr (!std::is_same_v<policy_t,
                                          distributed::shard_policy::I>) {
                split_axes[Axis] = 1;
            }
        };
        auto mark_split_axes = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (mark_split_axes_impl.template operator()<Is>(), ...);
        };
        mark_split_axes(std::make_index_sequence<rank>{});
        return split_axes;
    }

    static constexpr auto get_non_split_mesh_axes() noexcept {
        constexpr auto split_axes_mask = get_split_mesh_axes_mask();
        constexpr auto non_split_axes_count =
            std::count_if(split_axes_mask.begin(), split_axes_mask.end(),
                          [](auto x) { return x == 0; });
        ranked_shape<non_split_axes_count> non_split_axes{};
        size_t non_split_axis_index = 0;
        for (size_t i = 0; i < split_axes_mask.rank(); i++) {
            if (split_axes_mask[i] == 0) {
                non_split_axes[non_split_axis_index++] = i;
            }
        }
        return non_split_axes;
    }

    static constexpr ranked_shape<mesh_type::rank()>
    get_split_mesh_axes_mask() noexcept {
        ranked_shape<mesh_type::rank()> split_axes{};
        auto mark_split_axes_impl = [&]<size_t Axis>() {
            using policy_t =
                std::tuple_element_t<Axis,
                                     typename sharding_type::axis_policy_type>;
            if constexpr (!std::is_same_v<policy_t,
                                          distributed::shard_policy::I>) {
                for (size_t i = 0; i < split_axes.rank(); i++) {
                    if (policy_t::axes_type::contains(i))
                        split_axes[i] = 1;
                }
            }
        };
        auto mark_split_axes = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (mark_split_axes_impl.template operator()<Is>(), ...);
        };
        mark_split_axes(std::make_index_sequence<rank>{});
        return split_axes;
    }
};

// reshard
template <IsShardedTensor SrcTensor, IsShardedTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename SrcTensor::mesh_type;
    using src_sharding_type = typename SrcTensor::sharding_type;
    using dest_sharding_type = typename DestTensor::sharding_type;

    static_assert(std::is_same_v<mesh_type, typename DestTensor::mesh_type>,
                  "Cannot reshard between different mesh types.");

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        copy_to_global(src);
        copy_from_global(dest);
    }

  private:
    void copy_to_global(const SrcTensor &src) noexcept {
        using local_tensor_type = typename SrcTensor::local_tensor_type;
        constexpr auto global_size = linear_size(
            typename SrcTensor::global_shape_type{},
            default_strides(typename SrcTensor::global_shape_type{}));
        auto global_buffer =
            std::span<typename local_tensor_type::element_type, global_size>(
                reinterpret_cast<typename local_tensor_type::element_type *>(
                    tar::collective_pool_ptr),
                global_size);
        tensor_view<typename local_tensor_type::element_type,
                    typename SrcTensor::global_shape_type>
            global_tensor(global_buffer);
        reshard(src, global_tensor);
    }

    void copy_from_global(DestTensor &dest) noexcept {
        using local_tensor_type = typename DestTensor::local_tensor_type;
        constexpr auto global_size = linear_size(
            typename DestTensor::global_shape_type{},
            default_strides(typename DestTensor::global_shape_type{}));
        auto global_buffer =
            std::span<typename local_tensor_type::element_type, global_size>(
                reinterpret_cast<typename local_tensor_type::element_type *>(
                    tar::collective_pool_ptr),
                global_size);
        tensor_view<typename local_tensor_type::element_type,
                    typename DestTensor::global_shape_type>
            global_tensor(global_buffer);
        reshard(global_tensor, dest);
        distributed::topology_synchronize();
    }
};
} // namespace detail

template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept {
    detail::reshard_impl<std::decay_t<SrcTensor>, std::decay_t<DestTensor>>()(
        src, dest);
}
} // namespace nncase::ntt
