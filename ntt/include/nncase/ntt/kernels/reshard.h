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
#include "../distributed/mesh.h"
#include "../distributed/sharding.h"
#include "../distributed/topology.h"
#include "../primitive_ops.h"
#include "../shape.h"
#include "../tensor_traits.h"
#include "copy.h"
#include <cstddef>
#include <numeric>
#include <type_traits>

// #define ENABLE_RESHARD_DEBUG 1
#if ENABLE_RESHARD_DEBUG
#include <iostream>
#endif

namespace tar {
#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
__device__ extern uint8_t collective_pool_ptr[];
#else
extern uint8_t collective_pool_ptr[];
#endif
} // namespace tar

namespace nncase::ntt {
template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept;

namespace detail {
template <class SrcTensor, class DestTensor> struct reshard_impl;

#if ENABLE_RESHARD_DEBUG
template <typename Tshape>
constexpr void dump_shape(const std::string &info, Tshape shape) {
    std::cout << info << ": ";
    for (size_t i = 0; i < shape.rank(); i++)
        std::cout << shape[i] << " ";
    std::cout << std::endl;
}

template <typename T> void dump_tensor(const std::string &info, const T &t) {
    std::cout << info << ":";
    apply(t.shape(), [&](auto index) { std::cout << t(index) << " "; });
    std::cout << std::endl;
}
#endif

// shard
template <Tensor SrcTensor, ShardedTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename DestTensor::mesh_type;
    using sharding_type = typename DestTensor::sharding_type;

    // Make TestGatherReduceScatter happy.
    // static_assert(std::is_same_v<typename
    // sharding_type::implicit_policy_type,
    //                              distributed::shard_policy::B>,
    //               "Cannot shard to a non-Broadcast sharding type.");

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        const auto local_shard_index = mesh_type::local_index();
        const auto global_offset =
            dest.sharding().global_offset(dest.shape(), local_shard_index);
        auto local = dest.local();
        tensor_copy_sync(src.view(global_offset, local.shape()), local);
    }
};

template <size_t Rank> struct slice_with_global_offset {
    dynamic_shape_t<Rank> global_offset;
    dynamic_shape_t<Rank> local_offset;
    dynamic_shape_t<Rank> shape;
};

// unshard
template <ShardedTensor SrcTensor, Tensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename SrcTensor::mesh_type;
    using sharding_type = typename SrcTensor::sharding_type;
    using global_shape_type = typename SrcTensor::shape_type;
    using local_shape_type = typename SrcTensor::local_shape_type;

    static constexpr auto rank = global_shape_type::rank();

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        const auto local_shard_index = mesh_type::local_index();
        auto [global_offset, local_offset, shape] =
            shard_to_slice_with_global_offset(src, local_shard_index);
        if (shape.length() != 0) {
            // Not empty slice
            auto local = src.local().view(local_offset, shape);
            tensor_copy_sync(local, dest.view(global_offset, shape));
        }
        distributed::topology_synchronize();
    }

  private:
    template <distributed::ShardIndex<mesh_type> TShardIndex>
    static constexpr auto
    shard_to_slice_with_global_offset(const SrcTensor &src,
                                      const TShardIndex &shard_index) {
        // 1. Fill split axes.
        auto split_phase1 = src.shape().aggregate(
            std::make_tuple(fixed_shape_v<>, fixed_shape_v<>, fixed_shape_v<>),
            [&](auto last_acc, auto global_dim, auto axis) {
                auto [last_global_offset, last_local_offset, last_shape] =
                    last_acc;
                const auto policy =
                    std::get<axis>(src.sharding().axis_policies);
                if constexpr (distributed::SplitShardPolicy<
                                  std::decay_t<decltype(policy)>>) {
                    // Split axis, simply calculate the global offset and
                    // shape.
                    const auto global_offset =
                        policy.template global_offset<mesh_type>(global_dim,
                                                                 shard_index);
                    const auto local_offset = dim_zero;
                    const auto local_shape =
                        policy.template shard_dim<mesh_type>(global_dim,
                                                             shard_index);
                    return std::make_tuple(
                        last_global_offset.append(global_offset),
                        last_local_offset.append(local_offset),
                        last_shape.append(local_shape));
                } else {
                    return std::make_tuple(last_global_offset.append(dim_zero),
                                           last_local_offset.append(dim_zero),
                                           last_shape.append(dim_zero));
                }
            });

        // 2. Fill non-split axes.
        constexpr auto non_split_mesh_dims = get_non_split_mesh_dims();
        constexpr auto non_split_tensor_axes =
            distributed::detail::tensor_axes_of_non_split_shard_policies<
                sharding_type>();
        const auto split_counts =
            get_non_split_tensor_axes_split_counts(src.shape());
        const auto non_split_shard_index =
            get_non_split_shard_index(shard_index);
        const auto non_split_mesh_linear_offset =
            linear_offset(non_split_shard_index, non_split_mesh_dims);
        const auto local_split_id =
            unravel_index(non_split_mesh_linear_offset, split_counts);
        return non_split_tensor_axes.aggregate(
            split_phase1, [&](auto last_acc, auto axis, auto axis_index) {
                auto [last_global_offset, last_local_offset, last_shape] =
                    last_acc;
                const auto global_dim = src.shape()[axis];
                const auto split_count = split_counts[axis_index];
                const auto local_dim = ntt::ceil_div(global_dim, split_count);
                const auto global_offset =
                    local_dim * local_split_id[axis_index];
                const auto global_end =
                    ntt::min(global_offset + local_dim, global_dim);
                const auto in_bound = global_offset < global_dim;
                return std::make_tuple(
                    last_global_offset.template replace_at<axis>(
                        ntt::where(in_bound, global_offset, dim_zero)),
                    last_local_offset.template replace_at<axis>(
                        ntt::where(in_bound, global_offset, dim_zero)),
                    last_shape.template replace_at<axis>(ntt::where(
                        in_bound, global_end - global_offset, dim_zero)));
            });
    }

    static constexpr auto get_non_split_mesh_dims() noexcept {
        constexpr auto non_split_mesh_axes =
            distributed::detail::mesh_axes_of_non_split_shard_policies<
                sharding_type>();
        return mesh_type::shape.select(non_split_mesh_axes);
    }

    template <distributed::ShardIndex<mesh_type> TShardIndex>
    static constexpr auto
    get_non_split_shard_index(const TShardIndex &shard_index) noexcept {
        constexpr auto non_split_mesh_axes =
            distributed::detail::mesh_axes_of_non_split_shard_policies<
                sharding_type>();
        return shard_index.select(non_split_mesh_axes);
    }

    template <Shape TShape>
    static constexpr auto get_non_split_tensor_axes_split_counts(
        [[maybe_unused]] const TShape &shape) noexcept {
#if defined(__GNUC__) && !defined(__clang__)
        // clang doesn't support constexpr math functions
        if constexpr (FixedShape<TShape>) {
            constexpr auto split_counts =
                get_non_split_tensor_axes_split_counts_impl(TShape{});
            return generate_shape<split_counts.size()>(
                [&](auto axis) { return fixed_dim_v<split_counts.at(axis)>; });
        } else {
#endif
            const auto split_counts =
                get_non_split_tensor_axes_split_counts_impl(shape);
            return generate_shape<split_counts.size()>(
                [&](auto axis) { return split_counts.at(axis); });
#if defined(__GNUC__) && !defined(__clang__)
        }
#endif
    }

    template <Shape TShape>
    static constexpr auto
    get_non_split_tensor_axes_split_counts_impl(const TShape &shape) noexcept {
        constexpr auto non_split_mesh_dims = get_non_split_mesh_dims();
        constexpr auto expected_split_count =
            (std::ptrdiff_t)non_split_mesh_dims.length();

        // Split non-split axes into split_count groups, based on each size of
        // the tensor dimensions.
        constexpr auto non_split_tensor_axes =
            distributed::detail::tensor_axes_of_non_split_shard_policies<
                sharding_type>();
        std::array<dim_t, non_split_tensor_axes.rank()> split_counts{};

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
                auto split_factor =
                    total_log_dim == 0.f
                        ? 0.f
                        : (log_dims[i] / total_log_dim *
                           std::log((float)expected_split_count));
                auto dim = shape.at(non_split_tensor_axes[i]);
                split_counts[i] = std::max(
                    dim_t(1),
                    std::min(dim, static_cast<dim_t>(std::exp(split_factor))));
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
                    for (size_t i = 0; i < split_counts.size(); i++) {
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
                    for (size_t i = 0; i < split_counts.size(); i++) {
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
};

// reshard
template <ShardedTensor SrcTensor, ShardedTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    using mesh_type = typename SrcTensor::mesh_type;
    using src_sharding_type = typename SrcTensor::sharding_type;
    using dest_sharding_type = typename DestTensor::sharding_type;

    static_assert(std::is_same_v<mesh_type, typename DestTensor::mesh_type>,
                  "Cannot reshard between different mesh types.");

    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        if constexpr (std::is_same_v<typename SrcTensor::shape_type,
                                     typename DestTensor::shape_type>) {
            if (src.shape() == dest.shape()) {
                // make sure src ready.
                distributed::topology_synchronize();
                overlap_aware_reshard(src, dest);
                distributed::topology_synchronize();
                return;
            }
        }

        copy_to_global(src);
        copy_from_global(dest);
    }

  private:
    void copy_to_global(const SrcTensor &src) noexcept {
        auto global_buffer_address =
            reinterpret_cast<typename SrcTensor::value_type *>(
                tar::collective_pool_ptr);
        auto global_tensor =
            make_tensor_view_from_address(global_buffer_address, src.shape());
        reshard(src, global_tensor);
    }

    void copy_from_global(DestTensor &dest) noexcept {
        auto global_buffer_address =
            reinterpret_cast<const typename DestTensor::value_type *>(
                tar::collective_pool_ptr);
        const auto global_tensor =
            make_tensor_view_from_address(global_buffer_address, dest.shape());
        reshard(global_tensor, dest);
        distributed::topology_synchronize();
    }

    constexpr void overlap_aware_reshard(const SrcTensor &src,
                                         DestTensor &dest) noexcept {
        auto dest_local_shape = dest.local().shape();
        if (dest_local_shape.length() == 0) {
            return;
        }

        // 1. get dest global offset
        auto global_shape = dest.shape();
        auto local_mesh_index = mesh_type::local_index();
        auto dest_start_offset =
            dest.sharding().global_offset(global_shape, local_mesh_index);
        constexpr auto tensor_rank = SrcTensor::shape_type::rank();

        // 2. get mesh index of src candidates
        // 2.1 generate coords for each split axis
        auto make_coords = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(std::array<size_t, mesh_type::shape[Is]>{}...);
        };
        auto coords = make_coords(std::make_index_sequence<mesh_type::rank()>{});
        std::array<size_t, mesh_type::rank()> counts{};

        auto get_coord = [&]<size_t tensor_axis>() {
            const auto policy = std::get<tensor_axis>(src.sharding().axis_policies);
            if constexpr (distributed::SplitShardPolicy<std::decay_t<decltype(policy)>>) {
                size_t num_blocks = 1;
                constexpr auto policy_rank = policy.axes.rank();
                for (size_t i = 0; i < policy_rank; ++i) {
                    num_blocks *= mesh_type::shape.at(policy.axes.at(i));
                }

                size_t block_size = (global_shape[tensor_axis] + num_blocks - 1) / num_blocks;
                size_t begin_block = dest_start_offset[tensor_axis] / block_size;
                size_t end_block = (dest_start_offset[tensor_axis] + dest_local_shape[tensor_axis] - 1) / block_size;
                for (size_t block = begin_block; block <= end_block; ++block) {
                    size_t remainder = block;
                    auto axes_reverse = policy.axes.reverse();
                    loop<policy_rank>([&](auto idx) {
                        auto mesh_axis = axes_reverse[idx];
                        size_t dim = mesh_type::shape[mesh_axis];
                        size_t c = remainder % dim;
                        remainder /= dim;

                        auto& coord = std::get<mesh_axis>(coords);
                        bool exist = false;
                        for (size_t i = 0; i < counts[mesh_axis]; ++i) {
                            if (coord[i] == c) {
                                exist = true;
                                break;
                            }
                        }

                        if (!exist) {
                            coord[counts[mesh_axis]++] = c;
                        }
                    });
                }
            }
        };

        auto get_all_coords = [&]<size_t... Is>(std::index_sequence<Is...>) {
            (get_coord.template operator()<Is>(), ...);
        };
        get_all_coords(std::make_index_sequence<tensor_rank>{});

        // 2.2 update coords for mesh broadcast axis
        auto update_broadcast_axis = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&] {
                if (counts[Is] == 0) {
                    auto& coord = std::get<Is>(coords);
                    coord[counts[Is]++] = local_mesh_index[Is];
                }
            }()), ...);
        };
        update_broadcast_axis(std::make_index_sequence<mesh_type::rank()>{});

        // 2.3 compute Cartesian product of coords
        std::array<bool, mesh_type::shape.length()> candidates{};
        std::array<size_t, mesh_type::rank()> current_coord{};
        auto compute_cartesian_product = [&](auto&& self, auto axis_const) -> void {
            constexpr size_t axis = decltype(axis_const)::value;
            if constexpr (axis == mesh_type::rank()) {
                dynamic_shape_t<mesh_type::rank()> coord{};
                loop<mesh_type::rank()>([&](auto i) {
                    coord[i] = current_coord[i];
                });
                size_t linear_idx = linear_offset(coord, mesh_type::shape);
                candidates[linear_idx] = true;
                return;
            } else {
                const auto& coord = std::get<axis>(coords);
                for (size_t i = 0; i < counts[axis]; ++i) {
                    current_coord[axis] = coord[i];
                    self(self, std::integral_constant<size_t, axis + 1>{});
                }
            }
        };
        compute_cartesian_product(compute_cartesian_product, std::integral_constant<size_t, 0>{});

        // 3. traverse src index
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i])
                continue;

            const auto shard_index = unravel_index(i, mesh_type::shape);
            auto remote_tensor = src.remote(shard_index);
            slice_with_global_offset<tensor_rank> src_slice{};
            slice_with_global_offset<tensor_rank> dest_slice{};

            // get src slice range
            bool overlap = true;
            auto src_start_offset =
                src.sharding().global_offset(global_shape, shard_index);
            loop<tensor_rank>([&](auto index) {
                // check overlap between src and dest slices
                auto idx = dim_value(index);
                size_t start =
                    std::max(src_start_offset[idx], dest_start_offset[idx]);
                size_t stop =
                    std::min(src_start_offset[idx] + remote_tensor.shape()[idx],
                             dest_start_offset[idx] + dest_local_shape[idx]);
                if (start >= stop) {
                    overlap = false;
                }

                // src_slice.global_offset[index] = start;
                src_slice.local_offset[index] = start - src_start_offset[idx];
                src_slice.shape[index] = stop - start;
                dest_slice.local_offset[index] = start - dest_start_offset[idx];
            });

            // copy overlap
            if (overlap) {
                auto src_block =
                    remote_tensor.view(src_slice.local_offset, src_slice.shape);
                auto dest_block =
                    dest.local().view(dest_slice.local_offset, src_slice.shape);
                tensor_copy_async(src_block, dest_block);
            }
        }

        tensor_copy_wait<void>();
    }
};
} // namespace detail

template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept {
    detail::reshard_impl<std::decay_t<SrcTensor>, std::decay_t<DestTensor>>()(
        src, dest);
}
} // namespace nncase::ntt
