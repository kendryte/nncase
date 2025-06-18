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
#include "nncase/ntt/dimension.h"
#include <cstddef>
#include <numeric>
#include <type_traits>

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
namespace cx {
template <typename T> constexpr T abs(T x) { return x < T{0} ? -x : x; }
// test whether values are within machine epsilon, used for algorithm
// termination
template <typename T> constexpr bool feq(T x, T y) {
    return abs(x - y) <= std::numeric_limits<T>::epsilon();
}

template <typename T> constexpr T exp(T x, T sum, T n, int i, T t) {
    return feq(sum, sum + t / n) ? sum
                                 : exp(x, sum + t / n, n * i, i + 1, t * x);
}

template <typename FloatingPoint>
constexpr FloatingPoint
exp(FloatingPoint x,
    typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type
        * = nullptr) {
    if (std::is_constant_evaluated()) {
        return exp(x, FloatingPoint{1}, FloatingPoint{1}, 2, x);
    } else {
        // Use std::exp for runtime evaluation
        return std::exp(x);
    }
}

//----------------------------------------------------------------------------
// natural logarithm using
// https://en.wikipedia.org/wiki/Natural_logarithm#High_precision
// domain error occurs if x <= 0
template <typename T> constexpr T log_iter(T x, T y) {
    return y + T{2} * (x - cx::exp(y)) / (x + cx::exp(y));
}
template <typename T> constexpr T log(T x, T y) {
    return feq(y, log_iter(x, y)) ? y : log(x, log_iter(x, y));
}
constexpr long double e() { return 2.71828182845904523536l; }

// For numerical stability, constrain the domain to be x > 0.25 && x < 1024
// - multiply/divide as necessary. To achieve the desired recursion depth
// constraint, we need to account for the max double. So we'll divide by
// e^5. If you want to compute a compile-time log of huge or tiny long
// doubles, YMMV.

// if x <= 1, we will multiply by e^5 repeatedly until x > 1
template <typename T> constexpr T logGT(T x) {
    return x > T{0.25} ? log(x, T{0})
                       : logGT<T>(x * e() * e() * e() * e() * e()) - T{5};
}
// if x >= 2e10, we will divide by e^5 repeatedly until x < 2e10
template <typename T> constexpr T logLT(T x) {
    return x < T{1024} ? log(x, T{0})
                       : logLT<T>(x / (e() * e() * e() * e() * e())) + T{5};
}

template <typename FloatingPoint>
constexpr FloatingPoint
log(FloatingPoint x,
    typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type
        * = nullptr) {
    if (std::is_constant_evaluated()) {
        return x >= FloatingPoint{1024} ? logLT(x) : logGT(x);
    } else {
        // Use std::log for runtime evaluation
        return std::log(x);
    }
}
} // namespace cx

template <class SrcTensor, class DestTensor> struct reshard_impl;

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
        tensor_copy(src.view(global_offset, local.shape()), local);
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
            tensor_copy(local, dest.view(global_offset, shape));
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
        if constexpr (FixedShape<TShape>) {
            constexpr auto split_counts =
                get_non_split_tensor_axes_split_counts_impl(TShape{});
            return generate_shape<split_counts.size()>(
                [&](auto axis) { return fixed_dim_v<split_counts.at(axis)>; });
        } else {
            const auto split_counts =
                get_non_split_tensor_axes_split_counts_impl(shape);
            return generate_shape<split_counts.size()>(
                [&](auto axis) { return split_counts.at(axis); });
        }
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
                log_dims[i] = detail::cx::log(dim);
            }

            auto total_log_dim =
                std::accumulate(log_dims.begin(), log_dims.end(), 0.f);
            for (size_t i = 0; i < non_split_tensor_axes.rank(); i++) {
                auto split_factor =
                    total_log_dim == 0.f
                        ? 0.f
                        : (log_dims[i] / total_log_dim *
                           detail::cx::log((float)expected_split_count));
                auto dim = shape.at(non_split_tensor_axes[i]);
                split_counts[i] =
                    std::max(dim_t(1),
                             std::min(dim, static_cast<dim_t>(
                                               detail::cx::exp(split_factor))));
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
    }
};
} // namespace detail

template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept {
    detail::reshard_impl<std::decay_t<SrcTensor>, std::decay_t<DestTensor>>()(
        src, dest);
}
} // namespace nncase::ntt
