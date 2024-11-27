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
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include <type_traits>

namespace tar {
extern uint8_t collective_pool_ptr[];
}

namespace nncase::ntt {
namespace detail {
template <class SrcTensor, class DestTensor> struct reshard_impl;

// shard
template <IsTensor SrcTensor, IsShardedTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        using mesh_type = typename DestTensor::mesh_type;
        using sharding_type = typename DestTensor::sharding_type;
        auto local_mesh_index = mesh_type::local_index();
        auto global_offset =
            sharding_type::global_offset(dest.global_shape(), local_mesh_index);
        auto local = dest.local();
        tensor_copy(src.view(global_offset, local.shape()), local);
    }
};

// unshard
template <IsShardedTensor SrcTensor, IsTensor DestTensor>
struct reshard_impl<SrcTensor, DestTensor> {
    constexpr void operator()(const SrcTensor &src, DestTensor &dest) noexcept {
        using mesh_type = typename SrcTensor::mesh_type;
        using sharding_type = typename SrcTensor::sharding_type;
        auto local_mesh_index = mesh_type::local_index();
        auto global_offset =
            sharding_type::global_offset(src.global_shape(), local_mesh_index);
        auto local = src.local();
        tensor_copy(local, dest.view(global_offset, local.shape()));
        distributed::topology_synchronize();
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
        auto local_mesh_index = mesh_type::local_index();
        auto global_offset = src_sharding_type::global_offset(
            src.global_shape(), local_mesh_index);
        auto local = src.local();

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
        tensor_copy(local, global_tensor.view(global_offset, local.shape()));
        distributed::topology_synchronize();
    }

    void copy_from_global(DestTensor &dest) noexcept {
        auto local_mesh_index = mesh_type::local_index();
        auto global_offset = dest_sharding_type::global_offset(
            dest.global_shape(), local_mesh_index);
        auto local = dest.local();

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
        tensor_copy(global_tensor.view(global_offset, local.shape()), local);
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
