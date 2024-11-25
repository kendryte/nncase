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
#include <type_traits>

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
    }
};
} // namespace detail

template <class SrcTensor, class DestTensor>
constexpr void reshard(const SrcTensor &src, DestTensor &&dest) noexcept {
    detail::reshard_impl<std::decay_t<SrcTensor>, std::decay_t<DestTensor>>()(
        src, dest);
}
} // namespace nncase::ntt
