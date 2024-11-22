
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
#include "remote_tensor.h"
#include "sharding.h"
#include "tensor.h"

namespace nncase::ntt::distributed {
template <
    class T, class GlobalShape, class Sharding,
    class LocalStrides = default_strides_t<
        typename detail::local_shard_shape_type<GlobalShape, Sharding>::type>>
class sharded_tensor_view
    : private tensor_view<
          T,
          typename detail::local_shard_shape_type<GlobalShape, Sharding>::type,
          LocalStrides> {
  public:
    using sharding_type = Sharding;
    using mesh_type = typename sharding_type::mesh_type;

    using local_shape_type =
        typename detail::local_shard_shape_type<GlobalShape, Sharding>::type;
    using local_tensor_type = tensor_view<T, local_shape_type, LocalStrides>;
    using remote_tensor_type =
        remote_tensor_view<T, local_shape_type, LocalStrides>;

    using local_tensor_type::local_tensor_type;

    local_tensor_type &local() noexcept {
        return static_cast<local_tensor_type &>(*this);
    }

    const local_tensor_type &local() const noexcept {
        return static_cast<const local_tensor_type &>(*this);
    }

    template <class... ShardIndexes>
    remote_tensor_type remote(ShardIndexes &&...shardIndexes) const noexcept {
        static_assert(sizeof...(shardIndexes) == mesh_type::shape_type::rank(),
                      "Invalid index.");
        auto local_address = local().elements().data();
        return remote_tensor_type::create(
            mesh_type::remote_program_id(
                ranked_shape<mesh_type::shape_type::rank()>(
                    std::forward<ShardIndexes>(shardIndexes)...)),
            local_address);
    }

    template <class DestSharding, class DestStrides>
    void reshard(sharded_tensor_view<T, GlobalShape, DestSharding, DestStrides>
                     dest_view) noexcept;
};
} // namespace nncase::ntt::distributed
