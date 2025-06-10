
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
#include "../detail/shape_storage.h"
#include "../tensor.h"
#include "remote_tensor.h"
#include "sharding.h"
#include <type_traits>

namespace nncase::ntt::distributed {
template <class T, Shape TShape, Sharding TSharding, Shape TLocalShape,
          Strides LocalStrides>
class sharded_tensor_view : public ntt::detail::shape_storage<TShape> {
  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    using sharding_type = TSharding;
    using mesh_type = typename TSharding::mesh_type;
    using shape_type = TShape;
    using shape_storage_type = ntt::detail::shape_storage<TShape>;

    using local_shape_type = TLocalShape;
    using local_tensor_type = tensor_view<T, TLocalShape, LocalStrides>;
    using local_buffer_type = typename local_tensor_type::buffer_type;

    using shape_storage_type::rank;
    using shape_storage_type::shape;
    using shape_storage_type::size;

    constexpr sharded_tensor_view(local_buffer_type local_buffer,
                                  const TShape &shape,
                                  const TSharding &sharding,
                                  const TLocalShape &local_shape,
                                  const LocalStrides &local_strides) noexcept
        : shape_storage_type(shape),
          sharding_(sharding),
          local_(local_buffer, local_shape, local_strides) {}

    constexpr const TSharding &sharding() const noexcept { return sharding_; }

    constexpr local_tensor_type &local() noexcept { return local_; }
    constexpr const local_tensor_type &local() const noexcept { return local_; }

    template <topology RemoteScope, Shape ShardIndex>
    constexpr auto remote(const ShardIndex &shard_index) const noexcept {
        auto local_address = local().elements().data();
        return make_remote_tensor<RemoteScope>(
            local_address, sharding_, shape(), local().strides(), shard_index);
    }

    template <topology RemoteScope, Dimension... ShardIndices>
    constexpr auto remote(const ShardIndices &...shard_index) const noexcept {
        return remote<RemoteScope>(make_shape(shard_index...));
    }

    template <class DestSharding, Shape TDestLocalShape, Strides TDestStrides>
    constexpr void reshard(sharded_tensor_view<T, TShape, DestSharding,
                                               TDestLocalShape, TDestStrides>
                               dest_view) noexcept;

  private:
    NTT_NO_UNIQUE_ADDRESS TSharding sharding_;
    local_tensor_type local_;
};

template <class T = void, class TBuffer, Shape TShape, Sharding TSharding,
          Strides TLocalStrides>
constexpr auto
make_sharded_tensor_view(TBuffer &&buffer, const TShape &shape,
                         const TSharding &sharding,
                         const TLocalStrides &local_strides) noexcept {
    using element_type =
        ntt::detail::tensor_element_type_from_buffer_t<T, TBuffer>;
    using mesh_type = typename TSharding::mesh_type;
    const auto local_index = mesh_type::local_index();
    const auto local_shape = sharding.shard_shape(shape, local_index);
    return sharded_tensor_view<element_type, TShape, TSharding,
                               std::remove_cv_t<decltype(local_shape)>,
                               TLocalStrides>(std::forward<TBuffer>(buffer),
                                              shape, sharding, local_shape,
                                              local_strides);
}

template <class T = void, class TBuffer, Shape TShape, Sharding TSharding,
          Strides TLocalStrides>
constexpr auto make_sharded_tensor_view_from_address(
    T *address, const TShape &shape, const TSharding &sharding,
    const TLocalStrides &local_strides) noexcept {
    using element_type =
        ntt::detail::tensor_element_type_from_buffer_t<T, TBuffer>;
    using mesh_type = typename TSharding::mesh_type;
    const auto local_index = mesh_type::local_index();
    const auto local_shape = sharding.shard_shape(shape, local_index);
    auto buffer = make_span(address, local_shape, local_strides);
    return sharded_tensor_view<T, TShape, TSharding,
                               std::remove_cv_t<decltype(local_shape)>,
                               TLocalStrides>(
        std::move(buffer), shape, sharding, local_shape, local_strides);
}
} // namespace nncase::ntt::distributed
