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
#include "distributed/sharding.h"
#include "kernels/copy.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor_traits.h"
#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nncase::ntt::caching {
enum class attention_cache_kind : size_t {
    key,
    value,
};

enum class attention_dim_kind : size_t {
    seq = 0,
    head,
    dim,
    count__,
};

enum class paged_kvcache_dim_kind : size_t {
    num_blocks = 0,
    num_layers,
    kv,
    block_size,
    num_kv_heads,
    head_dim,
    count__,
};

template <size_t NumLayer, size_t NumKVHead, size_t HeadDim,
          typename KVPrimType>
struct attention_config {
    using kv_prim_type = KVPrimType;

    static inline constexpr auto num_layers = fixed_dim_v<NumLayer>;
    static inline constexpr auto num_kv_heads = fixed_dim_v<NumKVHead>;
    static inline constexpr auto head_dim = fixed_dim_v<HeadDim>;
    static inline constexpr auto kv = fixed_dim_v<2>;
    static inline constexpr auto kv_prim_size =
        fixed_dim_v<sizeof(kv_prim_type)>;
};

template <size_t NumLayer, size_t NumKVHead, size_t HeadDim,
          typename KVPrimType, size_t BlockSize, FixedDimensions CacheLayout,
          FixedDimensions BlockLayout, FixedDimensions PackedAxes,
          FixedDimensions Lanes, FixedDimensions ShardingAxes,
          class... AxisPolicies>
struct paged_attention_config
    : public attention_config<NumLayer, NumKVHead, HeadDim, KVPrimType> {
    static inline constexpr auto block_size = fixed_dim_v<BlockSize>;
    using cache_layout_t = CacheLayout;
    using block_layout_t = BlockLayout;
    using packed_axes_t = PackedAxes;
    using lanes_t = Lanes;
    using sharding_axes_t = ShardingAxes;
    using axis_policies_t = std::tuple<AxisPolicies...>;

    static inline constexpr auto cache_layout = cache_layout_t{};
    static inline constexpr auto block_layout = block_layout_t{};
    static inline constexpr auto packed_axes = packed_axes_t{};
    static inline constexpr auto lanes = lanes_t{};
    static inline constexpr auto sharding_axes = sharding_axes_t{};
    static inline constexpr auto axis_policies = axis_policies_t{};

    template <paged_kvcache_dim_kind DimKind>
    static constexpr auto axis_policy() noexcept {
        constexpr auto index =
            sharding_axes.index_of(fixed_dim_v<(dim_t)DimKind>);
        if constexpr (index == -1_dim) {
            return distributed::shard_policy::B;
        } else {
            return std::get<index>(axis_policies);
        }
    }
};

template <size_t NumLayer, size_t NumKVHead, size_t HeadDim,
          typename KVPrimType, size_t BlockSize, FixedDimensions CacheLayout,
          FixedDimensions BlockLayout, FixedDimensions PackedAxes,
          FixedDimensions Lanes, FixedDimensions ShardingAxes,
          class... AxisPolicies>
constexpr auto make_paged_attention_config(const CacheLayout &,
                                           const BlockLayout &,
                                           const PackedAxes &, const Lanes &,
                                           const ShardingAxes &,
                                           const AxisPolicies &...) noexcept {
    return paged_attention_config<
        NumLayer, NumKVHead, HeadDim, KVPrimType, BlockSize, CacheLayout,
        BlockLayout, PackedAxes, Lanes, ShardingAxes, AxisPolicies...>{};
}

template <class TConfig> class attention_kv_cache {
  public:
    using context_lens_t = decltype(make_tensor_view_from_address(
        std::declval<const int64_t *>(), std::declval<dynamic_shape_t<1>>()));
    using seq_lens_t = decltype(make_tensor_view_from_address(
        std::declval<const int64_t *>(), std::declval<dynamic_shape_t<1>>()));

    static constexpr TConfig config() noexcept { return TConfig{}; }

    constexpr attention_kv_cache(size_t num_seqs, size_t num_tokens,
                                 context_lens_t context_lens,
                                 seq_lens_t seq_lens) noexcept
        : num_seqs_(num_seqs),
          num_tokens_(num_tokens),
          context_lens_(std::move(context_lens)),
          seq_lens_(std::move(seq_lens)) {}

    size_t num_seqs() const noexcept { return num_seqs_; }
    size_t num_tokens() const noexcept { return num_tokens_; }

    int64_t context_len(int64_t request_id) const noexcept {
        return context_lens_(request_id);
    }

    int64_t seq_len(int64_t seq_id) const noexcept { return seq_lens_(seq_id); }

  protected:
    size_t num_seqs_;
    size_t num_tokens_;

    context_lens_t context_lens_;
    seq_lens_t seq_lens_;
};

namespace detail {
template <class Mesh, size_t... Axes>
constexpr auto
kv_dim(const distributed::shard_policy::split<Axes...> &split) noexcept {
    return split.template divider<Mesh>();
}

template <class Mesh, class... AxisPolicies>
constexpr auto kv_addr_shape(std::tuple<AxisPolicies...>) noexcept {
    return fixed_shape_v<kv_dim<Mesh>(AxisPolicies{})...>;
}

template <typename T, size_t N>
concept HasValidShape =
    requires { std::is_same_v<typename T::shape_type, shape_t<fixed_dim<N>>>; };

template <size_t N, typename T>
concept ValidIdTensor = Tensor<T> && HasValidShape<T, N>;

template <class Mesh, class TConfig>
constexpr auto origin_kv_cache_one_block_shape() noexcept {
    constexpr auto unpacked_shape =
        fixed_shape_v<1 /* one block */, TConfig::num_layers, 2,
                      TConfig::block_size, TConfig::num_kv_heads,
                      TConfig::head_dim>;

    auto packed_shape = TConfig::packed_axes.aggregate(
        unpacked_shape, [&](auto last_shape, auto packed_axis, auto i) {
            return last_shape.template replace_at<packed_axis>(
                last_shape[packed_axis] / TConfig::lanes[i]);
        });

    auto shard_shape = TConfig::sharding_axes.aggregate(
        packed_shape, [&](auto last_shape, auto sharding_axis, auto i) {
            using axis_policy_t =
                std::tuple_element_t<i, typename TConfig::axis_policies_t>;
            auto dim =
                axis_policy_t::template try_shard_dim_without_shard_index<Mesh>(
                    last_shape[sharding_axis]);
            static_assert(dim != -1_dim,
                          "Only uniform shard dim is supported.");
            return last_shape.template replace_at<sharding_axis>(dim);
        });
    return shard_shape;
}

template <class TConfig, Dimensions TDims>
constexpr auto transform_kv_cache_dims(const TDims &dims) noexcept {
    constexpr auto cache_layout = TConfig::cache_layout;
    return dims.select(cache_layout);
}

template <class TConfig, Dimensions TDims>
constexpr auto transform_block_dims(const TDims &dims) noexcept {
    constexpr auto default_layout = fixed_shape_v<-1, -1, -1, 0, -1, 1>;
    constexpr auto block_layout = TConfig::block_layout;
    return generate_shape<TDims::rank()>(
        [&](auto axis) { return dims[default_layout[block_layout[axis]]]; });
}
} // namespace detail

template <class Mesh, class TConfig>
class paged_attention_kv_cache : public attention_kv_cache<TConfig> {
  public:
    using config_t = TConfig;
    using typename attention_kv_cache<TConfig>::context_lens_t;
    using typename attention_kv_cache<TConfig>::seq_lens_t;

    static constexpr auto id_length = TConfig::sharding_axes_t::rank() + 1_dim;
    static constexpr auto id_shape = fixed_shape_v<id_length>;

    // [seq_len, context_len / block_size, id_length]
    using block_table_shape_t = shape_t<dim_t, dim_t, fixed_dim<id_length>>;
    using block_table_t = decltype(make_tensor_view_from_address(
        std::declval<const int64_t *>(), std::declval<block_table_shape_t>()));

    // [num_tokens, id_length]
    using slot_mapping_shape_t = shape_t<dim_t, fixed_dim<id_length>>;
    using slot_mapping_t = decltype(make_tensor_view_from_address(
        std::declval<const int64_t *>(), std::declval<slot_mapping_shape_t>()));

    using kv_storage_type_t =
        basic_vector<typename TConfig::kv_prim_type, typename TConfig::lanes_t>;

    static constexpr auto origin_kv_cache_one_block_shape =
        detail::origin_kv_cache_one_block_shape<Mesh, TConfig>();
    static constexpr auto kv_cache_one_block_shape =
        detail::transform_kv_cache_dims<TConfig>(
            origin_kv_cache_one_block_shape);
    static constexpr auto kv_cache_block_length =
        kv_cache_one_block_shape.length();

    static constexpr auto kv_addrs_shape =
        detail::kv_addr_shape<Mesh>(TConfig::axis_policies);
    using kv_addrs_t = decltype(make_tensor_view_from_address(
        std::declval<kv_storage_type_t **>(), kv_addrs_shape));

    paged_attention_kv_cache(size_t num_seqs, size_t num_tokens,
                             context_lens_t context_lens, seq_lens_t seq_lens,
                             block_table_t block_table,
                             slot_mapping_t slot_mapping, kv_addrs_t kv_addrs)
        : attention_kv_cache<TConfig>(num_seqs, num_tokens, context_lens,
                                      seq_lens),
          block_table_(block_table),
          slot_mapping_(slot_mapping),
          kv_addrs_(kv_addrs) {}

    constexpr TConfig config() const noexcept { return TConfig{}; }

    template <class TId>
    constexpr auto kv_cache_address(const TId &shard_index) noexcept
        requires(TId::rank() == id_length - 1)
    {
        return kv_addrs_(shard_index);
    }

    constexpr auto get_block_id(int64_t seq_id,
                                int64_t context_id) const noexcept {
        return block_table_.view(make_shape(seq_id, context_id));
    }

    constexpr auto get_slot_id(int64_t token_id) const noexcept {
        return slot_mapping_.view(make_shape(token_id));
    }

    template <attention_cache_kind Kind, class T>
    constexpr auto get_block(dim_t layer_id, dim_t head_id,
                             const T &block_id) noexcept
        requires detail::ValidIdTensor<id_length, T>
    {
        return get_block_view<Kind>(layer_id, head_id, block_id);
    }

    template <attention_cache_kind Kind, typename T>
    constexpr auto get_slot(int layer_id, int head_id,
                            const T &slot_id) noexcept
        requires detail::ValidIdTensor<id_length, T>
    {
        return get_slot_view<Kind>(layer_id, head_id, slot_id);
    }

    template <attention_cache_kind Kind, typename T, typename TId>
    constexpr void update_slot(dim_t layer_id, dim_t head_id,
                               const TId &slot_id, const T &slot) noexcept
        requires detail::ValidIdTensor<id_length, TId>
    {
        auto destView = get_slot_view<Kind>(layer_id, head_id, slot_id);
        ntt::tensor_copy(slot, destView);
    }

    template <attention_cache_kind Kind, typename T>
    constexpr void update_slots(dim_t layer_id, dim_t head_id,
                                const T &slots) noexcept {
        // slots : [num_tokens, numHeads, headDim]
        auto slots_shape = slots.shape();
        for (dim_t i = 0; i < this->num_tokens(); i++) {
            auto slot_id = get_slot_id(i);
            auto slot = slots.view(ntt::make_shape(i, head_id));
            update_slot<Kind>(layer_id, head_id, slot_id, slot);
        }
    }

  private:
    template <class T>
    constexpr auto get_kv_cache_one_block_view(const T &block_id) noexcept
        requires detail::ValidIdTensor<id_length, T>
    {
        const auto block_shard_index =
            block_id.view(fixed_shape_v<0>, fixed_shape_v<id_length - 1_dim>);
        const auto block_offset = block_id(-1_dim);
        const auto local_index = Mesh::local_index();
        const auto kv_cache_index =
            generate_shape<block_shard_index.size()>([&](auto axis) {
                const auto index = block_shard_index(axis);
                const auto mesh_axis =
                    std::get<axis>(TConfig::axis_policies).axes.front();
                return ntt::where(index == -1_dim, local_index[mesh_axis],
                                  index);
            });
        auto address = kv_cache_address(kv_cache_index) +
                       block_offset * kv_cache_block_length;
        return make_tensor_view_from_address(address, kv_cache_one_block_shape);
    }

    template <attention_cache_kind Kind, class T>
    constexpr auto get_block_view(dim_t layer_id, dim_t head_id,
                                  const T &block_id) noexcept
        requires detail::ValidIdTensor<id_length, T>
    {
        auto cache_layout = config().cache_layout;
        const auto origin_starts =
            ntt::make_shape(dim_zero, layer_id, fixed_dim_v<(dim_t)Kind>,
                            dim_zero, head_id, dim_zero);
        const auto origin_shape =
            origin_kv_cache_one_block_shape
                .template replace_at<(
                    size_t)paged_kvcache_dim_kind::num_layers>(1_dim)
                .template replace_at<(size_t)paged_kvcache_dim_kind::kv>(1_dim)
                .template replace_at<(
                    size_t)paged_kvcache_dim_kind::num_kv_heads>(1_dim);

        const auto starts =
            detail::transform_kv_cache_dims<TConfig>(origin_starts);
        const auto shape =
            detail::transform_kv_cache_dims<TConfig>(origin_shape);
        const auto squeeze_axes = cache_layout.aggregate(
            fixed_shape_v<>, [&](auto last_axes, auto kv_cache_dim, auto i) {
                if constexpr (kv_cache_dim !=
                                  (size_t)paged_kvcache_dim_kind::block_size &&
                              kv_cache_dim !=
                                  (size_t)paged_kvcache_dim_kind::head_dim) {
                    return last_axes.append(i);
                } else {
                    return last_axes;
                }
            });
        auto block_view = get_kv_cache_one_block_view(block_id);
        // printf("[nncase_log] default_shape [%ld, %ld, %ld, %ld, %ld,
        // %ld]\n",
        //        default_shape[0], default_shape[1], default_shape[2],
        //        default_shape[3], default_shape[4], default_shape[5]);
        // printf("[nncase_log] try get block kv_storage view starts: [%ld,
        // %ld,
        // "
        //        "%ld, %ld, %ld, %ld], shape [%ld, %ld, %ld, %ld, %ld,
        //        %ld]\n", starts[0], starts[1], starts[2], starts[3],
        //        starts[4], starts[5], shape[0], shape[1], shape[2],
        //        shape[3], shape[4], shape[5]);
        return block_view.view(starts, shape).squeeze(squeeze_axes);
    }

    template <attention_cache_kind Kind, class T>
    constexpr auto get_slot_view(dim_t layer_id, dim_t head_id,
                                 const T &slot_id) noexcept
        requires detail::ValidIdTensor<id_length, T>
    {
        const auto global_slot_offset = slot_id(-1_dim);
        const auto block_offset = global_slot_offset / TConfig::block_size;
        const auto local_slot_offset = global_slot_offset % TConfig::block_size;
        auto block_id = make_tensor<int64_t>(id_shape);
        tensor_copy(slot_id, block_id);
        block_id(-1_dim) = block_offset;
        // printf("[nncase_log] try get slot: [%ld, %ld, %ld]\n",
        // slot_id(0),
        //        slot_id(1), slot_id(2));

        // block_view = [BlockSize, HeadDim];
        auto block_view = get_block_view<Kind>(layer_id, head_id, block_id);
        const auto origin_starts = make_shape(local_slot_offset, dim_zero);
        const auto origin_shape =
            make_shape(dim_one, origin_kv_cache_one_block_shape[fixed_dim_v<(
                                    dim_t)paged_kvcache_dim_kind::head_dim>]);
        const auto starts =
            detail::transform_block_dims<TConfig>(origin_starts);
        const auto shape = detail::transform_block_dims<TConfig>(origin_shape);
        const auto squeeze_axes = fixed_shape_v<TConfig::block_layout.index_of(
            fixed_dim_v<(dim_t)paged_kvcache_dim_kind::block_size>)>;
        return block_view.view(starts, shape).squeeze(squeeze_axes);
    }

  private:
    block_table_t block_table_;
    slot_mapping_t slot_mapping_;
    kv_addrs_t kv_addrs_;
};
} // namespace nncase::ntt::caching
