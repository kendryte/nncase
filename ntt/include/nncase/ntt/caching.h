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
#include "kernels/copy.h"
#include "nncase/ntt/sharding.h"
#include <cstddef>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/tensor_traits.h>

namespace nncase::ntt::caching {
enum class attention_cache_kind : int {
    key,
    value,
};

enum class attention_dim_kind : int {
    seq = 0,
    head,
    dim,
    count__,
};

enum class paged_kvcache_dim_kind : int {
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
    static inline constexpr size_t num_layers = NumLayer;
    static inline constexpr size_t num_kv_heads = NumKVHead;
    static inline constexpr size_t head_dim = HeadDim;
    static inline constexpr size_t kv = 2;
    using kv_prim_type = KVPrimType;
    static inline constexpr size_t kv_prim_size = sizeof(kv_prim_type);
};

template <size_t NumLayer, size_t NumKVHead, size_t HeadDim,
          typename KVPrimType, size_t BlockSize, IsFixedDims CacheLayout,
          IsFixedDims BlockLayout, IsFixedDims PackedAxes, IsFixedDims Lanes,
          IsFixedDims ShardingAxes, class... AxisPolicies>
struct paged_attention_config
    : public attention_config<NumLayer, NumKVHead, HeadDim, KVPrimType> {
    static inline constexpr int block_size = BlockSize;

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
};

template <class TConfig> class attention_kv_cache {
  public:
    attention_kv_cache(size_t num_seqs, size_t num_tokens,
                       tensor_view<int64_t, ranked_shape<1>> context_lens,
                       tensor_view<int64_t, ranked_shape<1>> seq_lens)
        : num_seqs_(num_seqs),
          num_tokens_(num_tokens),
          context_lens_(context_lens),
          seq_lens_(seq_lens) {}

    virtual ~attention_kv_cache() = default;

    constexpr TConfig config() const noexcept { return TConfig{}; }

    size_t num_seqs() const noexcept { return num_seqs_; }

    size_t num_tokens() const noexcept { return num_tokens_; }

    size_t context_len(size_t request_id) const noexcept {
        return (size_t)context_lens_(request_id);
    }

    size_t seq_len(size_t seq_id) const noexcept {
        return (size_t)seq_lens_(seq_id);
    }

  protected:
    size_t num_seqs_;
    size_t num_tokens_;

    tensor_view<int64_t, ranked_shape<1>> context_lens_;
    tensor_view<int64_t, ranked_shape<1>> seq_lens_;
};

namespace detail {
template <class Mesh, size_t... Axes>
constexpr auto calc_kv_dim(distributed::shard_policy::S<Axes...>) {
    return (1 * ... * Mesh::shape_type::at(Axes));
}

template <class Mesh, class... AxisPolicies>
constexpr auto calc_kv_shape(std::tuple<AxisPolicies...>) {
    return fixed_shape<calc_kv_dim<Mesh>(AxisPolicies{})...>{};
}

} // namespace detail

template <class Mesh, class TConfig>
class paged_attention_kv_cache : public attention_kv_cache<TConfig> {
  public:
    using config_t = TConfig;

    using kv_storage_type_t = make_vector_t<typename TConfig::kv_prim_type,
                                            typename TConfig::lanes_t>;
    using kv_storage_shape_t =
        ntt::ranked_shape<TConfig::cache_layout_t::rank()>;
    using kv_storage_tensor_type_t =
        tensor_view<kv_storage_type_t, kv_storage_shape_t>;

    using kv_type_t = intptr_t;
    using kv_shape_t = decltype(detail::calc_kv_shape<Mesh>(
        typename TConfig::axis_policies_t{}));
    using kv_tensor_type_t = tensor<intptr_t, kv_shape_t>;

    paged_attention_kv_cache(size_t num_seqs, size_t num_tokens,
                             tensor_view<int64_t, ranked_shape<1>> context_lens,
                             tensor_view<int64_t, ranked_shape<1>> seq_lens,
                             tensor_view<int64_t, ranked_shape<3>> block_table,
                             tensor_view<int64_t, ranked_shape<2>> slot_mapping,
                             size_t num_blocks, kv_tensor_type_t kv_caches)
        : attention_kv_cache<TConfig>(num_seqs, num_tokens, context_lens,
                                      seq_lens),
          block_table_(block_table),
          slot_mapping_(slot_mapping),
          num_blocks_(num_blocks),
          kv_caches_(kv_caches) {}

    constexpr TConfig config() const noexcept { return TConfig{}; }

    tensor_view<int64_t, ranked_shape<1>> get_block_id(int seq_id,
                                                       int context_id) {
        return block_table_
            .view(ntt::make_ranked_shape(seq_id, context_id, 0),
                  ntt::make_ranked_shape(1, 1, block_table_.shape()[2]))
            .squeeze(ntt::fixed_shape<0, 1>());
    }

    tensor_view<int64_t, ranked_shape<1>> get_slot_id(int token_id) {
        return slot_mapping_
            .view(ntt::make_ranked_shape(token_id, 0),
                  ntt::make_ranked_shape(1, slot_mapping_.shape()[1]))
            .squeeze(ntt::fixed_shape<0>());
    }

    size_t num_blocks() const noexcept { return num_blocks_; }

    auto get_kv_storage_tensor(
        const ranked_shape<TConfig::sharding_axes_t::rank()> &indices) {
        auto storage_ptr = kv_caches_(indices);
        auto storage_shape = get_kv_storage_shape();
        auto storage_strides = default_strides(storage_shape);
        auto storage_size = linear_size(storage_shape, storage_strides);
        return kv_storage_tensor_type_t(
            std::span<kv_storage_type_t>(
                reinterpret_cast<kv_storage_type_t *>(storage_ptr),
                storage_size),
            storage_shape, storage_strides);
    }

  private:
    const kv_storage_shape_t get_default_kv_storage_shape() {
        auto cfg = config();
        auto shape = ntt::make_ranked_shape(num_blocks(), cfg.num_layers, 2,
                                            cfg.block_size, cfg.num_kv_heads,
                                            cfg.head_dim);
        // pack
        loop<TConfig::packed_axes_t::rank()>([&](auto i) {
            constexpr auto axis = TConfig::packed_axes_t::at(i);
            shape[axis] /= TConfig::lanes_t::at(i);
        });

        loop<TConfig::sharding_axes_t::rank()>([&](auto i) {
            constexpr auto axis = TConfig::sharding_axes_t::at(i);
            shape[axis] =
                std::tuple_element_t<i, typename TConfig::axis_policies_t>::
                    template local_dim<Mesh>(shape[axis]);
        });

        return shape;
    }

    const kv_storage_shape_t get_kv_storage_shape() {
        auto default_shape = get_default_kv_storage_shape();
        auto shape = ntt::ranked_shape<6>();
        for (size_t i = 0; i < 6; i++) {
            shape[i] = default_shape[config().cache_layout[i]];
        }
        return shape;
    }

    auto get_kv_storage(tensor_view<int64_t, ranked_shape<1>> block_id) {
        constexpr auto sharding_axes = TConfig::sharding_axes;
        constexpr size_t sharding_rank = sharding_axes.rank();

        auto program_ids = distributed::program_ids();
        auto indices = ntt::ranked_shape<sharding_rank>();
        loop<sharding_rank>([&](auto shard_id) {
            auto index = block_id(shard_id);
            // todo need process partial sharding.
            constexpr auto program_id_index = std::tuple_element_t<
                shard_id, typename TConfig::axis_policies_t>::axes_type::at(0);
            indices[shard_id] = index == -1 ? program_ids[program_id_index] : index;
        });

        auto storage_ptr = kv_caches_(indices);
        auto storage_shape = get_kv_storage_shape();
        auto storage_strides = default_strides(storage_shape);
        auto storage_size = linear_size(storage_shape, storage_strides);
        return kv_storage_tensor_type_t(
            std::span<kv_storage_type_t>(
                reinterpret_cast<kv_storage_type_t *>(storage_ptr),
                storage_size),
            storage_shape, storage_strides);
    }

    auto get_block_view_from_storage(
        attention_cache_kind kind, int layer_id, int head_id,
        tensor_view<int64_t, ranked_shape<1>> block_id) {
        auto block_id_value = block_id(block_id.shape().last() - 1);

        auto cache_layout = config().cache_layout;
        auto default_starts = ntt::make_ranked_shape(block_id_value, layer_id,
                                                     kind, 0, head_id, 0);
        auto default_shape = get_default_kv_storage_shape();
        default_shape[(size_t)paged_kvcache_dim_kind::num_blocks] = 1;
        default_shape[(size_t)paged_kvcache_dim_kind::num_layers] = 1;
        default_shape[(size_t)paged_kvcache_dim_kind::kv] = 1;
        default_shape[(size_t)paged_kvcache_dim_kind::num_kv_heads] = 1;

        auto starts = ntt::ranked_shape<default_starts.rank()>();
        auto shape = ntt::ranked_shape<default_shape.rank()>();
        auto squeeze_axes = ntt::ranked_shape<4>();
        for (size_t i = 0, j = 0; i < default_starts.rank(); i++) {
            starts[i] = default_starts[(size_t)cache_layout[i]];
            shape[i] = default_shape[(size_t)cache_layout[i]];
            // printf("[nncase_log] block_start[%ld] = %ld\n", i, starts[i]);
            // printf("[nncase_log] block_shape[%ld] = %ld\n", i, shape[i]);
            if ((cache_layout[i] !=
                 (size_t)paged_kvcache_dim_kind::block_size) &&
                (cache_layout[i] != (size_t)paged_kvcache_dim_kind::head_dim)) {
                squeeze_axes[j] = i;
                // printf("[nncase_log] block_squeeze[%ld] = %ld\n", j,
                //        squeeze_axes[j]);
                j++;
            }
        }
        auto kv_storage = get_kv_storage(block_id);
        // printf("[nncase_log] default_shape [%ld, %ld, %ld, %ld, %ld, %ld]\n",
        //        default_shape[0], default_shape[1], default_shape[2],
        //        default_shape[3], default_shape[4], default_shape[5]);
        // printf("[nncase_log] try get block kv_storage view starts: [%ld, %ld,
        // "
        //        "%ld, %ld, %ld, %ld], shape [%ld, %ld, %ld, %ld, %ld, %ld]\n",
        //        starts[0], starts[1], starts[2], starts[3], starts[4],
        //        starts[5], shape[0], shape[1], shape[2], shape[3], shape[4],
        //        shape[5]);
        return kv_storage.view(starts, shape).squeeze(squeeze_axes);
    }

    auto
    get_slot_view_from_storage(attention_cache_kind kind, int layer_id,
                               int head_id,
                               tensor_view<int64_t, ranked_shape<1>> slot_id) {
        auto slot_id_value = slot_id(slot_id.shape().last() - 1);
        // printf("[nncase_log] try get slot: [%ld, %ld, %ld]\n", slot_id(0),
        //        slot_id(1), slot_id(2));
        auto block_id_value = slot_id_value / config().block_size;
        auto block_offset_value = slot_id_value % config().block_size;
        auto block_id = tensor<int64_t, ranked_shape<1>>(slot_id.shape());
        std::copy(slot_id.elements().begin(), slot_id.elements().end(),
                  block_id.elements().begin());
        block_id(block_id.shape().last() - 1) = block_id_value;
        auto block_view = get_block_view_from_storage(kind, layer_id, head_id,
                                                      block_id.view());

        auto block_layout = config().block_layout;

        // default_layout = [BlockSize, HeadDim];
        auto default_storage_shape = get_default_kv_storage_shape();
        auto default_layout = ntt::make_ranked_shape(-1, -1, -1, 0, -1, 1);
        auto default_starts = ntt::make_ranked_shape(block_offset_value, 0);
        auto default_shape = ntt::make_ranked_shape(
            1, default_storage_shape[(size_t)paged_kvcache_dim_kind::head_dim]);

        auto starts = ntt::ranked_shape<2>();
        auto shape = ntt::ranked_shape<2>();
        for (size_t i = 0; i < 2; i++) {
            starts[i] = default_starts[default_layout[(size_t)block_layout[i]]];
            shape[i] = default_shape[default_layout[(size_t)block_layout[i]]];
        }

        return block_view.view(starts, shape)
            .squeeze(ntt::make_ranked_shape(block_layout.indexof(
                (size_t)paged_kvcache_dim_kind::block_size)));
    }

  public:
    auto get_block(attention_cache_kind kind, int layer_id, int head_id,
                   tensor_view<int64_t, ranked_shape<1>> block_id) {
        return get_block_view_from_storage(kind, layer_id, head_id, block_id);
    }

    void
    update_block(attention_cache_kind kind, int layer_id, int head_id,
                 tensor_view<int64_t, ranked_shape<1>> block_id,
                 tensor_view<kv_storage_type_t, ntt::ranked_shape<2>> block) {
        ntt::tensor_copy(block, get_block_view_from_storage(kind, layer_id,
                                                            head_id, block_id));
    }

    auto get_slot(attention_cache_kind kind, int layer_id, int head_id,
                  tensor_view<int64_t, ntt::ranked_shape<1>> slot_id) {
        return get_slot_view_from_storage(kind, layer_id, head_id, slot_id);
    }

    template <typename T>
    void update_slot(attention_cache_kind kind, int layer_id, int head_id,
                     tensor_view<int64_t, ntt::ranked_shape<1>> slot_id,
                     T slot) {
        auto destView =
            get_slot_view_from_storage(kind, layer_id, head_id, slot_id);
        ntt::tensor_copy(slot, destView);
    }

    template <typename T>
    void update_slots(attention_cache_kind kind, int layer_id, int head_id,
                      T slots) {
        // slots : [num_tokens, numHeads, headDim]
        auto slots_shape = slots.shape();
        for (int i = 0; i < this->num_tokens(); i++) {
            auto slot_id = get_slot_id(i);
            auto slot = slots
                            .view(ntt::make_ranked_shape(i, head_id, 0),
                                  ntt::make_ranked_shape(1, 1, slots_shape[2]))
                            .squeeze(ntt::fixed_shape<0, 1>{});
            update_slot(kind, layer_id, head_id, slot_id, slot);
        }
    }

    auto block_table() const noexcept { return block_table_; }

    auto &kv_caches() const noexcept { return kv_caches_; }

  private:
    tensor_view<int64_t, ranked_shape<3>> block_table_;
    tensor_view<int64_t, ranked_shape<2>> slot_mapping_;
    size_t num_blocks_;
    kv_tensor_type_t kv_caches_;
};
} // namespace nncase::ntt::caching
