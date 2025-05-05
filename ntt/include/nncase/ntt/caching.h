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
#include <cstddef>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/tensor_traits.h>

namespace nncase::ntt::caching {
enum class attention_cache_kind : int {
    key,
    value,
};

enum class paged_attention_dim_kind : int {
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
          IsFixedDims Topology>
struct paged_attention_config
    : public attention_config<NumLayer, NumKVHead, HeadDim, KVPrimType> {
    static inline constexpr int block_size = BlockSize;

    using cache_layout_t = CacheLayout;

    using block_layout_t = BlockLayout;

    using packed_axes_t = PackedAxes;

    using lanes_t = Lanes;

    using kv_topo_t = Topology;

    static inline constexpr cache_layout_t cache_layout = cache_layout_t{};
    static inline constexpr block_layout_t block_layout = block_layout_t{};
    static inline constexpr packed_axes_t packed_axes = packed_axes_t{};
    static inline constexpr lanes_t lanes = lanes_t{};
    static inline constexpr kv_topo_t kv_topo = kv_topo_t{};
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
template <typename TPagedAttentionConfig, size_t Rank> struct kv_type_trait {
    using kv_storage_type_t =
        make_vector_t<typename TPagedAttentionConfig::kv_prim_type,
                      typename TPagedAttentionConfig::lanes_t>;
    using kv_storage_shape_t =
        ntt::ranked_shape<TPagedAttentionConfig::cache_layout_t::rank()>;
    using kv_storage_tensor_type_t =
        tensor_view<kv_storage_type_t, kv_storage_shape_t>;

    using kv_type_t = intptr_t;
    using kv_shape_t =
        decltype(indirect_indexing(typename TPagedAttentionConfig::kv_topo_t{},
                                   typename distributed::topology_shape_t{}));
    using kv_tensor_type_t = tensor<intptr_t, kv_shape_t>;
};

template <typename TPagedAttentionConfig>
struct kv_type_trait<TPagedAttentionConfig, 0> {
    using kv_storage_type_t =
        make_vector_t<typename TPagedAttentionConfig::kv_prim_type,
                      typename TPagedAttentionConfig::lanes_t>;
    using kv_storage_shape_t =
        ntt::ranked_shape<TPagedAttentionConfig::cache_layout_t::rank()>;
    using kv_storage_tensor_type_t =
        tensor_view<kv_storage_type_t, kv_storage_shape_t>;

    using kv_type_t = kv_storage_type_t;
    using kv_shape_t = kv_storage_shape_t;
    using kv_tensor_type_t = kv_storage_tensor_type_t;
};
} // namespace detail

template <typename TConfig>
class paged_attention_kv_cache : public attention_kv_cache<TConfig> {
  public:
    using config_t = TConfig;
    using kv_trait = detail::kv_type_trait<TConfig, TConfig::kv_topo_t::rank()>;
    using kv_storage_type_t = kv_trait::kv_storage_type_t;
    using kv_storage_shape_t = kv_trait::kv_storage_shape_t;
    using kv_storage_tensor_type_t = kv_trait::kv_storage_tensor_type_t;
    using kv_type_t = kv_trait::kv_type_t;
    using kv_shape_t = kv_trait::kv_shape_t;
    using kv_tensor_type_t = kv_trait::kv_tensor_type_t;

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

  private:
    const kv_storage_shape_t get_default_kv_storage_shape() {
        auto cfg = config();
        auto shape = ntt::make_ranked_shape(num_blocks(), cfg.num_layers, 2,
                                            cfg.block_size, cfg.num_kv_heads,
                                            cfg.head_dim);
        for (size_t i = 0; i < (size_t)paged_attention_dim_kind::count__; i++) {
            auto it = cfg.packed_axes.indexof((paged_attention_dim_kind)i);
            if (it != -1) {
                shape[i] /= cfg.lanes[it];
            }
        }
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
        constexpr auto kv_topo = TConfig::kv_topo;
        constexpr size_t topo_rank = kv_topo.rank();
        auto program_ids = distributed::program_ids();
        if constexpr (topo_rank == 0) {
            return kv_caches_;
        } else {
            auto indices = ntt::ranked_shape<topo_rank>();
            for (size_t i = 0; i < topo_rank; i++) {
                auto id = block_id(i);
                // support broadcasting
                indices[i] = id == -1 ? program_ids[kv_topo[i]] : id;
            }

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
    }

    auto get_block_view_from_storage(
        attention_cache_kind kind, int layer_id, int head_id,
        tensor_view<int64_t, ranked_shape<1>> block_id) {
        auto block_id_value = block_id(block_id.size() - 1);

        auto cache_layout = config().cache_layout;
        auto default_starts = ntt::make_ranked_shape(block_id_value, layer_id,
                                                     kind, 0, head_id, 0);
        auto default_shape = get_default_kv_storage_shape();
        default_shape[(size_t)paged_attention_dim_kind::num_blocks] = 1;
        default_shape[(size_t)paged_attention_dim_kind::num_layers] = 1;
        default_shape[(size_t)paged_attention_dim_kind::kv] = 1;
        default_shape[(size_t)paged_attention_dim_kind::num_kv_heads] = 1;

        auto starts = ntt::ranked_shape<default_starts.rank()>();
        auto shape = ntt::ranked_shape<default_shape.rank()>();
        auto squeeze_axes = ntt::ranked_shape<4>();
        for (size_t i = 0, j = 0; i < default_starts.rank(); i++) {
            starts[i] = default_starts[(size_t)cache_layout[i]];
            shape[i] = default_shape[(size_t)cache_layout[i]];
            // printf("[nncase_log] block_start[%ld] = %ld\n", i, starts[i]);
            // printf("[nncase_log] block_shape[%ld] = %ld\n", i, shape[i]);
            if ((cache_layout[i] != paged_attention_dim_kind::block_size) &&
                (cache_layout[i] != paged_attention_dim_kind::head_dim)) {
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
        auto slot_id_value = slot_id(slot_id.size() - 1);
        // printf("[nncase_log] try get slot: [%ld, %ld, %ld]\n", slot_id(0),
        //        slot_id(1), slot_id(2));
        auto block_id_value = slot_id_value / config().block_size;
        auto block_offset_value = slot_id_value % config().block_size;
        auto block_id = tensor<int64_t, ranked_shape<1>>(slot_id.shape());
        std::copy(slot_id.elements().begin(), slot_id.elements().end(),
                  block_id.elements().begin());
        block_id(block_id.size() - 1) = block_id_value;
        auto block_view = get_block_view_from_storage(kind, layer_id, head_id,
                                                      block_id.view());

        auto block_layout = config().block_layout;

        // default_layout = [BlockSize, HeadDim];
        auto default_storage_shape = get_default_kv_storage_shape();
        auto default_layout = ntt::make_ranked_shape(-1, -1, -1, 0, -1, 1);
        auto default_starts = ntt::make_ranked_shape(block_offset_value, 0);
        auto default_shape = ntt::make_ranked_shape(
            1,
            default_storage_shape[(size_t)paged_attention_dim_kind::head_dim]);

        auto starts = ntt::ranked_shape<2>();
        auto shape = ntt::ranked_shape<2>();
        for (size_t i = 0; i < 2; i++) {
            starts[i] = default_starts[default_layout[(size_t)block_layout[i]]];
            shape[i] = default_shape[default_layout[(size_t)block_layout[i]]];
        }
        // printf(
        //     "[nncase_log] get slot view, starts [%ld, %ld], shape [%ld,
        //     %ld]\n", starts[0], starts[1], shape[0], shape[1]);
        return block_view.view(starts, shape)
            .squeeze(ntt::make_ranked_shape(
                block_layout.indexof(paged_attention_dim_kind::block_size)));
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

        auto program_ids = distributed::program_ids();
        if (std::all_of(program_ids.begin(), program_ids.end(),
                        [](auto i) { return i == 0; })) {
            printf("[nncase_log] try_copy_to_destView\n");
        }

        ntt::tensor_copy(slot, destView);
    }

    template <typename T>
    void update_slots(attention_cache_kind kind, int layer_id, int head_id,
                      T slots) {
        // slots : [num_tokens, numHeads, headDim]
        auto slots_shape = slots.shape();
        for (int i = 0; i < this->template num_tokens(); i++) {
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
