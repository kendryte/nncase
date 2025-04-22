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
};
} // namespace nncase::ntt::caching

#if defined(NNCASE_CPU_MODULE) || defined(NNCASE_XPU_MODULE)
#include <attention_config.h>
#elif !defined(NNCASE_NTT_ATTENTION_CONFIG_DEFINED)
namespace nncase::ntt::caching {
struct attention_config {
    static inline constexpr size_t num_layers = 24;
    static inline constexpr size_t num_kv_heads = 12;
    static inline constexpr size_t head_dim = 64;
    using kv_prim_type = float;
};

struct paged_attention_config : public attention_config {
    int num_blocks;

    static inline constexpr int block_size = 16;

    using cache_layout_t =
        ntt::fixed_shape<(size_t)paged_attention_dim_kind::num_blocks,
                         (size_t)paged_attention_dim_kind::num_layers,
                         (size_t)paged_attention_dim_kind::num_kv_heads,
                         (size_t)paged_attention_dim_kind::kv,
                         (size_t)paged_attention_dim_kind::head_dim,
                         (size_t)paged_attention_dim_kind::block_size>;

    using block_layout_t =
        ntt::fixed_shape<(size_t)paged_attention_dim_kind::head_dim,
                         (size_t)paged_attention_dim_kind::block_size>;

    using packed_axes_t =
        ntt::fixed_shape<(size_t)paged_attention_dim_kind::head_dim>;

    using lanes_t = ntt::fixed_shape<32>;

    using kv_topo_t = ntt::fixed_shape<(size_t)distributed::topology::chip,
                                       (size_t)distributed::topology::block>;

    cache_layout_t cache_layout = cache_layout_t{};
    block_layout_t block_layout = block_layout_t{};
    packed_axes_t packed_axes = packed_axes_t{};
    lanes_t lanes = lanes_t{};
    kv_topo_t kv_topo = kv_topo_t{};
};

} // namespace nncase::ntt::caching
#endif

namespace nncase::ntt::caching {
class attention_kv_cache {
  public:
    attention_kv_cache(attention_config config, size_t num_seqs,
                       size_t num_tokens,
                       tensor_view<int64_t, ranked_shape<1>> context_lens,
                       tensor_view<int64_t, ranked_shape<1>> seq_lens)
        : config_(config),
          num_seqs_(num_seqs),
          num_tokens_(num_tokens),
          context_lens_(context_lens),
          seq_lens_(seq_lens) {}

    virtual ~attention_kv_cache() = default;

    const attention_config &config() const noexcept { return config_; }

    size_t num_seqs() const noexcept { return num_seqs_; }

    size_t num_tokens() const noexcept { return num_tokens_; }

    size_t context_len(size_t request_id) const noexcept {
        return (size_t)context_lens_(request_id);
    }

    size_t seq_len(size_t seq_id) const noexcept {
        return (size_t)seq_lens_(seq_id);
    }

  protected:
    attention_config config_;
    size_t num_seqs_;
    size_t num_tokens_;

    tensor_view<int64_t, ranked_shape<1>> context_lens_;
    tensor_view<int64_t, ranked_shape<1>> seq_lens_;
};

template <size_t Rank> struct kv_type_trait {
    using kv_storage_type_t =
        make_vector_t<paged_attention_config::kv_prim_type,
                      paged_attention_config::lanes_t>;
    using kv_storage_shape_t =
        ntt::ranked_shape<paged_attention_config::cache_layout_t::rank()>;
    using kv_storage_tensor_type_t =
        tensor_view<kv_storage_type_t, kv_storage_shape_t>;

    using kv_type_t = intptr_t;
    using kv_shape_t = decltype(indirect_indexing(
        paged_attention_config::kv_topo_t{}, distributed::topology_shape_t{}));
    using kv_tensor_type_t = tensor_view<intptr_t, kv_shape_t>;
};

template <> struct kv_type_trait<0> {
    using kv_storage_type_t =
        make_vector_t<paged_attention_config::kv_prim_type,
                      paged_attention_config::lanes_t>;
    using kv_storage_shape_t =
        ntt::ranked_shape<paged_attention_config::cache_layout_t::rank()>;
    using kv_storage_tensor_type_t =
        tensor_view<kv_storage_type_t, kv_storage_shape_t>;

    using kv_type_t = kv_storage_type_t;
    using kv_shape_t = kv_storage_shape_t;
    using kv_tensor_type_t = kv_storage_tensor_type_t;
};

class paged_attention_kv_cache : public attention_kv_cache {
  public:
    using kv_trait = kv_type_trait<paged_attention_config::kv_topo_t::rank()>;
    using kv_storage_type_t = kv_trait::kv_storage_type_t;
    using kv_storage_shape_t = kv_trait::kv_storage_shape_t;
    using kv_storage_tensor_type_t = kv_trait::kv_storage_tensor_type_t;
    using kv_type_t = kv_trait::kv_type_t;
    using kv_shape_t = kv_trait::kv_shape_t;
    using kv_tensor_type_t = kv_trait::kv_tensor_type_t;

    paged_attention_kv_cache(paged_attention_config config, size_t num_seqs,
                             size_t num_tokens,
                             tensor_view<int64_t, ranked_shape<1>> context_lens,
                             tensor_view<int64_t, ranked_shape<1>> seq_lens,
                             tensor_view<int64_t, ranked_shape<3>> block_table,
                             tensor_view<int64_t, ranked_shape<2>> slot_mapping,
                             kv_tensor_type_t kv_caches)
        : attention_kv_cache(config, num_seqs, num_tokens, context_lens,
                             seq_lens),
          block_table_(block_table),
          slot_mapping_(slot_mapping),
          kv_caches_(kv_caches) {}

    const paged_attention_config &config() const noexcept {
        return static_cast<const paged_attention_config &>(config_);
    }

    tensor_view<int64_t, ranked_shape<2>> get_block_ids(int seq_id) {
        return block_table_
            .view(ntt::make_ranked_shape(seq_id, 0, 0),
                  ntt::make_ranked_shape(1, block_table_.shape()[1],
                                         block_table_.shape()[2]))
            .squeeze(ntt::fixed_shape<0>());
    }

    auto get_slot_ids() { return slot_mapping_; }

  private:
    const kv_storage_shape_t get_default_kv_storage_shape() {
        auto cfg = config();
        auto shape = ntt::make_ranked_shape(cfg.num_blocks, cfg.num_layers, 2,
                                            cfg.block_size, cfg.num_kv_heads,
                                            cfg.head_dim);
        for (size_t i = 0; i < 6; i++) {
            auto it = cfg.packed_axes.indexof(i);
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
        constexpr size_t rank = paged_attention_config::kv_topo_t::rank();
        if constexpr (rank == 0) {
            return kv_caches_;
        } else {
            auto indices = ntt::ranked_shape<rank>();
            for (size_t i = 0; i < rank; i++) {
                indices[i] = block_id(i);
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
            starts[i] = default_starts[config().cache_layout[i]];
            shape[i] = default_shape[config().cache_layout[i]];
            if ((config().cache_layout[i] !=
                 (size_t)paged_attention_dim_kind::block_size) ||
                (config().cache_layout[i] !=
                 (size_t)paged_attention_dim_kind::head_dim)) {
                squeeze_axes[j++] = i;
            }
        }
        return get_kv_storage(block_id)
            .view(starts, shape)
            .squeeze(squeeze_axes);
    }

    auto
    get_slot_view_from_storage(attention_cache_kind kind, int layer_id,
                               int head_id,
                               tensor_view<int64_t, ranked_shape<1>> slot_id) {
        auto slot_id_value = slot_id(slot_id.size() - 1);
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
            starts[i] = default_starts[default_layout[block_layout[i]]];
            shape[i] = default_shape[default_layout[block_layout[i]]];
        }

        return block_view.view(starts, shape)
            .squeeze(ntt::make_ranked_shape(block_layout.indexof(
                (size_t)paged_attention_dim_kind::block_size)));
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

    void
    update_slot(attention_cache_kind kind, int layer_id, int head_id,
                tensor_view<int64_t, ntt::ranked_shape<1>> slot_id,
                tensor_view<kv_storage_type_t, ntt::ranked_shape<1>> slot) {
        auto destView =
            get_slot_view_from_storage(kind, layer_id, head_id, slot_id);
        ntt::tensor_copy(slot, destView);
    }

    void
    update_slots(attention_cache_kind kind, int layer_id, int head_id,
                 tensor_view<int64_t, ntt::ranked_shape<2>> slot_ids,
                 tensor_view<kv_storage_type_t, ntt::ranked_shape<3>> slots) {
        // slots : [num_tokens, numHeads, headDim]
        auto slots_shape = slots.shape();
        for (int i = 0; i < slot_ids.shape()[0]; i++) {
            auto slot_id = slot_ids
                               .view(ntt::make_ranked_shape(i, 0),
                                     ntt::make_ranked_shape(
                                         1, config().kv_topo.rank() + 1))
                               .squeeze(ntt::fixed_shape<0>{});
            auto slot = slots
                            .view(ntt::make_ranked_shape(i, head_id, 0),
                                  ntt::make_ranked_shape(1, 1, slots_shape[2]))
                            .squeeze(ntt::fixed_shape<0, 1>{});
            update_slot(kind, layer_id, head_id, slot_id, slot);
        }
    }

  private:
    tensor_view<int64_t, ranked_shape<3>> block_table_;
    tensor_view<int64_t, ranked_shape<2>> slot_mapping_;
    kv_tensor_type_t kv_caches_;
};
} // namespace nncase::ntt::caching
