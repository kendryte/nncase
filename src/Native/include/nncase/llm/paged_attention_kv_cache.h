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
#include "attention_kv_cache.h"
#include "paged_attention_config.h"

namespace nncase::llm {
class paged_attention_kv_cache_node;
using paged_attention_kv_cache = object_t<paged_attention_kv_cache_node>;

class NNCASE_API paged_attention_kv_cache_node
    : public attention_kv_cache_node {
    DEFINE_OBJECT_KIND(attention_kv_cache_node,
                       object_paged_attention_kv_cache);

  public:
    paged_attention_kv_cache_node(paged_attention_config config,
                                  size_t num_seqs, size_t num_tokens,
                                  tensor context_lens, tensor seq_lens,
                                  tensor block_tables, tensor slot_mapping,
                                  tensor kv_caches) noexcept;

    const paged_attention_config &config() const noexcept {
        return attention_kv_cache_node::config()
            .as<paged_attention_config>()
            .unwrap();
    }

    void config(paged_attention_config config) noexcept {
        attention_kv_cache_node::config(std::move(config));
    }

    tensor get_block_ids(int seq_id) const;
    tensor get_slot_ids() const;
    tensor get_block(attention_cache_kind kind, int layer_id, int head_id,
                     const tensor &block_id) const;
    void update_block(attention_cache_kind kind, int layer_id, int head_id,
                      const tensor &block_id, const tensor &block);
    tensor get_slot(attention_cache_kind kind, int layer_id, int head_id,
                    const tensor &slot_id) const;
    void update_slot(attention_cache_kind kind, int layer_id, int head_id,
                     const tensor &slot_id, const tensor &slot);
    void update_slots(attention_cache_kind kind, int layer_id, int head_id,
                      const tensor &slot_ids, const tensor &slots);

    tensor block_tables() const noexcept { return block_tables_; }
    tensor slot_mapping() const noexcept { return slot_mapping_; }
    tensor kv_caches() const noexcept { return kv_caches_; }

  private:
    tensor context_lens_;
    tensor seq_lens_;
    tensor block_tables_;
    tensor slot_mapping_;
    tensor kv_caches_;
};
} // namespace nncase::llm
