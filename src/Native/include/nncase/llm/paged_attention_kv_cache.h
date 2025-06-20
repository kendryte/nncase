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

namespace nncase::llm {
class paged_attention_kv_cache_node;
using paged_attention_kv_cache = object_t<paged_attention_kv_cache_node>;

class NNCASE_API paged_attention_kv_cache_node
    : public attention_kv_cache_node {
    DEFINE_OBJECT_KIND(attention_kv_cache_node,
                       object_paged_attention_kv_cache);

  public:
    paged_attention_kv_cache_node() noexcept = default;
    paged_attention_kv_cache_node(int32_t num_seqs, int32_t num_tokens,
                                  tensor context_lens, tensor seq_lens,
                                  tensor block_tables, tensor slot_mapping,
                                  std::vector<tensor> kv_caches) noexcept
        : attention_kv_cache_node(num_seqs, num_tokens, context_lens, seq_lens),
          block_tables_(std::move(block_tables)),
          slot_mapping_(std::move(slot_mapping)),
          kv_caches_(std::move(kv_caches)) {}

    void block_tables(tensor block_tables) noexcept {
        block_tables_ = block_tables;
    }

    tensor block_tables() const noexcept { return block_tables_; }

    void slot_mapping(tensor slot_mapping) noexcept {
        slot_mapping_ = slot_mapping;
    }

    tensor slot_mapping() const noexcept { return slot_mapping_; }

    const std::vector<tensor> &kv_caches() const noexcept { return kv_caches_; }
    void kv_caches(std::vector<tensor> kv_caches) noexcept {
        kv_caches_ = std::move(kv_caches);
    }

  private:
    tensor seq_lens_;
    tensor block_tables_;
    tensor slot_mapping_;
    std::vector<tensor> kv_caches_;
};
} // namespace nncase::llm
