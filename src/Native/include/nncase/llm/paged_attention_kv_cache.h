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
#include "nncase/runtime/runtime_op_utility.h"
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
                                  int32_t num_seqs, int32_t num_tokens,
                                  tensor context_lens, tensor seq_lens,
                                  tensor block_table, tensor slot_mapping,
                                  int32_t num_blocks,
                                  const dims_t &kv_shape) noexcept
        : attention_kv_cache_node(config, num_seqs, num_tokens, context_lens,
                                  seq_lens),
          block_table_(std::move(block_table)),
          slot_mapping_(std::move(slot_mapping)),
          num_blocks_(num_blocks),
          kv_shape_(kv_shape),
          kv_strides_(runtime::get_default_strides(kv_shape)),
          kv_storages_(runtime::compute_size(kv_shape)) {}

    paged_attention_config config() const noexcept {
        auto cfg = attention_kv_cache_node::config();
        auto pcfg = cfg.as<paged_attention_config>().expect(
            "paged attention kv cache can't get paged attention config");
        return pcfg;
    }

    int32_t num_blocks() const noexcept { return num_blocks_; }

    void block_table(tensor block_table) noexcept {
        block_table_ = block_table;
    }

    tensor block_table() const noexcept { return block_table_; }

    void slot_mapping(tensor slot_mapping) noexcept {
        slot_mapping_ = slot_mapping;
    }

    tensor slot_mapping() const noexcept { return slot_mapping_; }

    void kv_cache(dims_t indices, tensor kv_storage) noexcept {
        auto index = runtime::linear_offset(indices, kv_strides_);
        kv_storages_[index] = kv_storage;
    }

    tensor kv_cache(dims_t indices) const noexcept {
        auto index = runtime::linear_offset(indices, kv_strides_);
        return kv_storages_[index];
    }

    void kv_cache(int index, tensor kv_storage) noexcept {
        kv_storages_[index] = kv_storage;
    }

    tensor kv_cache(int index) const noexcept { return kv_storages_[index]; }

    const dims_t &kv_shape() const noexcept { return kv_shape_; }

    const auto &kv_storages() const noexcept { return kv_storages_; }

  private:
    tensor seq_lens_;
    tensor block_table_;
    tensor slot_mapping_;
    int32_t num_blocks_;
    dims_t kv_shape_;
    strides_t kv_strides_;
    std::vector<tensor> kv_storages_;
};
} // namespace nncase::llm
