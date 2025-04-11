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

namespace nncase {
class paged_attention_kv_cache_node;
using paged_attention_kv_cache = object_t<paged_attention_kv_cache_node>;

class NNCASE_API paged_attention_kv_cache_node
    : public attention_kv_cache_node {
    DEFINE_OBJECT_KIND(attention_kv_cache_node,
                       object_paged_attention_kv_cache);

  public:
    paged_attention_kv_cache_node(paged_attention_config config,
                                  size_t num_request, tensor context_lens,
                                  tensor seq_lens, tensor block_tables,
                                  tensor slot_mapping) noexcept
        : attention_kv_cache_node(std::move(config), num_request,
                                  std::move(context_lens), std::move(seq_lens)),
          block_tables_(std::move(block_tables)),
          slot_mapping_(std::move(slot_mapping)) {};

    /**@brief Gets attention config. */
    const paged_attention_config &config() const noexcept {
        return attention_kv_cache_node::config()
            .as<paged_attention_config>()
            .unwrap();
    }

    /**@brief Sets attention config.
     * @param config Attention config.
     */
    void set_config(paged_attention_config config) noexcept {
        attention_kv_cache_node::set_config(std::move(config));
    }

    /**@brief Gets block tables. */
    const tensor &block_tables() const noexcept { return block_tables_; }

    /**@brief Sets block tables.
     * @param block_tables Block tables.
     */
    void set_block_tables(tensor block_tables) noexcept {
        block_tables_ = std::move(block_tables);
    }

    /**@brief Gets slot mapping. */
    const tensor &slot_mapping() const noexcept { return slot_mapping_; }

    /**@brief Sets slot mapping.
     * @param slot_mapping Slot mapping.
     */
    void set_slot_mapping(tensor slot_mapping) noexcept {
        slot_mapping_ = std::move(slot_mapping);
    }

  private:
    tensor block_tables_;
    tensor slot_mapping_;
};
} // namespace nncase
