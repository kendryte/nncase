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
                                  tensor slot_mapping,
                                  tensor kv_caches) noexcept;

    /**@brief Gets attention config. */
    const paged_attention_config &config() const noexcept {
        return attention_kv_cache_node::config()
            .as<paged_attention_config>()
            .unwrap();
    }

    /**@brief Sets attention config.
     * @param config Attention config.
     */
    void config(paged_attention_config config) noexcept {
        attention_kv_cache_node::set_config(std::move(config));
    }

    /**@brief Gets block tables. */
    const tensor &block_tables() const noexcept { return block_tables_; }

    /**@brief Sets block tables.
     * @param block_tables Block tables.
     */
    void block_tables(tensor block_tables) noexcept {
        block_tables_ = std::move(block_tables);
    }

    /**@brief Gets slot mapping. */
    const tensor &slot_mapping() const noexcept { return slot_mapping_; }

    /**@brief Sets slot mapping.
     * @param slot_mapping Slot mapping.
     */
    void slot_mapping(tensor slot_mapping) noexcept {
        slot_mapping_ = std::move(slot_mapping);
    }

    nncase::tensor sub_block(const std::vector<int> &indices);
    void sub_block(const std::vector<int> &indices, nncase::tensor block);

    /*
    nncase::tensor get_block(attention_kv_cache_kind kind, int32_t layer_id,
                             int64_t block_id);

    nncase::tensor get_context_block_ids(int32_t request_id);

    nncase::tensor get_output_slot_ids();

    nncase::tensor get_slot(attention_kv_cache_kind kind, int layer_id,
                            long slot_id);

    nncase::tensor get_slots(tensor block, int start_slot, int count);

    void update_output_slot(attention_kv_cache_kind kind, int layer_id,
                            int64_t slot_id, tensor slot);
    */

  private:
    tensor block_tables_;
    tensor slot_mapping_;
    tensor kv_caches_;
};
} // namespace nncase
