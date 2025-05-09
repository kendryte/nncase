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

#include <algorithm>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;

llm::paged_attention_kv_cache_node::paged_attention_kv_cache_node(
    paged_attention_config config, size_t num_seqs, size_t num_tokens,
    tensor context_lens, tensor seq_lens, tensor block_tables,
    tensor slot_mapping, tensor kv_caches) noexcept
    : attention_kv_cache_node(std::move(config), num_seqs, num_tokens,
                              context_lens, seq_lens),
      block_tables_(std::move(block_tables)),
      slot_mapping_(std::move(slot_mapping)),
      kv_caches_(std::move(kv_caches)) {}

nncase::tensor
llm::paged_attention_kv_cache_node::get_block_ids(int seq_id) const {
    auto block_table_buffer =
        block_tables_->buffer().as_host().unwrap_or_throw();
    auto mapped_block_table_buffer =
        block_table_buffer.map(runtime::map_read).unwrap_or_throw();
    auto stride = (*block_tables_->shape().rbegin()) * sizeof(int64_t);

    return hrt::create(dt_int64, {*block_tables_->shape().rbegin()},
                       mapped_block_table_buffer.buffer().subspan(
                           seq_id * stride, stride),
                       false)
        .unwrap()
        .impl();
}

nncase::tensor llm::paged_attention_kv_cache_node::get_slot_ids() const {
    return slot_mapping_;
}

nncase::tensor llm::paged_attention_kv_cache_node::get_block(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id,
    NNCASE_UNUSED const nncase::tensor &block_id) const {
    throw std::runtime_error("Not implemented");
}

void llm::paged_attention_kv_cache_node::update_block(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &block_id,
    NNCASE_UNUSED const nncase::tensor &block) {
    // Placeholder implementation, you should add real logic here
    throw std::runtime_error("Not implemented");
}

nncase::tensor llm::paged_attention_kv_cache_node::get_slot(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id,
    NNCASE_UNUSED const nncase::tensor &slot_id) const {
    // Placeholder implementation, you should add real logic here
    throw std::runtime_error("Not implemented");
}

void llm::paged_attention_kv_cache_node::update_slot(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &slot_id,
    NNCASE_UNUSED const nncase::tensor &slot) {}

void llm::paged_attention_kv_cache_node::update_slots(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &slot_ids,
    NNCASE_UNUSED const nncase::tensor &slots) {}
