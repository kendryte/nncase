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
#include "nncase/runtime/simple_types.h"
#include "nncase/runtime/util.h"
#include <cstdint>
#include <nncase/paged_attention_scheduler.h>
#include <nncase/runtime/runtime_tensor.h>
#include <span>

using namespace nncase;
using namespace nncase::runtime;

paged_attention_scheduler_node::paged_attention_scheduler_node(
    nncase::paged_attention_config config, size_t num_blocks,
    size_t max_model_len)
    : config_(std::move(config)),
      num_blocks_(num_blocks),
      max_model_len_(max_model_len),
      kv_cache_shape_{(size_t)num_blocks_,
                      (size_t)config_->num_layers(),
                      (size_t)config_->num_kv_heads(),
                      2,
                      (size_t)config_->block_size(),
                      (size_t)config_->head_dim()},
      kv_caches_(
          hrt::create(dt_float32, kv_cache_shape_).unwrap_or_throw().impl()) {}

result<nncase::paged_attention_kv_cache>
paged_attention_scheduler_node::schedule(tensor session_ids,
                                         tensor tokens_count) {
    try_input_with_ty(session_ids_value, session_ids, int64_t);
    try_input_with_ty(tokens_count_value, tokens_count, int64_t);

    auto session_num = session_ids->length();
    auto query_num = (size_t)std::accumulate(
        tokens_count_value, tokens_count_value + session_num, 0);
    std::vector<int64_t> seq_lens(session_num);
    std::vector<int64_t> context_lens(session_num);
    std::vector<int64_t> slot_maping(query_num);
    int64_t max_seq_len = 0;
    size_t query_token_index = 0;
    for (size_t i = 0; i < session_num; i++) {
        auto session_id = session_ids_value[i];
        auto token_count = tokens_count_value[i];
        if (session_infos_.find(session_id) == session_infos_.end()) {
            session_infos_[session_id] = detail::session_info{
                session_id * (int64_t)max_model_len_,
                (session_id + 1) * (int64_t)max_model_len_,
                0,
            };
        }
        auto &info = session_infos_[session_id];
        context_lens[i] = info.context_len;
        seq_lens[i] = info.context_len + token_count;
        if (seq_lens[i] > max_model_len_) {
            throw std::runtime_error(
                "the seq lens is large than max model length!");
        }
        max_seq_len = std::max(max_seq_len, seq_lens[i]);
        for (size_t j = query_token_index; j < token_count; j++) {
            slot_maping[j] =
                info.slot_start + info.context_len + (j - query_token_index);
        }
    }
    auto max_block_nums =
        (max_seq_len + (config_->block_size() - 1)) / config_->block_size();
    std::vector<int64_t> block_tables(session_num * max_block_nums);
    for (size_t i = 0; i < session_num; i++) {
        auto session_id = session_ids_value[i];
        auto &info = session_infos_[session_id];
        for (size_t j = 0; j < (seq_lens[i] + (config_->block_size() - 1)) /
                                   config_->block_size();
             j++) {
            block_tables[i * max_block_nums + j] =
                (info.slot_start / config_->block_size()) + j;
        }
    }

    try_var(seq_lens_tensor,
            hrt::create(dt_int64, {session_num},
                        std::as_writable_bytes(std::span(seq_lens)), true,
                        hrt::memory_pool_t::pool_shared));
    try_var(context_lens_tensor,
            hrt::create(dt_int64, {session_num},
                        std::as_writable_bytes(std::span(context_lens)), true,
                        hrt::memory_pool_t::pool_shared));
    try_var(block_tables_tensor,
            hrt::create(dt_int64, {session_num, max_block_nums},
                        std::as_writable_bytes(std::span(block_tables)), true,
                        hrt::memory_pool_t::pool_shared));
    try_var(slot_maping_tensor,
            hrt::create(dt_int64, {query_num},
                        std::as_writable_bytes(std::span(slot_maping)), true,
                        hrt::memory_pool_t::pool_shared));
    return ok(nncase::paged_attention_kv_cache(
        std::in_place, config_, session_num, context_lens_tensor.impl(),
        seq_lens_tensor.impl(), block_tables_tensor.impl(),
        slot_maping_tensor.impl(), kv_caches_));
}
