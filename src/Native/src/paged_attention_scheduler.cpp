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
#include <nncase/paged_attention_scheduler.h>

using namespace nncase;

nncase::paged_attention_kv_cache
paged_attention_scheduler_node::schedule(std::vector<int64_t> session_ids,
                                         std::vector<int64_t> tokens_count) {

    auto session_num = session_ids.size();
    std::vector<int64_t> seq_lens(session_num);
    std::vector<int64_t> context_lens(session_num);
    std::vector<int64_t> slot_maping(
        std::accumulate(tokens_count.begin(), tokens_count.end(), 0));
    int64_t max_seq_len = 0;
    size_t query_token_index = 0;
    for (size_t i = 0; i < session_num; i++) {
        auto session_id = session_ids[i];
        auto token_count = tokens_count[i];
        if (session_infos_.find(session_id) == session_infos_.end()) {
            session_infos_[session_id] = detail::session_info{
                session_id * max_model_len_,
                (session_id + 1) * max_model_len_,
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
        (max_seq_len + (config_->block_size - 1)) / config_->block_size;
    std::vector<int64_t> block_tables(session_num * max_block_nums);
    for (size_t i = 0; i < session_num; i++) {
        auto session_id = session_ids[i];
        auto &info = session_infos_[session_id];
        for (size_t j = 0; j < (seq_lens[i] + (config_->block_size - 1)) /
                                   config_->block_size;
             j++) {
            block_tables[i * max_block_nums + j] =
                (info.slot_start / config_->block_size) + j;
        }
    }

    return nncase::paged_attention_kv_cache(
        std::in_place, kv_cache_storage_, kv_cache_shape_, seq_lens,
        context_lens, block_tables, slot_maping);
}
