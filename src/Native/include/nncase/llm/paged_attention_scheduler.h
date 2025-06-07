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

#include "nncase/runtime/simple_types.h"
#include "paged_attention_config.h"
#include "paged_attention_kv_cache.h"
#include <nncase/object.h>
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/tensor.h>
#include <numeric>

namespace nncase::llm {

namespace detail {
class session_info {
  public:
    int64_t slot_start;
    int64_t slot_end;
    int64_t context_len;
};

inline void materialize_block_table(int64_t *block_table,
                                    const strides_t &strides,
                                    std::vector<size_t> &indices,
                                    int64_t logical_block_id, int num_blocks,
                                    const dims_t &hierarchy,
                                    const llm::paged_attention_config &config) {
    int64_t physical_block_id = logical_block_id;

    for (size_t topo_id = 0; topo_id < config->sharding_axes().size();
         topo_id++) {
        switch (config->sharding_axes()[topo_id]) {
        case llm::paged_kvcache_dim_kind::num_blocks: {
            int parallelism = 1;
            for (int axis : config->axis_policies()[topo_id]) {
                parallelism *= hierarchy[axis];
            }

            if (num_blocks < parallelism && num_blocks % parallelism != 0) {
                throw std::runtime_error("numBlocks < parallelism");
            }

            int num_block_tile = num_blocks / parallelism;
            int value = physical_block_id / num_block_tile;
            physical_block_id = physical_block_id % num_block_tile;
            block_table[runtime::linear_offset(indices, strides)] = value;
            break;
        }
        case llm::paged_kvcache_dim_kind::num_kv_heads: {
            block_table[runtime::linear_offset(indices, strides)] = -1;
            break;
        }
        default:
            throw std::invalid_argument("Invalid sharding axis");
        }

        indices.back()++;
    }

    block_table[runtime::linear_offset(indices, strides)] = physical_block_id;
}

inline void materialize_slot_mapping_id(
    int64_t *slot_mapping, const strides_t &strides,
    std::vector<size_t> &indices, int64_t logical_slot_id, int num_blocks,
    const dims_t &hierarchy, const llm::paged_attention_config &config) {
    int64_t physical_slot_id = logical_slot_id;

    for (size_t shard_id = 0; shard_id < config->sharding_axes().size();
         shard_id++) {
        switch (config->sharding_axes()[shard_id]) {
        case llm::paged_kvcache_dim_kind::num_blocks: {
            int parallelism = 1;
            for (int axis : config->axis_policies()[shard_id]) {
                parallelism *= hierarchy[axis];
            }

            if (num_blocks < parallelism && num_blocks % parallelism != 0) {
                throw std::runtime_error("numBlocks < parallelism");
            }

            int64_t num_block_tile =
                (num_blocks / parallelism) * config->block_size();
            int64_t value = physical_slot_id / num_block_tile;
            physical_slot_id = physical_slot_id % num_block_tile;
            slot_mapping[runtime::linear_offset(indices, strides)] = value;
            break;
        }
        case llm::paged_kvcache_dim_kind::num_kv_heads: {
            if (config->axis_policies()[shard_id].size() == 1) {
                slot_mapping[runtime::linear_offset(indices, strides)] = -1;
            } else {
                throw std::invalid_argument(
                    "Invalid axis policy for num_kv_heads");
            }
            break;
        }
        default:
            throw std::invalid_argument("Invalid sharding axis");
        }

        indices.back()++;
    }

    slot_mapping[runtime::linear_offset(indices, strides)] = physical_slot_id;
}

} // namespace detail

class paged_attention_scheduler_node : public object_node {
    DEFINE_OBJECT_KIND(paged_attention_scheduler_node,
                       object_paged_attention_scheduler);

  public:
    paged_attention_scheduler_node(paged_attention_config config,
                                   size_t num_blocks, size_t max_model_len,
                                   const std::vector<int> &hierarchy)
        : config_(std::move(config)),
          num_blocks_(num_blocks),
          max_model_len_(max_model_len),
          hierarchy_(hierarchy.begin(), hierarchy.end()),
          conversation_id_(0) {
        // Validate max_model_len is multiple of block_size
        if (max_model_len_ % config_->block_size() != 0) {
            throw std::invalid_argument(
                "Max model length must be a multiple of block size.");
        }

        auto kv_shard_shape =
            config_->get_logical_shard_dimensions(num_blocks_, hierarchy_);
        dims_t kv_topo_shape = {kv_shard_shape.begin(),
                                kv_shard_shape.begin() +
                                    config_->sharding_axes().size()};

        kv_cache_ = llm::paged_attention_kv_cache(
            std::in_place, config_, 0, 0, tensor(), tensor(), tensor(),
            tensor(), num_blocks_, kv_topo_shape);
        for (size_t i = 0; i < runtime::compute_size(kv_topo_shape); i++) {
            auto storage =
                runtime::hrt::create(
                    config_->kv_type(),
                    {kv_shard_shape.begin() + config_->sharding_axes().size(),
                     kv_shard_shape.end()})
                    .unwrap();
            kv_cache_->kv_cache(i, storage.impl());
        }
    }

    paged_attention_kv_cache schedule(const std::vector<long> &session_ids,
                                      const std::vector<long> &query_lens) {

        auto num_seqs = session_ids.size();
        if (num_seqs != query_lens.size()) {
            throw std::invalid_argument(
                "Session IDs and query lengths must have the same length.");
        }

        auto num_tokens =
            (size_t)std::accumulate(query_lens.begin(), query_lens.end(), 0);

        std::vector<int64_t> seq_lens(num_seqs);
        std::vector<int64_t> context_lens(num_seqs);
        int64_t max_seq_len = 0;

        // Process each session
        for (int seq_id = 0; seq_id < num_seqs; seq_id++) {
            long session_id = session_ids[seq_id];
            long query_len = query_lens[seq_id];

            auto it = session_infos_.find(session_id);
            detail::session_info *info;
            if (it == session_infos_.end()) {
                info = &session_infos_[session_id];
                info->slot_start = session_id * max_model_len_;
                info->slot_end = (session_id + 1) * max_model_len_;
                info->context_len = 0;
            } else {
                info = &it->second;
            }

            if (info->slot_end > num_blocks_ * config_->block_size()) {
                throw std::runtime_error(
                    "Can't allocate KV cache for new session!");
            }

            context_lens[seq_id] = info->context_len;
            info->context_len += query_len;
            seq_lens[seq_id] = context_lens[seq_id] + query_len;

            if (seq_lens[seq_id] > max_model_len_) {
                throw std::runtime_error(
                    "The sequence length is larger than max model length !");
            }

            max_seq_len = std::max(max_seq_len, seq_lens[seq_id]);
        }

        // block table tensor
        auto block_table_dims =
            config_->get_block_table_dimensions(num_seqs, max_seq_len);
        auto block_table_strides =
            runtime::get_default_strides(block_table_dims);
        std::vector<int64_t> block_tables(
            runtime::compute_size(block_table_dims), 0);
        for (int seq_id = 0; seq_id < num_seqs; seq_id++) {
            auto &info = session_infos_[session_ids[seq_id]];
            for (int64_t item_id = 0, logical_slot_id = info.slot_start;
                 logical_slot_id <
                 runtime::align_up(info.slot_start + seq_lens[seq_id],
                                   (int64_t)config_->block_size());
                 logical_slot_id += config_->block_size(), item_id++) {
                auto logical_block_id = logical_slot_id / config_->block_size();
                std::vector<size_t> indices{(size_t)seq_id, (size_t)item_id,
                                            (size_t)0};
                detail::materialize_block_table(
                    block_tables.data(), block_table_strides, indices,
                    logical_block_id, num_blocks_, hierarchy_, config_);
            }
        }

        // slot mapping tensor
        auto slot_mapping_dims =
            config_->get_slot_mapping_dimensions(num_tokens);
        auto slot_mapping_strides =
            runtime::get_default_strides(slot_mapping_dims);
        std::vector<int64_t> slot_mappings(
            runtime::compute_size(slot_mapping_dims), 0);

        for (int64_t token_id = 0, seq_id = 0; seq_id < num_seqs; seq_id++) {
            auto &info = session_infos_[session_ids[seq_id]];
            auto context_len = context_lens[seq_id];
            for (int64_t logical_slot_id = info.slot_start + context_len;
                 logical_slot_id <
                 info.slot_start + context_len + query_lens[seq_id];
                 logical_slot_id++) {
                std::vector<size_t> indices{(size_t)token_id, 0};
                detail::materialize_slot_mapping_id(
                    slot_mappings.data(), slot_mapping_strides, indices,
                    logical_slot_id, num_blocks_, hierarchy_, config_);
                token_id++;
            }
        }

        auto seq_lens_tensor =
            runtime::hrt::create(dt_int64, {num_seqs},
                                 std::as_writable_bytes(std::span(seq_lens)),
                                 true, runtime::hrt::memory_pool_t::pool_shared)
                .unwrap();
        auto context_lens_tensor =
            runtime::hrt::create(
                dt_int64, {num_seqs},
                std::as_writable_bytes(std::span(context_lens)), true,
                runtime::hrt::memory_pool_t::pool_shared)
                .unwrap();
        auto block_tables_tensor =
            runtime::hrt::create(
                dt_int64, block_table_dims,
                std::as_writable_bytes(std::span(block_tables)), true,
                runtime::hrt::memory_pool_t::pool_shared)
                .unwrap();
        auto slot_mapping_tensor =
            runtime::hrt::create(
                dt_int64, slot_mapping_dims,
                std::as_writable_bytes(std::span(slot_mappings)), true,
                runtime::hrt::memory_pool_t::pool_shared)
                .unwrap();

        kv_cache_->num_seqs(num_seqs);
        kv_cache_->num_tokens(num_tokens);
        kv_cache_->seq_lens(seq_lens_tensor.impl());
        kv_cache_->context_lens(context_lens_tensor.impl());
        kv_cache_->block_table(block_tables_tensor.impl());
        kv_cache_->slot_mapping(slot_mapping_tensor.impl());
        kv_cache_->conversation_id(conversation_id_++);
        return kv_cache_;
    }

  private:
    paged_attention_config config_;
    size_t num_blocks_;
    size_t max_model_len_;
    dims_t hierarchy_;
    paged_attention_kv_cache kv_cache_;
    std::unordered_map<int64_t, detail::session_info> session_infos_;
    size_t conversation_id_;
};

using paged_attention_scheduler = object_t<paged_attention_scheduler_node>;
} // namespace nncase::llm
