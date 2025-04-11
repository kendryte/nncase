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

#include "object.h"
#include "paged_attention_config.h"
#include "paged_attention_kv_cache.h"
#include "tensor.h"
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <numeric>

namespace nncase {

namespace detail {
class session_info {
  public:
    int64_t slot_start;
    int64_t slot_end;
    int64_t context_len;
};
} // namespace detail

class paged_attention_scheduler_node : public object_node {
    DEFINE_OBJECT_KIND(paged_attention_scheduler_node,
                       object_paged_attention_scheduler);

  public:
    paged_attention_scheduler_node(int max_model_len)
        : kv_cache_storage_(),
          kv_cache_shape_(),
          max_model_len_(max_model_len),
          session_infos_() {}

    void initialize(nncase::paged_attention_config config, int num_blocks) {
        config_ = config;
        num_blocks_ = num_blocks;
        kv_cache_shape_ = {2, (size_t)config_->num_layers, (size_t)num_blocks_,
                           (size_t)config_->num_kv_heads,
                           (size_t)config_->head_dim};
        kv_cache_storage_.reserve(sizeof(float) *
                                  std::accumulate(kv_cache_shape_.begin(),
                                                  kv_cache_shape_.end(), 1,
                                                  std::multiplies<int>()));
    }

    nncase::paged_attention_kv_cache
    schedule(std::vector<int64_t> session_ids,
             std::vector<int64_t> tokens_count);

  private:
    nncase::paged_attention_config config_;
    int num_blocks_;
    std::vector<std::byte> kv_cache_storage_;
    std::vector<size_t> kv_cache_shape_;
    int64_t max_model_len_;
    std::unordered_map<int64_t, detail::session_info> session_infos_;
};
} // namespace nncase