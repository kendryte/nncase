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
#include "allocator.h"
#include "dump_manager.h"
#include "model.h"
#include "result.h"
#include "runtime_module.h"
#include "runtime_tensor.h"
#include <memory>
#include <nncase/runtime/attention_kv_cache.h>
#include <nncase/runtime/stream.h>
#include <nncase/shape.h>
#include <nncase/tensor.h>
#include <nncase/type.h>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API paged_attention_kv_cache : public attention_kv_cache {
  public:
    paged_attention_kv_cache() {}
    int64_t num_prefills;
    int64_t num_prefill_tokens;
    int64_t num_decode_tokens;

    /**
     * @brief [chip][die][core] tensor[num_blocks, layers, num_kv_head, 2,
     * head_dim//64, block_size, 64]<kv_type>
     *
     */
    std::vector<std::vector<std::vector<runtime_tensor>>> kv_caches;

    /**
     * @brief [chip] tensor[seq_len, block_size, 3]
     *
     * @return runtime_tensor
     */
    std::vector<runtime_tensor> block_tables;

    /**
     * @brief tensor[seq_len, block_size, 3]
     *
     * @return runtime_tensor
     */
    std::vector<runtime_tensor> slot_mapping;

    // void clear_kv_caches() { kv_caches.clear(); }
};

END_NS_NNCASE_RUNTIME
