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

#include <cstdint>
namespace nncase::llm {
enum class paged_kvcache_dim_kind : int32_t {
    num_blocks = 0,
    num_layers,
    kv,
    block_size,
    num_kv_heads,
    head_dim
};

enum class attention_cache_kind : int32_t { key = 0, value };
} // namespace nncase::llm
