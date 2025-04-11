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
#include "attention_config.h"
#include "object.h"

namespace nncase {
class paged_attention_config_node : public attention_config_node {
    DEFINE_OBJECT_KIND(attention_config_node, object_paged_attention_config);

  public:
    paged_attention_config_node(size_t num_layers, size_t num_kv_heads,
                                size_t head_dim, size_t block_size) noexcept
        : attention_config_node(num_layers, num_kv_heads, head_dim),
          block_size_(block_size) {}

    size_t block_size() const noexcept { return block_size_; }
    void block_size(size_t block_size) noexcept { block_size_ = block_size; }

  private:
    size_t block_size_;
};

using paged_attention_config = object_t<paged_attention_config_node>;
} // namespace nncase
