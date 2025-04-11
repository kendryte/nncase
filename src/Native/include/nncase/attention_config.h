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

namespace nncase {
class attention_config_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_attention_config);

  public:
    attention_config_node(size_t num_layers, size_t num_kv_heads,
                          size_t head_dim) noexcept
        : num_layers_(num_layers),
          num_kv_heads_(num_kv_heads),
          head_dim_(head_dim) {}

    size_t num_layers() const noexcept { return num_layers_; }
    void num_layers(size_t num_layers) noexcept { num_layers_ = num_layers; }

    size_t num_kv_heads() const noexcept { return num_kv_heads_; }
    void num_kv_heads(size_t num_kv_heads) noexcept {
        num_kv_heads_ = num_kv_heads;
    }

    size_t head_dim() const noexcept { return head_dim_; }
    void head_dim(size_t head_dim) noexcept { head_dim_ = head_dim; }

  private:
    size_t num_layers_;
    size_t num_kv_heads_;
    size_t head_dim_;
};

using attention_config = object_t<attention_config_node>;
} // namespace nncase
