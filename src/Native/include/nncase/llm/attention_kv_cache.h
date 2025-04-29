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
#include <nncase/object.h>
#include <nncase/tensor.h>

namespace nncase::llm {
class attention_kv_cache_node;
using attention_kv_cache = object_t<attention_kv_cache_node>;

class NNCASE_API attention_kv_cache_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_attention_kv_cache);

  public:
    attention_kv_cache_node(attention_config config, size_t num_seqs,
                            size_t num_tokens, tensor context_lens,
                            tensor seq_lens) noexcept
        : config_(std::move(config)),
          num_seqs_(num_seqs),
          num_tokens_(num_tokens),
          context_lens_(std::move(context_lens)),
          seq_lens_(std::move(seq_lens)) {}

    /**@brief Gets attention config. */
    attention_config config() const noexcept { return config_; }

    void config(attention_config config) noexcept {
        config_ = std::move(config);
    }

    size_t num_seqs() const noexcept { return num_seqs_; }

    void num_seqs(size_t num_requests) noexcept { num_seqs_ = num_requests; }

    size_t num_tokens() const noexcept { return num_tokens_; }

    void num_tokens(size_t num_tokens) noexcept { num_tokens_ = num_tokens; }

    result<int64_t> context_len(size_t seq_id) const noexcept;

    result<int64_t> seq_len(size_t seq_id) const noexcept;

  private:
    attention_config config_;
    size_t num_seqs_;
    size_t num_tokens_;
    tensor context_lens_;
    tensor seq_lens_;
};
} // namespace nncase::llm
