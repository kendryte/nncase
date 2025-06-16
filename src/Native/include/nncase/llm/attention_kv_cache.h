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
    attention_kv_cache_node() noexcept = default;
    attention_kv_cache_node(int32_t num_seqs, int32_t num_tokens,
                            tensor context_lens, tensor seq_lens) noexcept
        : num_seqs_(num_seqs),
          num_tokens_(num_tokens),
          context_lens_(std::move(context_lens)),
          seq_lens_(std::move(seq_lens)) {}

    int32_t num_seqs() const noexcept { return num_seqs_; }

    void num_seqs(int32_t num_requests) noexcept { num_seqs_ = num_requests; }

    int32_t num_tokens() const noexcept { return num_tokens_; }

    void num_tokens(int32_t num_tokens) noexcept { num_tokens_ = num_tokens; }

    void context_lens(tensor context_lens) noexcept {
        context_lens_ = context_lens;
    }

    tensor context_lens() const noexcept { return context_lens_; }

    void seq_lens(tensor seq_lens) noexcept { seq_lens_ = seq_lens; }

    tensor seq_lens() const noexcept { return seq_lens_; }

  private:
    int32_t num_seqs_;
    int32_t num_tokens_;
    tensor context_lens_;
    tensor seq_lens_;
};
} // namespace nncase::llm
