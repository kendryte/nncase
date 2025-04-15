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
#include "tensor.h"

namespace nncase {
class attention_kv_cache_node;
using attention_kv_cache = object_t<attention_kv_cache_node>;

class NNCASE_API attention_kv_cache_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_attention_kv_cache);

  public:
    attention_kv_cache_node(attention_config config, size_t num_request,
                            tensor context_lens, tensor seq_lens) noexcept
        : config_(std::move(config)),
          num_requests_(num_request),
          context_lens_(std::move(context_lens)),
          seq_lens_(std::move(seq_lens)) {}

    /**@brief Gets attention config. */
    const attention_config &config() const noexcept { return config_; }

    /**@brief Sets attention config.
     * @param config Attention config.
     */
    void set_config(attention_config config) noexcept {
        config_ = std::move(config);
    }

    /**@brief Gets number of requests. */
    size_t num_requests() const noexcept { return num_requests_; }

    /**@brief Sets number of requests.
     * @param num_requests Number of requests.
     */
    void set_num_requests(size_t num_requests) noexcept {
        num_requests_ = num_requests;
    }

    /**@brief Gets context lens. */
    const tensor &context_lens() const noexcept { return context_lens_; }

    /**@brief Sets context lens.
     * @param context_lens Context lens.
     */
    void set_context_lens(tensor context_lens) noexcept {
        context_lens_ = std::move(context_lens);
    }

    /**@brief Gets sequence lens. */
    const tensor &seq_lens() const noexcept { return seq_lens_; }

    /**@brief Sets sequence lens.
     * @param seq_lens Sequence lens.
     */
    void set_seq_lens(tensor seq_lens) noexcept {
        seq_lens_ = std::move(seq_lens);
    }

    /**@brief Gets context length of a request.
     * @param request_id Request ID.
     * @return Context length.
     */
    result<int64_t> context_len(size_t request_id) const noexcept;

    /**@brief Gets sequence length of a request.
     * @param request_id Request ID.
     * @return Sequence length.
     */
    result<int64_t> seq_len(size_t request_id) const noexcept;

  private:
    attention_config config_;
    size_t num_requests_;
    tensor context_lens_;
    tensor seq_lens_;
};
} // namespace nncase
