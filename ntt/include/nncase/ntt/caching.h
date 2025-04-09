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
#include "primitive_ops.h"
#include "shape.h"
#include "tensor.h"
#include "utility.h"
#include <cstddef>
#include <tuple>

#if defined(NNCASE_CPU_MODULE) || defined(NNCASE_XPU_MODULE)
#include <attention_config.h>
#elif !defined(NNCASE_NTT_ATTENTION_CONFIG_DEFINED)
namespace nncase::ntt::caching {
struct attention_config {
    static inline constexpr size_t block_size = 16;
    static inline constexpr size_t num_layers = 24;
    static inline constexpr size_t num_kv_heads = 12;
    static inline constexpr size_t head_dim = 64;
};

using kv_type = float;
using kv_cache_block_id_t = ranked_shape<1>;
} // namespace nncase::ntt::caching
#endif

namespace nncase::ntt::caching {
class attention_kv_cache {
  public:
    attention_kv_cache() noexcept {}

    virtual ~attention_kv_cache() = default;

    size_t num_decode_tokens() const noexcept { return num_decode_tokens_; }
    size_t num_prefill_tokens() const noexcept { return num_prefill_tokens_; }
    size_t num_requests() const noexcept { return num_requests_; }
    size_t num_prefills() const noexcept { return num_prefills_; }

    size_t context_len(size_t request_id) const noexcept {
        return (size_t)context_lens_(request_id);
    }

    size_t seq_len(size_t request_id) const noexcept {
        return (size_t)seq_lens_(request_id);
    }

  private:
    size_t num_decode_tokens_;
    size_t num_prefill_tokens_;
    size_t num_requests_;
    size_t num_prefills_;

    tensor_view<int64_t, ranked_shape<1>> context_lens_;
    tensor_view<int64_t, ranked_shape<1>> seq_lens_;
};

class paged_attention_kv_cache : public attention_kv_cache {
  public:
    // [num_kv_heads, 2, block_size, head_dim]
    using block_shape_type =
        fixed_shape<attention_config::num_kv_heads, 2,
                    attention_config::block_size, attention_config::head_dim>;

    // [num_blocks, num_layers, **layer_cache_shape]
    using cache_shape_type = ranked_shape<2 + block_shape_type::rank()>;
    using block_type = tensor_view<kv_type, block_shape_type>;

    std::span<const kv_cache_block_id_t>
    context_block_ids(size_t request_id, size_t layer_id) const noexcept;

    block_type key_block(const kv_cache_block_id_t &block_id) const noexcept;
    block_type value_block(const kv_cache_block_id_t &block_id) const noexcept;
};
} // namespace nncase::ntt::caching
