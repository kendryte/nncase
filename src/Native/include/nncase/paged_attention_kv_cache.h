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
#include "shape.h"
#include "tensor.h"
#include "value.h"
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>

namespace nncase {
class paged_attention_kv_cache_node;
using paged_attention_kv_cache = object_t<paged_attention_kv_cache_node>;

class NNCASE_API paged_attention_kv_cache_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_paged_attention_kv_cache);

  public:
    paged_attention_kv_cache_node(std::vector<std::byte> &kv_cache_storage,
                                  std::vector<size_t> &kv_cache_shape,
                                  const std::vector<int64_t> &seq_lens,
                                  const std::vector<int64_t> &context_lens,
                                  const std::vector<int64_t> &block_tables,
                                  const std::vector<int64_t> &slot_mapping);

    /** @brief Gets element type. */
    const datatype_t &dtype() const noexcept { return dtype_; }

    int32_t num_requests() const noexcept { return seq_lens_.size(); }

    int64_t context_len(int32_t request_id) const noexcept {
        return context_lens_[request_id];
    }

    int64_t seq_len(int32_t request_id) const noexcept {
        return seq_lens_[request_id];
    }

  private:
    datatype_t dtype_;
    std::vector<std::byte> &kv_cache_storage_;
    std::vector<size_t> &kv_cache_shape_;
    const std::vector<int64_t> seq_lens_;
    const std::vector<int64_t> context_lens_;
    const std::vector<int64_t> block_tables_;
    const std::vector<int64_t> slot_mapping_;
};
} // namespace nncase
