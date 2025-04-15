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

#include "nncase/runtime/runtime_op_utility.h"
#include <nncase/ntt/ntt.h>
#include <nncase/paged_attention_kv_cache.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_tensor.h>

using namespace nncase;

template <size_t Rank, typename T>
inline auto unpack_vector(const std::vector<T> &vec) {
    return std::apply(
        [&vec](auto &&...indices) { return std::make_tuple(vec[indices]...); },
        std::make_index_sequence<Rank>{});
}

paged_attention_kv_cache_node::paged_attention_kv_cache_node(
    paged_attention_config config, size_t num_request, tensor context_lens,
    tensor seq_lens, tensor block_tables, tensor slot_mapping,
    tensor kv_caches) noexcept
    : attention_kv_cache_node(std::move(config), num_request,
                              std::move(context_lens), std::move(seq_lens)),
      block_tables_(std::move(block_tables)),
      slot_mapping_(std::move(slot_mapping)),
      kv_caches_(std::move(kv_caches)) {}

nncase::tensor
paged_attention_kv_cache_node::sub_block(const std::vector<int> &indices) {
    auto cfg = config();
    auto origin_kv_shape = kv_caches_->shape();
    auto kv_shape = ntt::make_ranked_shape(
        origin_kv_shape[0], origin_kv_shape[1], origin_kv_shape[2],
        origin_kv_shape[3], origin_kv_shape[4], origin_kv_shape[5],
        typecode_bytes(cfg->kv_type()));
    auto kv_stride = ntt::default_strides(kv_shape);

    auto index = ntt::ranked_shape<7>();
    for (size_t i = 0; i < indices.size(); i++) {
        index[i] = indices[i];
    }
    auto sub_shape = ntt::make_ranked_shape(1, 1, 1, 1, 1, 1, 1);
    for (size_t i = indices.size(); i < kv_shape.length(); i++) {
        sub_shape[i] = kv_shape[i];
    }

    auto offset = linear_offset(index, kv_stride);
    auto kv_cache_buffer = kv_caches_->buffer().as_host().unwrap_or_throw();
    auto mapped_kv_cache_buffer =
        kv_cache_buffer.map(runtime::map_read).unwrap_or_throw();
    auto kv_cache_span = mapped_kv_cache_buffer.buffer();
    auto begin = kv_cache_span.data() + offset;
    auto size = linear_size(sub_shape, kv_stride);
    auto sub_span = std::span<std::byte>(begin, size);
    auto new_shape =
        dims_t(sub_shape.begin() + indices.size(), sub_shape.end());
    auto new_stride = strides_t();
    runtime::compute_strides(new_shape, new_stride);

    auto runtime_tensor = runtime::hrt::create(cfg->kv_type(), new_shape,
                                               new_stride, sub_span, false)
                              .unwrap();
    return runtime_tensor.impl();
}

void paged_attention_kv_cache_node::sub_block(const std::vector<int> &indices,
                                              nncase::tensor block) {
    auto cfg = config();
    auto origin_kv_shape = kv_caches_->shape();
    auto kv_shape = ntt::make_ranked_shape(
        origin_kv_shape[0], origin_kv_shape[1], origin_kv_shape[2],
        origin_kv_shape[3], origin_kv_shape[4], origin_kv_shape[5],
        typecode_bytes(cfg->kv_type()));
    auto kv_stride = ntt::default_strides(kv_shape);

    auto index = ntt::ranked_shape<7>();
    for (size_t i = 0; i < indices.size(); i++) {
        index[i] = indices[i];
    }
    auto sub_shape = ntt::make_ranked_shape(1, 1, 1, 1, 1, 1, 1);
    for (size_t i = indices.size(); i < kv_shape.length(); i++) {
        sub_shape[i] = kv_shape[i];
    }

    auto offset = linear_offset(index, kv_stride);
    auto kv_cache_buffer = kv_caches_->buffer().as_host().unwrap_or_throw();
    auto mapped_kv_cache_buffer =
        kv_cache_buffer.map(runtime::map_read).unwrap_or_throw();
    auto kv_cache_span = mapped_kv_cache_buffer.buffer();
    auto begin = kv_cache_span.data() + offset;
    auto size = linear_size(sub_shape, kv_stride);
    auto sub_span = std::span<std::byte>(begin, size);

    auto host_buffer = block->buffer().as_host().unwrap_or_throw();
    auto maped_buffer = host_buffer.map(runtime::map_read).unwrap_or_throw();
    auto maped_buffer_span = maped_buffer.buffer();
    std::copy_n(maped_buffer_span.begin(), maped_buffer_span.size(),
                sub_span.begin());
}