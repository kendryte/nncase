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
#include <nncase/paged_attention_kv_cache.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <numeric>

using namespace nncase;

paged_attention_kv_cache_node::paged_attention_kv_cache_node(
    std::vector<std::byte> &kv_cache_storage,
    std::vector<size_t> &kv_cache_shape, const std::vector<int64_t> &seq_lens,
    const std::vector<int64_t> &context_lens,
    const std::vector<int64_t> &block_tables,
    const std::vector<int64_t> &slot_mapping)
    : dtype_(datatype_t::attention_kv_cache),
      kv_cache_storage_(kv_cache_storage),
      kv_cache_shape_(kv_cache_shape),
      seq_lens_(std::move(seq_lens)),
      context_lens_(std::move(context_lens)),
      block_tables_(std::move(block_tables)),
      slot_mapping_(std::move(slot_mapping)) {}
