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
#include <nncase/attention_kv_cache.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <numeric>

using namespace nncase;

attention_kv_cache_node::attention_kv_cache_node(
    runtime::runtime_tensor kv_cache, runtime::runtime_tensor seq_lens,
    runtime::runtime_tensor context_lens, runtime::runtime_tensor block_tables,
    runtime::runtime_tensor slot_mapping)
    : dtype_(datatype_t::attention_kv_cache),
      kv_cache_(kv_cache.impl()),
      seq_lens_(seq_lens.impl()),
      context_lens_(context_lens.impl()),
      block_tables_(block_tables.impl()),
      slot_mapping_(slot_mapping.impl()) {}