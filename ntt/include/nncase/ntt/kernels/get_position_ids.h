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
#include <nncase/ntt/caching.h>
#include <nncase/ntt/shape.h>
#include <type_traits>

namespace nncase::ntt {

template <class TKVCache, ShardedTensor TOut>
void get_position_ids(TKVCache &&kv_cache_tensor, TOut output) {
    using TOutType = typename std::decay_t<TOut>;
    using mesh_type = typename TOutType::mesh_type;
    using TOutElem = typename TOutType::local_tensor_type::value_type;

    const auto local_mesh_index = mesh_type::local_index();
    const auto global_offset =
        output.sharding().global_offset(output.shape(), local_mesh_index);
    auto local_output = output.local();
    const auto local_shape = local_output.shape();

    const auto global_start = global_offset[0_dim];
    const auto global_end = global_start + local_shape[0_dim];

    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    size_t out_i = 0;
    size_t acc = 0;
    for (size_t seq_id = 0; seq_id < kv_cache.num_seqs(); seq_id++) {
        size_t context_len = kv_cache.context_len(seq_id);
        size_t seq_len = kv_cache.seq_len(seq_id);
        auto query_len = seq_len - context_len;
        if (acc >= global_end)
            break;

        if (acc + query_len < global_start)
            continue;

        for (size_t i = global_start > acc ? global_start - acc : 0;
             i < query_len, out_i < local_shape[0_dim]; i++) {
            local_output(out_i) = (TOutElem)(context_len + i);
            out_i++;
        }
        acc += query_len;
    }
}
} // namespace nncase::ntt
