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

template <class TKVCache, Tensor TOut, class TSharding, Shape TGlobalShape>
void get_position_ids(TKVCache &&kv_cache_tensor, TOut output,
                      const TSharding &sharding,
                      const TGlobalShape &global_shape) {
    using TOutType = typename std::decay_t<TOut>;
    using mesh_type = typename TSharding::mesh_type;
    using TOutElem = typename TOutType::value_type;

    const auto local_mesh_index = mesh_type::local_index();
    const auto global_offset =
        sharding.global_offset(global_shape, local_mesh_index);
    const auto local_shape = output.shape();

    const auto global_start = global_offset[0_dim];
    const auto global_end = global_start + local_shape[0_dim];

    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    for (size_t seq_id = 0, query_start_loc = 0, out_loc = 0;
         seq_id < kv_cache.num_seqs(); seq_id++) {
        size_t context_len = kv_cache.context_len(seq_id);
        size_t seq_len = kv_cache.seq_len(seq_id);
        auto query_len = seq_len - context_len;
        auto query_end_loc = query_start_loc + query_len;
        if (query_start_loc >= global_end) {
            return;
        }

        if (query_end_loc <= global_start) {
            query_start_loc = query_end_loc;
            continue;
        }

        const size_t pos_id_start =
            context_len + (global_start + out_loc - query_start_loc);
        const size_t pos_id_end = ntt::min(
            static_cast<size_t>(pos_id_start + (local_shape[0_dim] - out_loc)),
            seq_len);

        for (size_t pos_id = pos_id_start; pos_id < pos_id_end; pos_id++) {
            output(out_loc) = static_cast<TOutElem>(pos_id);
            out_loc++;
        }

        query_start_loc = query_end_loc;
    }
}
} // namespace nncase::ntt
