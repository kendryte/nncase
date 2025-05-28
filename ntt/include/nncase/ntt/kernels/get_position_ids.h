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
#include "nncase/ntt/kernels/binary.h"
#include "nncase/ntt/kernels/matmul.h"
#include "nncase/ntt/kernels/reduce.h"
#include "nncase/ntt/kernels/unary.h"
#include "nncase/ntt/tensor_traits.h"
#include <nncase/ntt/caching.h>
#include <nncase/ntt/shape.h>
#include <nncase/ntt/sharding.h>
#include <type_traits>

namespace nncase::ntt {

template <class TKVCache, class TOut>
void get_position_ids(TKVCache kv_cache_tensor, TOut output){
    using TOutType = typename std::decay_t<TOut>;
    using TOutElem = typename TOutType::element_type;

    auto &kv_cache = kv_cache_tensor(0);
    using kv_cache_t = typename std::decay_t<decltype(kv_cache)>;
    using config_t = typename kv_cache_t::config_t;
    const auto output_size = output.size();
    size_t i = 0;
    for(size_t seq_id = 0; seq_id <kv_cache.num_seqs(); seq_id++){
        size_t history_len = kv_cache.context_len(seq_id);
        size_t seq_len = kv_cache.seq_len(seq_id);
        auto user_range = seq_len - history_len;
        for(size_t user_i = 0; user_i < user_range, i < output_size; i++, user_i++)
        {
            output(i) = (TOutType)history_len + (TOutType)user_i * 1;
        }  
    }
}
} // namespace nncase::ntt
