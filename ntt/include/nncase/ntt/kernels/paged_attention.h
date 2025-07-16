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
#include "../caching.h"
#include "binary.h"
#include "matmul.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "reduce.h"
#include "unary.h"
#include <type_traits>

namespace nncase::ntt {
#if false
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8>
void create_paged_attention_kv_cache(T0 num_seqs, T1 num_tokens,
                                     T2 context_lens, T3 seq_lens,
                                     T4 block_table, T5 slot_mapping,
                                     T6 num_blocks, T7 kv_caches, T8 output) {
    auto *kv_cache = output.elements().data();
    using paged_attention_kv_cache_t = typename T8::element_type;

    using kv_type_t = typename paged_attention_kv_cache_t::kv_type_t;
    using kv_shape_t = typename paged_attention_kv_cache_t::kv_shape_t;
    using kv_topo_t = typename paged_attention_kv_cache_t::config_t::kv_topo_t;

    if constexpr (kv_topo_t::rank() > 0) {
        typename paged_attention_kv_cache_t::kv_tensor_type_t kv_tensor;

        using mesh_type = typename T7::mesh_type;
        auto program_ids = distributed::program_ids();
        // printf("[nncase_log] try create kv_cache at die:%ld core: %ld\n",
        //        program_ids[1], program_ids[2]);
        distributed::topology_synchronize(); // sync for shard kv cache.

        apply(kv_shape_t{}, [&](auto kv_index) {
            for (size_t i = 0; i < kv_topo_t::rank(); i++) {
                program_ids[kv_topo_t::at(i)] = kv_index[i];
            }
            auto shard_index = mesh_type::index_from_program_id(program_ids);
            auto remote =
                kv_caches.template remote<mesh_type::scope>(shard_index);
            kv_tensor(kv_index) = (intptr_t)remote.elements().data();
        });

        distributed::topology_synchronize(); // sync for shard kv cache.

        new (kv_cache) caching::paged_attention_kv_cache<
            typename paged_attention_kv_cache_t::config_t>(
            num_seqs(0), num_tokens(0),
            tensor_view<int64_t, ranked_shape<1>>(
                context_lens.buffer(), to_ranked_shape(context_lens.shape())),
            tensor_view<int64_t, ranked_shape<1>>(
                seq_lens.buffer(), to_ranked_shape(seq_lens.shape())),
            tensor_view<int64_t, ranked_shape<3>>(
                block_table.buffer(), to_ranked_shape(block_table.shape())),
            tensor_view<int64_t, ranked_shape<2>>(
                slot_mapping.buffer(), to_ranked_shape(slot_mapping.shape())),
            num_blocks(0), kv_tensor);

        program_ids = distributed::program_ids();
        // if (std::all_of(program_ids.begin(), program_ids.end(),
        //                 [](auto &i) { return i == 0; })) {
        //     apply(kv_shape_t{}, [&](auto kv_index) {
        //         printf("kv_tensor [%d, %d]: %p\n", kv_index[0], kv_index[1],
        //                kv_tensor(kv_index));
        //     });
        // }
        // printf("[nncase_log] kv_cache created at die:%ld core: %ld\n",
        //        program_ids[1], program_ids[2]);
    }
}
#endif

template <caching::attention_cache_kind Kind, FixedDimensions TLayout,
          class TSlots, Tensor TKVCache>
    requires(Tensor<TSlots> || ShardedTensor<TSlots>)
constexpr void update_paged_attention_kv_cache(const TSlots &slots_tensor,
                                               TKVCache &kv_cache_tensor,
                                               size_t layer_id,
                                               const TLayout &layout) noexcept {
    // FIXME: Use DP rank
    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    using config_t = typename std::decay_t<decltype(kv_cache)>::config_t;
    constexpr auto num_heads = config_t::num_kv_heads;

    if constexpr (ShardedTensor<TSlots>) {
        const auto local_slots = slots_tensor.local();
        const auto slots_sharding = slots_tensor.sharding();
        using slots_mesh_type = typename TSlots::mesh_type;

        // slots: [seq, numHeads, headDim]
        const auto shard_index = slots_mesh_type::local_index();
        const auto slots_global_shape = slots_tensor.shape();
        const auto slots_local_shape = local_slots.shape();
        const auto slots_global_offset =
            slots_sharding.global_offset(slots_global_shape, shard_index);

        const auto seq_index = layout.index_of(
            fixed_dim_v<(dim_t)caching::attention_dim_kind::seq>);
        const auto head_index = layout.index_of(
            fixed_dim_v<(dim_t)caching::attention_dim_kind::head>);
        const auto dim_index = layout.index_of(
            fixed_dim_v<(dim_t)caching::attention_dim_kind::dim>);

        auto local_slots_starts = make_zeros_shape<3>()
                                      .template replace_at<seq_index>(0)
                                      .template replace_at<head_index>(0);
        const auto local_slots_shape =
            make_ones_shape<3>().replace_at<dim_index>(
                slots_local_shape[dim_index]);
        constexpr auto local_slots_squeeze =
            fixed_shape_v<seq_index, head_index>;

        for (dim_t local_token_id = 0;
             local_token_id < slots_local_shape[seq_index]; local_token_id++) {
            // slot mapping is broadcast, but slot maybe is sharding.
            auto global_token_id =
                slots_global_offset[seq_index] + local_token_id;

            // support plugin kernel style padding.
            if (global_token_id >= kv_cache.num_tokens()) {
                continue;
            }
            auto slot_id = kv_cache.get_slot_id(global_token_id);
            local_slots_starts[seq_index] = local_token_id;

            for (dim_t local_head_id = 0;
                 local_head_id < slots_local_shape[head_index];
                 local_head_id++) {

                local_slots_starts[head_index] = local_head_id;
                const auto local_slot =
                    local_slots.view(local_slots_starts, local_slots_shape)
                        .squeeze(local_slots_squeeze);

                // process kv_head different sharding on slot and kv cache.
                const auto kv_head_policy = config_t::template axis_policy<
                    caching::paged_kvcache_dim_kind::num_kv_heads>();
                const auto global_head_id =
                    slots_global_offset[head_index] + local_head_id;
                // todo need consider num_kv_head packed.
                const auto kv_local_head_dim =
                    kv_head_policy.template shard_dim<slots_mesh_type>(
                        config_t::num_kv_heads, shard_index);
                const auto kv_local_head_id =
                    global_head_id % kv_local_head_dim;

                kv_cache.template update_slot<Kind>(layer_id, kv_local_head_id,
                                                    slot_id, local_slot);
            }
        }

        distributed::topology_synchronize();
    } else {
        for (size_t head_id = 0; head_id < num_heads; head_id++) {
            kv_cache.template update_slots<Kind>(layer_id, head_id,
                                                 slots_tensor);
        }
    }
}

template <FixedDimensions QLayout, Tensor TQ, Tensor TKVCache, Tensor TScale,
          class TOutput, Tensor TExtra>
    requires(Tensor<std::decay_t<TOutput>>)
void paged_attention(
    const TQ &q_tensor, TKVCache &kv_cache_tensor,
    TExtra &extra_tensor, /* [head_q, max_query_len, max_seq_len] + [head_q,
                            max_query_len, 1] */
    const TScale &scale, size_t layer_id, TOutput &&output_tensor,
    const QLayout &q_layout) noexcept {
    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    using kv_cache_t = typename std::decay_t<decltype(kv_cache)>;
    using config_t = typename kv_cache_t::config_t;
    using kv_prim_type_t = typename config_t::kv_prim_type;

    constexpr auto num_kv_heads = config_t::num_kv_heads;
    auto q_shape = q_tensor.shape();

    // Get sequence and dimension information
    const auto seq_index =
        q_layout.index_of(fixed_dim_v<(dim_t)caching::attention_dim_kind::seq>);
    const auto head_index = q_layout.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::head>);
    const auto dim_index =
        q_layout.index_of(fixed_dim_v<(dim_t)caching::attention_dim_kind::dim>);

    auto q_slice_start = make_zeros_shape<3>()
                             .template replace_at<head_index>(0)
                             .template replace_at<seq_index>(0);
    const auto q_slice_shape =
        make_ones_shape<3>().template replace_at<dim_index>(q_shape[dim_index]);
    const auto q_squeeze = fixed_shape_v<seq_index, head_index>;

    for (dim_t query_start_loc = 0, seq_id = 0,
               seq_len = kv_cache.seq_len(seq_id),
               context_len = kv_cache.context_len(seq_id),
               query_len = seq_len - context_len;
         seq_id < kv_cache.num_seqs(); seq_id++, query_start_loc += query_len) {
        const auto s_shape =
            ntt::make_shape(q_shape[head_index], query_len, seq_len);
        const auto reduce_s_shape =
            ntt::make_shape(q_shape[head_index], query_len, dim_one);
        if (extra_tensor.elements().size_bytes() <
            (s_shape.length() + reduce_s_shape.length()) *
                (sizeof(kv_prim_type_t))) {
            printf("extra_tensor is not enough.\n");
            std::terminate();
        }

        auto s = make_tensor_view_from_address(
            reinterpret_cast<kv_prim_type_t *>(extra_tensor.elements().data()),
            s_shape);
        auto reduce_s = make_tensor_view_from_address(
            reinterpret_cast<kv_prim_type_t *>(extra_tensor.elements().data()) +
                s_shape.length(),
            reduce_s_shape);

        // s = q * k^T : [head_q, query_len, seq_len]
        for (size_t q_head_id = 0; q_head_id < q_shape[head_index];
             q_head_id++) {
            auto k_head_id = q_head_id / (q_shape[head_index] / num_kv_heads);
            q_slice_start[head_index] = q_head_id;

            for (size_t q_id = 0, q_id_batch = query_start_loc;
                 q_id_batch < query_start_loc + query_len;
                 q_id_batch++, q_id++) {
                q_slice_start[seq_index] = q_id_batch;

                // [1, dim']<dim>
                const auto q_slice = q_tensor.view(q_slice_start, q_slice_shape)
                                         .squeeze(q_squeeze)
                                         .unsqueeze(fixed_shape_v<0>);

                //  block_slice
                for (dim_t context_bid = 0;
                     context_bid < (seq_len + (config_t::block_size - 1)) /
                                       config_t::block_size;
                     context_bid++) {
                    const auto block_id =
                        kv_cache.get_block_id(seq_id, context_bid);
                    const auto k_block = kv_cache.template get_block<
                        caching::attention_cache_kind::key>(layer_id, k_head_id,
                                                            block_id);

                    static_assert(
                        config_t::block_layout ==
                            fixed_shape_v<
                                (dim_t)
                                    caching::paged_kvcache_dim_kind::head_dim,
                                (dim_t)caching::paged_kvcache_dim_kind::
                                    block_size>,
                        "block layout is not supported.");
                    auto valid_block_size =
                        ntt::min(seq_len - context_bid * config_t::block_size,
                                 config_t::block_size);

                    // [dim', valid_block_size]<dim>
                    auto k_slice =
                        k_block.view(make_zeros_shape<2>(),
                                     ntt::make_shape(k_block.shape()[0_dim],
                                                     valid_block_size));
                    // [1, valid_block_size]
                    auto s_slice =
                        s.view(ntt::make_shape(q_head_id, q_id,
                                               context_bid *
                                                   config_t::block_size),
                               ntt::make_shape(dim_one, dim_one,
                                               valid_block_size))
                            .squeeze(fixed_shape_v<1>);

                    // [1, valid_block_size] = [1, dim']<dim> @ [dim',
                    // valid_block_size]<dim>
                    ntt::matmul<false>(q_slice, k_slice, s_slice,
                                       fixed_shape_v<1>, {}, fixed_shape_v<0>,
                                       ntt::fixed_shape_v<>);
                }
            }
        }

        // scale s : [head_q, query_len, seq_len]
        ntt::binary<ntt::ops::mul>(s, scale, s);
        // add tril mask.
        size_t diagonal = seq_len - query_len;
        for (size_t q_head_id = 0; q_head_id < s.shape()[0]; q_head_id++) {
            for (size_t q_id = 0; q_id < s.shape()[1]; q_id++) {
                for (size_t context_id = q_id + diagonal + 1;
                     context_id < s.shape()[2]; context_id++) {
                    s(q_head_id, q_id, context_id) -=
                        std::numeric_limits<kv_prim_type_t>::infinity();
                }
            }
        }

        // d = softmax(s) : [head_q, query_len, seq_len]
        {
            // max = reduce max(s, -1)
            ntt::reduce_max(s, reduce_s, fixed_shape_v<2>);
            // sub（input） =  input - max;
            ntt::binary<ops::sub>(s, reduce_s, s);
            // exp（input） = exp(sub)
            ntt::unary<ops::exp>(s, s);
            // sum（max） = reduce sum(exp, -1)
            ntt::reduce_sum(s, reduce_s, fixed_shape_v<2>);
            // div（input） = exp / sum
            ntt::binary<ops::div>(s, reduce_s, s);
        }

        // d @ v : [head_q, query_len, dim], depend by qlayout.
        dynamic_shape_t<3> s_slice_start;
        for (size_t q_head_id = 0; q_head_id < q_shape[head_index];
             q_head_id++) {
            auto v_head_id = q_head_id / (q_shape[head_index] / num_kv_heads);
            s_slice_start[0_dim] = q_head_id;
            q_slice_start[head_index] = q_head_id;
            for (size_t q_id = 0, q_id_batch = query_start_loc;
                 q_id_batch < query_start_loc + query_len;
                 q_id_batch++, q_id++) {
                s_slice_start[1_dim] = q_id;
                q_slice_start[seq_index] = q_id_batch;

                // [1, dim']<dim>
                auto d_slice = output_tensor.view(q_slice_start, q_slice_shape)
                                   .squeeze(q_squeeze)
                                   .unsqueeze(fixed_shape_v<0>);

                for (size_t context_bid = 0;
                     context_bid < (seq_len + (config_t::block_size - 1)) /
                                       config_t::block_size;
                     context_bid++) {
                    s_slice_start[2_dim] = context_bid * config_t::block_size;

                    auto valid_block_size =
                        ntt::min(config_t::block_size,
                                 seq_len - context_bid * config_t::block_size);

                    // [1, valid_block_size]
                    const auto s_slice =
                        s.view(s_slice_start, ntt::make_shape(dim_one, dim_one,
                                                              valid_block_size))
                            .squeeze(fixed_shape_v<dim_one>);

                    const auto v_block = kv_cache.template get_block<
                        caching::attention_cache_kind::value>(
                        layer_id, v_head_id,
                        kv_cache.get_block_id(seq_id, context_bid));

                    // [dim',valid_block_size]<dim>
                    const auto v_slice = v_block.view(
                        make_zeros_shape<2>(),
                        ntt::make_shape(v_block.shape()[0], valid_block_size));
                    // clang-format off
                    // [1, dim']<dim> = [1, valid_block_size] @ [dim',valid_block_size]<dim>
                    // clang-format on
                    if (context_bid == 0) {
                        ntt::matmul<false, false, true>(
                            s_slice, v_slice, d_slice, fixed_shape_v<>,
                            fixed_shape_v<>, fixed_shape_v<0>, fixed_shape_v<>);
                    } else {
                        ntt::matmul<true, false, true>(
                            s_slice, v_slice, d_slice, fixed_shape_v<>,
                            fixed_shape_v<>, fixed_shape_v<0>, fixed_shape_v<>);
                    }
                }
            }
        }
    }
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8>
void identity_paged_attention_kv_cache(
    [[maybe_unused]] T0 input, [[maybe_unused]] T1 num_seqs,
    [[maybe_unused]] T2 num_tokens, [[maybe_unused]] T3 context_lens,
    [[maybe_unused]] T4 seq_lens, [[maybe_unused]] T5 block_table,
    [[maybe_unused]] T6 slot_mapping, [[maybe_unused]] T7 num_blocks,
    [[maybe_unused]] T8 kv_caches) {
    // just extent the kv cache liveness.
}

template <ShardedTensor T0, class T1, class T2>
void gather_paged_attention_kv_cache([[maybe_unused]] const T0 &value,
                                     T1 &&kv_cache_tensor, T2 &&output_tensor) {
    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    using kv_cache_t = typename std::decay_t<decltype(kv_cache)>;
    using config_t = typename kv_cache_t::config_t;
    using mesh_type = T0::mesh_type;

    const auto local_index = mesh_type::local_index();
    const auto kv_cache_index =
        generate_shape<config_t::sharding_axes_t::rank()>([&](auto axis) {
            const auto submesh_axes =
                std::get<axis>(config_t::axis_policies).axes;
            const auto submesh_shape = mesh_type::shape.select(submesh_axes);
            const auto local_program_id =
                linear_offset(local_index.select(submesh_axes), submesh_shape);
            return local_program_id;
        });
    const auto kv_cache_address = kv_cache.kv_cache_address(kv_cache_index);
    const auto storage_tensor =
        make_tensor_view_from_address(kv_cache_address, output_tensor.shape());
    ntt::tensor_copy(storage_tensor, output_tensor);
}
} // namespace nncase::ntt
