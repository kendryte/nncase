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
#include "nncase/ntt/tensor_traits.h"
#include <nncase/ntt/caching.h>
#include <nncase/ntt/shape.h>
#include <nncase/ntt/sharding.h>
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
            auto mesh_index = mesh_type::index_from_program_id(program_ids);
            auto remote =
                kv_caches.template remote<mesh_type::scope>(mesh_index);
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

template <IsFixedDims TLayout, class TSlots, class TKVCache>
void update_paged_attention_kv_cache(TSlots slots_tensor,
                                     TKVCache kv_cache_tensor,
                                     caching::attention_cache_kind kind,
                                     size_t layer_id) {
    auto &kv_cache = kv_cache_tensor(0);
    using config_t = typename std::decay_t<decltype(kv_cache)>::config_t;
    constexpr size_t num_heads = config_t::num_kv_heads;

    if constexpr (IsShardedTensor<TSlots>) {
        auto local_slot = slots_tensor.local();
        using sharding_type = typename TSlots::sharding_type;
        using mesh_type = typename sharding_type::mesh_type;
        using axis_policy_type = typename sharding_type::axis_policy_type;
        using default_layout =
            fixed_shape<(size_t)caching::attention_dim_kind::seq,
                        (size_t)caching::attention_dim_kind::head,
                        (size_t)caching::attention_dim_kind::dim>;

        // slots : [seq, numHeads, headDim]
        auto program_ids = distributed::program_ids();
        auto mesh_index = mesh_type::index_from_program_id(program_ids);
        auto global_shape = slots_tensor.shape();
        auto local_shape = local_slot.shape();
        auto global_offset =
            sharding_type::global_offset(global_shape, mesh_index);
        constexpr size_t seq_index =
            TLayout::indexof((size_t)caching::attention_dim_kind::seq);
        constexpr size_t head_index =
            TLayout::indexof((size_t)caching::attention_dim_kind::head);
        constexpr size_t dim_index =
            TLayout::indexof((size_t)caching::attention_dim_kind::dim);

        auto starts = ntt::make_ranked_shape(0, 0, 0);
        auto shape = ntt::ranked_shape<3>();
        shape[seq_index] = 1;
        shape[head_index] = 1;
        shape[dim_index] = local_shape[dim_index];
        auto squeeze = ntt::fixed_shape<seq_index, head_index>();

        for (size_t token_id = 0; token_id < local_shape[seq_index];
             token_id++) {
            // slot mapping is broadcast, but slot maybe is sharding.
            auto slot_id =
                kv_cache.get_slot_id(global_offset[seq_index] + token_id);
            starts[seq_index] = token_id;

            for (size_t head_id = 0; head_id < local_shape[head_index];
                 head_id++) {

                // todo maybe slot head sharding != kv head sharding.
                starts[head_index] = head_id;

                auto slot = local_slot.view(starts, shape).squeeze(squeeze);
                kv_cache.update_slot(kind, layer_id, head_id, slot_id, slot);
            }
        }

        distributed::topology_synchronize();
    } else {
        for (size_t head_id = 0; head_id < num_heads; head_id++) {
            kv_cache.update_slots(kind, layer_id, head_id, slots_tensor);
        }
    }
}

template <IsFixedDims TLayout, class T0, class T1, class T2, class T3>
void paged_attention([[maybe_unused]] T0 q_tensor,
                     [[maybe_unused]] T1 kv_cache_tensor,
                     [[maybe_unused]] T2 extra_tensor,
                     [[maybe_unused]] size_t layer_id,
                     [[maybe_unused]] T3 output_tensor) {}

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

namespace detail {

template <class Mesh, class AxisPolicy, distributed::topology I,
          distributed::topology End>
struct FindProgramIdImpl {
    static constexpr size_t value =
        AxisPolicy::axes_type::at(0) ==
                distributed::detail::get_submesh_start<Mesh, I>()
            ? static_cast<size_t>(I)
            : FindProgramIdImpl<Mesh, AxisPolicy,
                                static_cast<distributed::topology>(
                                    static_cast<size_t>(I) + 1),
                                End>::value;
};

template <class Mesh, class AxisPolicy, distributed::topology End>
struct FindProgramIdImpl<Mesh, AxisPolicy, End, End> {
    static constexpr size_t value = static_cast<size_t>(-1);
};

template <class Mesh, class AxisPolicy> struct FindProgramId {
    static constexpr size_t value =
        FindProgramIdImpl<Mesh, AxisPolicy,
                          static_cast<distributed::topology>(0),
                          distributed::topology::count__>::value;
};

template <class Mesh, class AxisPolicy>
constexpr size_t program_id_in_axis_policy() {
    return FindProgramId<Mesh, AxisPolicy>::value;
}

template <class Mesh, class AxisPolicies, size_t... I>
constexpr auto program_ids_in_axis_policies(std::index_sequence<I...>) {
    return fixed_shape<program_id_in_axis_policy<
        Mesh, std::tuple_element_t<I, AxisPolicies>>()...>{};
}

}; // namespace detail

template <class Mesh, class AxisPolicies>
constexpr auto program_ids_in_axis_policies() {
    return detail::program_ids_in_axis_policies<Mesh, AxisPolicies>(
        std::make_index_sequence<std::tuple_size_v<AxisPolicies>>{});
}

template <class T0, class T1, class T2>
void gather_paged_attention_kv_cache([[maybe_unused]] T0 value,
                                     T1 kv_cache_tensor, T2 output_tensor) {
    auto &kv_cache = kv_cache_tensor(0);
    using kv_cache_t = typename std::decay_t<decltype(kv_cache)>;
    using config_t = typename kv_cache_t::config_t;
    using mesh_type = T0::mesh_type;
    constexpr size_t shardingRank = config_t::sharding_axes_t::rank();
    using axis_policies_t = typename config_t::axis_policies_t;

    auto program_ids = distributed::program_ids();
    auto program_indices =
        program_ids_in_axis_policies<mesh_type, axis_policies_t>();

    auto kv_indices = ntt::ranked_shape<shardingRank>();
    loop<shardingRank>([&](auto shard_id) {
        kv_indices[shard_id] = program_ids[program_indices[shard_id]];
    });

    auto storage_tensor = kv_cache.get_kv_storage_tensor(kv_indices);
    ntt::tensor_copy(storage_tensor, output_tensor);
}
} // namespace nncase::ntt