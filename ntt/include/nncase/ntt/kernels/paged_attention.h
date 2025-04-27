#pragma once
#include <nncase/ntt/caching.h>
#include <nncase/ntt/shape.h>

namespace nncase::ntt {
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8>
void create_paged_attention_kv_cache(T0 num_seqs, T1 num_tokens,
                                     T2 context_lens, T3 seq_lens,
                                     T4 block_table, T5 slot_mapping,
                                     T6 num_blocks, T7 kv_caches, T8 output) {
    auto *kv_cache = output.elements().data();
    using paged_attention_kv_cache_t = typename T8::element_type;
    typename paged_attention_kv_cache_t::kv_tensor_type_t kv_tensor;

    using kv_type_t = typename paged_attention_kv_cache_t::kv_type_t;
    using kv_topo_t = typename paged_attention_kv_cache_t::config_t::kv_topo_t;

    using mesh_type = typename T7::mesh_type;
    auto program_ids = distributed::program_ids();
    distributed::topology_synchronize(); // sync for shard kv cache.

    apply(kv_topo_t{}, [&](auto program_index) {
        for (size_t i = 0; i < kv_topo_t::rank(); i++) {
            program_ids[kv_topo_t::at(i)] = program_index[i];
        }
        auto mesh_index = mesh_type::index_from_program_id(program_ids);
        auto remote = kv_caches.template remote<mesh_type::scope>(mesh_index);
        kv_tensor(program_index) = (intptr_t)remote.elements().data();
    });

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
}

template <class T0, class T1>
void update_paged_attention_kv_cache(
    [[maybe_unused]] T0 slot_tensor, T1 kv_cache_tensor,
    [[maybe_unused]] caching::attention_cache_kind kind,
    [[maybe_unused]] size_t layer_id) {
    auto kv_cache = kv_cache_tensor(0);
    auto local_slot = slot_tensor.local();
    using mesh_type = typename T0::mesh_type;
    // slots : [num_tokens -> x, numHeads -> [die,y], headDim]
    auto program_ids = distributed::program_ids();
    auto mesh_index = mesh_type::index_from_program_id(program_ids);
    auto head_id = mesh_index[3]; // note die head was broadcasting.
    auto token_id = mesh_index[2];

    // todo support core token nums > 1.
    auto slot_id = kv_cache.get_slot_id(token_id);
    auto slot = local_slot
                    .view(ntt::make_ranked_shape(0, 0, 0),
                          ntt::make_ranked_shape(1, 1, local_slot.shape()[2]))
                    .squeeze(ntt::fixed_shape<0, 1>{});
    kv_cache.update_slot(kind, layer_id, head_id, slot_id, slot);
    distributed::topology_synchronize();
}

template <class T0, class T1, class T2>
void paged_attention([[maybe_unused]] T0 q, [[maybe_unused]] T1 kv_cache_tensor,
                     [[maybe_unused]] size_t layer_id,
                     [[maybe_unused]] T2 output) {}

} // namespace nncase::ntt