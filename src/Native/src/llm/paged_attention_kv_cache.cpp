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

#include <algorithm>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/ntt/ntt.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;

namespace detail {

template <size_t Rank, typename T>
inline auto unpack_vector(const std::vector<T> &vec) {
    return std::apply(
        [&vec](auto &&...indices) { return std::make_tuple(vec[indices]...); },
        std::make_index_sequence<Rank>{});
}

datatype_t get_kv_storage_type(const llm::paged_attention_kv_cache_node &node) {
    auto elemtype = node.config()->kv_type();
    auto packed_axes = node.config()->packed_axes();
    return vector_type_t(
        std::in_place, elemtype,
        dims_t{node.config()->lanes().begin(), node.config()->lanes().end()});
}

template <typename T>
ntt::tensor_view<T, ntt::ranked_shape<2>>
get_block_view_from_storage(const llm::paged_attention_kv_cache_node &node,
                            llm::attention_cache_kind kind, int layer_id,
                            int head_id, nncase::tensor block_id) {
    auto block_id_span = as_span<int64_t>(get_input_span(block_id).unwrap());

    // get the kv cache storage.
    ntt::tensor_view<T, ntt::ranked_shape<6>> kv_cache_storage(
        std::span<T>((T *)nullptr, 0), ntt::ranked_shape<6>{},
        ntt::ranked_strides<6>{});
    ntt::ranked_shape<6> kv_cache_shape;
    auto cache_layout = node.config()->cache_layout();
    std::array<size_t, 6> default_kv_cache_shape = {
        node.config()->num_blocks(),
        node.config()->num_layers(),
        2,
        node.config()->block_size(),
        node.config()->num_kv_heads(),
        node.config()->head_dim()};
    for (size_t i = 0; i < 6; i++) {
        kv_cache_shape[i] = default_kv_cache_shape[(int)cache_layout[i]];
    }

    // process pack
    for (size_t i = 0; i < node.config()->packed_axes().size(); i++) {
        auto axis = node.config()->packed_axes()[i];
        kv_cache_shape[(size_t)axis] /= node.config()->lanes()[i];
    }

    auto kv_caches = node.kv_caches();
    if (*block_id->shape().end() == 1) {
        auto kv_cache_span = get_input_span(kv_caches).unwrap();
        kv_cache_storage = ntt::tensor_view<T, ntt::ranked_shape<6>>(
            as_span<T>(kv_cache_span), kv_cache_shape); // without distributed.
    } else {
        if (kv_caches->dtype()->typecode() != dt_pointer) {
            throw std::runtime_error("the kv_caches should be distributed.");
        }
        auto kv_cache_span =
            as_span<intptr_t>(get_input_span(kv_caches).unwrap());
        std::span<const size_t> index = {(size_t *)block_id_span.data(),
                                         (*kv_caches->shape().end() - 1)};
        auto storage_ptr = reinterpret_cast<T *>(
            kv_cache_span[kernels::linear_index(kv_caches->shape(), index)]);

        kv_cache_storage = ntt::tensor_view<T, ntt::ranked_shape<6>>(
            std::span<T>(storage_ptr,
                         ntt::linear_size(kv_cache_shape, ntt::default_strides(
                                                              kv_cache_shape))),
            kv_cache_shape);
    }

    std::array<size_t, 6> dafault_block_start = {
        (size_t)block_id_span[block_id_span.size() - 1],
        (size_t)layer_id,
        (size_t)kind,
        0,
        (size_t)head_id,
        0};

    std::array<size_t, 6> default_block_shape = {
        1, 1,
        1, kv_cache_shape[(size_t)llm::paged_attention_dim_kind::block_size],
        1, kv_cache_shape[(size_t)llm::paged_attention_dim_kind::head_dim]};

    ntt::ranked_shape<6> starts;
    for (size_t i = 0; i < 6; i++) {
        starts[i] = dafault_block_start[(size_t)cache_layout[i]];
    }
    ntt::ranked_shape<6> shape;
    for (size_t i = 0; i < 6; i++) {
        shape[i] = default_block_shape[(size_t)cache_layout[i]];
    }

    auto squeezeAxes = ntt::ranked_shape<4>();
    {
        size_t j = 0;
        for (size_t i = 0; i < 4; i++) {
            if (cache_layout[i] != llm::paged_attention_dim_kind::block_size &&
                cache_layout[i] != llm::paged_attention_dim_kind::head_dim) {
                squeezeAxes[i] = j++;
            }
        }
    }
    return kv_cache_storage.view(starts, shape).squeeze(squeezeAxes);
}

template <typename T>
ntt::tensor_view<T, ntt::ranked_shape<1>>
get_slot_view_from_storage(const llm::paged_attention_kv_cache_node &node,
                           llm::attention_cache_kind kind, int layer_id,
                           int head_id, const nncase::tensor &slot_id) {
    try_input_with_ty(slot_id_value, slot_id, int64_t);
    auto id_size = slot_id->shape().size();
    auto block_id = slot_id_value[id_size - 1] / node.config()->block_size();
    auto block_offset =
        slot_id_value[id_size - 1] % node.config()->block_size();
    auto block_view =
        get_block_view_from_storage<T>(node, kind, layer_id, head_id, block_id);
    auto block_shape = block_view.shape();
    auto block_layout = node.config()->block_layout();

    // PagedAttentionDimKind[] default_layout =
    // [PagedAttentionDimKind.BlockSize, PagedAttentionDimKind.HeadDim];
    std::array<size_t, 6> default_layout = {7, 7, 7, 0, 7, 1};
    std::array<size_t, 2> default_starts = {block_offset, 0};
    std::array<size_t, 2> default_shape = {1, node.config()->head_dim()};
    if (auto p = std::find(node.config()->packed_axes().begin(),
                           node.config()->packed_axes().end(),
                           llm::paged_attention_dim_kind::head_dim);
        p != node.config()->packed_axes().end()) {
        default_shape[1] /=
            node.config()->lanes()[p - node.config()->packed_axes().begin()];
    }

    ntt::ranked_shape<2> starts;
    for (size_t i = 0; i < 2; i++) {
        starts[i] = default_starts[default_layout[(int)block_layout[i]]];
    }

    ntt::ranked_shape<2> shape;
    for (size_t i = 0; i < 2; i++) {
        shape[i] = default_shape[default_layout[(int)block_layout[i]]];
    }

    return block_view.view(starts, shape)
        .squeeze(ntt::make_ranked_shape(
            default_layout[(int)llm::paged_attention_dim_kind::block_size]));
}

}; // namespace detail

llm::paged_attention_kv_cache_node::paged_attention_kv_cache_node(
    paged_attention_config config, size_t num_seqs, size_t num_tokens,
    tensor context_lens, tensor seq_lens, tensor block_tables,
    tensor slot_mapping, tensor kv_caches) noexcept
    : attention_kv_cache_node(std::move(config), num_seqs, num_tokens,
                              context_lens, seq_lens),
      block_tables_(std::move(block_tables)),
      slot_mapping_(std::move(slot_mapping)),
      kv_caches_(std::move(kv_caches)) {}

nncase::tensor
llm::paged_attention_kv_cache_node::get_block_ids(int seq_id) const {
    auto block_table_buffer =
        block_tables_->buffer().as_host().unwrap_or_throw();
    auto mapped_block_table_buffer =
        block_table_buffer.map(runtime::map_read).unwrap_or_throw();
    auto stride = (*block_tables_->shape().rbegin()) * sizeof(int64_t);

    return hrt::create(dt_int64, {*block_tables_->shape().rbegin()},
                       mapped_block_table_buffer.buffer().subspan(
                           seq_id * stride, stride),
                       false)
        .unwrap()
        .impl();
}

nncase::tensor llm::paged_attention_kv_cache_node::get_slot_ids() const {
    return slot_mapping_;
}

nncase::tensor llm::paged_attention_kv_cache_node::get_block(
    llm::attention_cache_kind kind, int layer_id, int head_id,
    const nncase::tensor &block_id) const {
    auto kv_type = ::detail::get_kv_storage_type(*this);
    if (kv_type.is_a<vector_type_t>()) {
        auto vector_kv_type = kv_type.as<vector_type_t>().unwrap();
        auto elem_type = vector_kv_type->elemtype();
        if (vector_kv_type->elemtype().is_a<prim_type_t>()) {
            switch (elem_type->typecode()) {
            case dt_float32:
                if (vector_kv_type->lanes() == dims_t({32})) {
                    auto block_view = ::detail::get_block_view_from_storage<
                        ntt::vector<float, 32>>(*this, kind, layer_id, head_id,
                                                block_id);
                    return hrt::create(
                               vector_kv_type->typecode(),
                               {block_view.shape()[0], block_view.shape()[1]},
                               {block_view.strides()[0],
                                block_view.strides()[1]},
                               as_span<std::byte>(block_view.buffer()), false)
                        .unwrap()
                        .impl();
                    break;
                } else {
                    goto fail;
                }
            default:
                goto fail;
                break;
            }
        }
    } else {
        goto fail;
    }
fail:
    throw std::runtime_error("Not supported");
}

void llm::paged_attention_kv_cache_node::update_block(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &block_id,
    NNCASE_UNUSED const nncase::tensor &block) {
    // Placeholder implementation, you should add real logic here
    throw std::runtime_error("Not implemented");
}

nncase::tensor llm::paged_attention_kv_cache_node::get_slot(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id,
    NNCASE_UNUSED const nncase::tensor &slot_id) const {
    // Placeholder implementation, you should add real logic here
    throw std::runtime_error("Not implemented");
}

void llm::paged_attention_kv_cache_node::update_slot(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &slot_id,
    NNCASE_UNUSED const nncase::tensor &slot) {}

void llm::paged_attention_kv_cache_node::update_slots(
    NNCASE_UNUSED llm::attention_cache_kind kind, NNCASE_UNUSED int layer_id,
    NNCASE_UNUSED int head_id, NNCASE_UNUSED const nncase::tensor &slot_ids,
    NNCASE_UNUSED const nncase::tensor &slots) {}

/*

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
} */