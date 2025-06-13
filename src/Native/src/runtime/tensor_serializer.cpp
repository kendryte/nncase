#include "nncase/llm/paged_attention_kv_cache.h"
#include "nncase/runtime/allocator.h"
#include "nncase/runtime/base64.h"
#include "nncase/runtime/runtime_op_utility.h"
#include "nncase/runtime/util.h"
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/tensor_serializer.h>

using namespace nncase;
using namespace nncase::runtime;

result<datatype_t>
runtime::deserialize_datatype(const nlohmann::json &json) noexcept {
    static const std::string TypeDiscriminator = "$type";

    if (!json.contains(TypeDiscriminator)) {
        return err(std::errc::invalid_argument);
    }
    auto type_name = json[TypeDiscriminator].get<std::string>();
    if (type_name == "PrimType") {
        auto type_code = static_cast<typecode_t>(json["TypeCode"].get<int>());
        return datatype_t::from_typecode(type_code);
    } else if (type_name == "PointerType") {
        try_var(elem_type, deserialize_datatype(json["ElemType"]));
        return ok(pointer_type_t(std::in_place, elem_type));
    } else if (type_name == "ReferenceType") {
        try_var(elem_type, deserialize_datatype(json["ElemType"]));
        return ok(reference_type_t(std::in_place, elem_type));
    } else if (type_name == "VectorType") {
        try_var(elem_type, deserialize_datatype(json["ElemType"]));
        auto lanes = json["Lanes"].get<dims_t>();
        return ok(vector_type_t(std::in_place, elem_type, lanes));
    } else if (type_name == "ValueType") {
        auto name = json["Name"].get<std::string>();
        if (name == "AttentionKVCacheType") {
            return ok(datatype_t::attention_kv_cache);
        } else if (name == "PagedAttentionKVCacheType") {
            return ok(datatype_t::paged_attention_kv_cache);
        } else {
            return err(std::errc::invalid_argument);
        }
    }

    return err(std::errc::invalid_argument);
}

result<llm::paged_attention_config> runtime::deserialize_paged_attention_config(
    const nlohmann::json &json) noexcept {
    auto num_layers = json["NumLayers"].get<int>();
    auto num_kv_heads = json["NumKVHeads"].get<int>();
    auto head_dim = json["HeadDim"].get<int>();
    auto block_size = json["BlockSize"].get<int>();
    auto kv_type_code = static_cast<typecode_t>(json["KVPrimType"].get<int>());

    auto cache_layout =
        json["CacheLayout"].get<std::array<llm::paged_kvcache_dim_kind, 6>>();

    auto packed_axes =
        json["PackedAxes"].get<std::vector<llm::paged_kvcache_dim_kind>>();

    auto lanes = json["Lanes"].get<dims_t>();

    auto sharding_axes =
        json["ShardingAxes"].get<std::vector<llm::paged_kvcache_dim_kind>>();

    auto axis_policies_json = json["AxisPolicies"];
    std::vector<dims_t> axis_policies;
    for (const auto &policy : axis_policies_json) {
        axis_policies.push_back(policy["Axes"].get<dims_t>());
    }

    auto config = llm::paged_attention_config(
        std::in_place, num_layers, num_kv_heads, head_dim, kv_type_code,
        block_size, cache_layout, packed_axes, lanes, sharding_axes,
        axis_policies);

    return ok(config);
}

result<intptr_t> runtime::deserialize_paged_attention_kv_cache(
    const nlohmann::json &json) noexcept {
    try_var(config, deserialize_paged_attention_config(json["Config"]));

    auto num_seqs = json["NumSeqs"].get<int>();
    auto num_tokens = json["NumTokens"].get<int>();

    try_var(context_lens, deserialize_tensor(json["ContextLens"]));
    try_var(seq_lens, deserialize_tensor(json["SeqLens"]));
    try_var(block_tables, deserialize_tensor(json["BlockTables"]));
    try_var(slot_mapping, deserialize_tensor(json["SlotMapping"]));
    try_var(kv_caches, deserialize_tensor(json["KVCaches"]));

    auto kv_caches_shape = kv_caches->shape();
    auto kv_caches_strides = kv_caches->strides();
    auto kv_shape =
        dims_t(kv_caches_shape.begin(),
               kv_caches_shape.begin() + config->sharding_axes().size());
    strides_t kv_strides(kv_shape.size());
    compute_strides(kv_shape, kv_strides);

    // assgin kv_storages
    auto num_kv_storages = compute_size(kv_shape);
    auto kv_cache_addrs =
        runtime::hrt::create(datatype_t::int64, {num_kv_storages})
            .unwrap_or_throw()
            .impl();
    auto mapped_kv_cache_addrs = kv_cache_addrs->buffer()
                                     .as_host()
                                     .unwrap_or_throw()
                                     .map(runtime::map_write)
                                     .unwrap_or_throw();
    auto kv_cache_addrs_data =
        reinterpret_cast<int64_t *>(mapped_kv_cache_addrs.buffer().data());
    {
        try_var(kv_caches_buffer, kv_caches->buffer().as_host());
        try_var(kv_caches_map_buffer,
                kv_caches_buffer.map(nncase::runtime::map_read));

        auto kv_caches_span = kv_caches_map_buffer.buffer();
        auto chunk_size = kv_caches_span.size_bytes() / num_kv_storages;

        for (size_t i = 0; i < num_kv_storages; i++) {
            auto kv_storage_span =
                kv_caches_span.subspan(i * chunk_size, chunk_size);
            auto kv_storage_shape =
                dims_t(kv_caches_shape.begin() + config->sharding_axes().size(),
                       kv_caches_shape.end());
            auto kv_storage_strides = dims_t(kv_caches_strides.begin() +
                                                 config->sharding_axes().size(),
                                             kv_caches_strides.end());
            try_var(kv_storage,
                    hrt::create(kv_caches->dtype(), kv_storage_shape,
                                kv_storage_strides, kv_storage_span, true));
            kv_cache_addrs_data[i] = (int64_t)kv_storage.impl()
                                         ->buffer()
                                         .as_host()
                                         .unwrap_or_throw()
                                         .map(nncase::runtime::map_read)
                                         .unwrap_or_throw()
                                         .buffer()
                                         .data();
            // FIXME: Memory leaks here
            kv_storage.impl().detach();
        }
    }

    // Create cache object
    auto cache = llm::paged_attention_kv_cache(
        std::in_place, num_seqs, num_tokens, context_lens, seq_lens,
        block_tables, slot_mapping, std::vector{kv_cache_addrs});
    return ok(reinterpret_cast<intptr_t>(cache.detach()));
}

result<intptr_t>
runtime::deserialize_reference(const nlohmann::json &json) noexcept {
    static const std::string TypeDiscriminator = "$type";

    if (!json.contains(TypeDiscriminator)) {
        return err(std::errc::invalid_argument);
    }

    auto type_name = json[TypeDiscriminator].get<std::string>();
    if (type_name != "Reference") {
        return err(std::errc::invalid_argument);
    }

    if (!json.contains("Value")) {
        return err(std::errc::invalid_argument);
    }

    auto ref_value = json["Value"]; // reference value
    auto ref_value_type = ref_value["$type"].get<std::string>();

    if (ref_value_type.find("PagedAttentionKVCache") != std::string::npos) {
        auto ref_value_value = ref_value["Value"];
        try_var(ref_value,
                deserialize_paged_attention_kv_cache(ref_value_value));
        return ok(ref_value);
    }
    // Add other reference type handlers here if needed

    return err(std::errc::invalid_argument);
}

result<tensor>
runtime::deserialize_tensor(const nlohmann::json &root) noexcept {
    try_var(element_type, deserialize_datatype(root["ElementType"]));

    auto dimensions = root["Dimensions"].get<dims_t>();

    auto strides = root["Strides"].get<strides_t>();

    auto buffer_json = root["Buffer"];
    if (element_type.is_a<reference_type_t>()) {
        // referenced buffer.
        auto elements = buffer_json.get<std::vector<nlohmann::json>>();
        auto element_size = elements.size();
        auto references = new intptr_t[element_size];
        for (auto ptr = references; const auto &element : elements) {
            try_var(ref_value, deserialize_reference(element));
            *(ptr++) = ref_value;
        }

        buffer_attach_options options{};
        options.deleter = [element_size](void *ptr) {
            auto obj_ptr = static_cast<tensor_node **>(ptr);
            for (size_t i = 0; i < element_size; i++) {
                if (nncase_object_release(obj_ptr[i]) <= 1) {
                    obj_ptr[i] = nullptr;
                }
            }
            delete[] obj_ptr;
        };
        try_var(buffer,
                buffer_allocator::host().attach(
                    as_span<std::byte>(std::span(references, element_size)),
                    options));

        auto tensor_result =
            tensor(std::in_place, element_type, dimensions, strides, buffer);
        return ok(tensor_result);
    } else if (element_type.is_a<value_type_t>()) {
        return err(std::errc::not_supported);
    } else {
        auto base64_str = buffer_json.get<std::string_view>();
        auto decoded_data =
            base64::decode_into<std::vector<std::byte>>(base64_str);

        try_var(tensor_result,
                hrt::create(element_type, dimensions,
                            std::span<std::byte>{decoded_data.data(),
                                                 decoded_data.size()},
                            true));
        return ok(tensor_result.impl());
    }
}