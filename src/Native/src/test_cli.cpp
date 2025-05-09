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
#include "nncase/runtime/util.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nncase/io_utils.h>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/runtime/base64.h>
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
constexpr size_t loop_count = 10;
// constexpr size_t loop_count = 1;

#define TRY(x)                                                                 \
    if (x)                                                                     \
        throw 1;

result<void> write_tensor_buffer(value_t value, std::ofstream &of) {
    try_var(v_tensor, value.as<tensor>());
    try_var(v_tensor_buffer, v_tensor->buffer().as_host());
    try_var(v_tensor_span, v_tensor_buffer.map(nncase::runtime::map_read));
    of.write(reinterpret_cast<const char *>(v_tensor_span.buffer().data()),
             v_tensor_span.buffer().size_bytes());
    return ok();
}

result<tensor> deserialize_tensor(const nlohmann::json &is);

result<datatype_t> deserialize_datatype(const nlohmann::json &json) {
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

result<llm::paged_attention_config>
deserialize_paged_attention_kv_cache_config(const nlohmann::json &json) {
    auto num_layers = json["NumLayers"].get<int>();
    auto num_kv_heads = json["NumKVHeads"].get<int>();
    auto head_dim = json["HeadDim"].get<int>();
    auto block_size = json["BlockSize"].get<int>();

    auto kv_type_code = static_cast<typecode_t>(json["KVType"].get<int>());

    auto cache_layout =
        json["CacheLayout"].get<std::array<llm::paged_attention_dim_kind, 6>>();
    auto packed_axes =
        json["PackedAxes"].get<std::vector<llm::paged_attention_dim_kind>>();
    auto lanes = json["Lanes"].get<dims_t>();
    auto topology = json["Topology"].get<dims_t>();

    // Create config object
    auto config = llm::paged_attention_config(
        std::in_place, num_layers, num_kv_heads, head_dim, kv_type_code,
        block_size, cache_layout, packed_axes, lanes, topology);

    return ok(config);
}

result<intptr_t>
deserialize_paged_attention_kv_cache(const nlohmann::json &json) {
    try_var(config,
            deserialize_paged_attention_kv_cache_config(json["Config"]));

    auto num_seqs = json["NumSeqs"].get<int>();
    auto num_tokens = json["NumTokens"].get<int>();
    auto num_blocks = json["NumBlocks"].get<int>();

    try_var(context_lens, deserialize_tensor(json["ContextLens"]));
    try_var(seq_lens, deserialize_tensor(json["SeqLens"]));
    try_var(block_table, deserialize_tensor(json["BlockTable"]));
    try_var(slot_mapping, deserialize_tensor(json["SlotMapping"]));
    try_var(kv_caches, deserialize_tensor(json["KVCaches"]));

    auto kv_caches_shape = kv_caches->shape();
    auto kv_caches_strides = kv_caches->strides();
    auto kv_shape = dims_t(kv_caches_shape.begin(),
                           kv_caches_shape.begin() + config->topology().size());
    strides_t kv_strides(kv_shape.size());
    compute_strides(kv_shape, kv_strides);

    // Create cache object
    auto cache = llm::paged_attention_kv_cache(
        std::in_place, config, num_seqs, num_tokens, context_lens, seq_lens,
        block_table, slot_mapping, num_blocks, kv_shape);

    // assgin kv_storages
    auto num_kv_storages = compute_size(kv_shape);
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
                dims_t(kv_caches_shape.begin() + config->topology().size(),
                       kv_caches_shape.end());
            auto kv_storage_strides =
                dims_t(kv_caches_strides.begin() + config->topology().size(),
                       kv_caches_strides.end());
            try_var(kv_storage,
                    hrt::create(kv_caches->dtype(), kv_storage_shape,
                                kv_storage_strides, kv_storage_span, true));
            cache->kv_cache(i, kv_storage.impl());
        }
    }

    return ok(reinterpret_cast<intptr_t>(cache.detach()));
}

result<intptr_t> deserialize_reference(const nlohmann::json &json) {
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

result<tensor> deserialize_tensor(const nlohmann::json &root) {
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
                delete obj_ptr[i];
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

result<void> run_core(const std::string &kmodel_path,
                      const std::vector<std::string> &bins) {
    std::ifstream kmodel(kmodel_path, std::ios::binary | std::ios::in);
    if (!kmodel.is_open())
        return err(std::errc::no_such_file_or_directory);
    runtime::std_istream stream(kmodel);
    interpreter interp;
    // auto dump_path =
    //     std::filesystem::path(arg_file_path).parent_path().string();
    // nncase_interp_set_dump_root(interp, dump_path.c_str());
    try_(interp.load_model(stream));

    try_var(entry, interp.entry_function());

    if (entry->parameters_size() > bins.size())
        return err(std::errc::argument_list_too_long);
    /* create the input parameters tensor
       note the input tenosr must be contiguous
    */
    std::vector<value_t> parameters;
    for (int i = 0; i < entry->parameters_size(); i++) {
        if (bins[i].ends_with(".json")) {
            std::ifstream json_file(bins[i]);
            nlohmann::json json_data = nlohmann::json::parse(json_file);
            try_var(ts, deserialize_tensor(json_data));
            parameters.push_back(ts);
        } else {
            try_var(type, entry->parameter_type(i));
            try_var(ts_type, type.as<tensor_type>());
            auto input_pool = read_file(bins[i]);
            std::span<std::byte> input_pool_span = {
                reinterpret_cast<std::byte *>(input_pool.data()),
                input_pool.size()};
            try_var(dims, ts_type->shape().as_fixed());
            try_var(_,
                    hrt::create(ts_type->dtype(), dims, input_pool_span, true));
            parameters.push_back(_.impl());
        }
    }

    // warm up
    try_var(ret, entry->invoke({parameters.data(), parameters.size()}));
    double total_time = 0.0;
    for (size_t i = 0; i < loop_count; i++) {
        auto start_time = std::chrono::steady_clock::now();
        try_var(ret, entry->invoke({parameters.data(), parameters.size()}));
        auto end_time = std::chrono::steady_clock::now();
        total_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count() /
                       1e6);

        if (i == (loop_count - 1) && (entry->parameters_size() < bins.size())) {
            if (ret.is_a<tensor>()) {
                auto output_bin = bins.back();
                std::ofstream output_stream(output_bin, std::ios::binary);
                try_(write_tensor_buffer(ret, output_stream));
                output_stream.close();
            } else if (ret.is_a<tuple>()) {
                try_var(tp, ret.as<tuple>());
                auto o = 0;
                for (auto &&ret_v : tp->fields()) {
                    auto output_bin = bins[entry->parameters_size() + (o++)];
                    std::ofstream output_stream(output_bin, std::ios::binary);
                    try_(write_tensor_buffer(ret_v, output_stream));
                    output_stream.close();
                }
            } else {
                return nncase::err(std::errc::bad_message);
            }
        }
    }

    std::cout << "interp run: " << (total_time / loop_count)
              << " ms, fps = " << 1000 / (total_time / loop_count) << std::endl;

    return ok();
}

/**
 * @brief interp cli for the test.
 *  interp_cli kmodel_path input0 ... inputN output
 * @param argc
 * @param argv
 * @return int
 */
int main(NNCASE_UNUSED int argc, char **argv) {
    assert(argc >= 3);
    std::vector<std::string> bins;
    for (int i = 2; i < argc; i++) {
        bins.push_back(argv[i]);
    }
    std::string kmodel_bin(argv[1]);
    run_core(kmodel_bin, bins).unwrap_or_throw();
    return 0;
}