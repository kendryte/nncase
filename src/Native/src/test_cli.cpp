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
#include "cxxopts.hpp"
#include "nncase/runtime/error.h"
#include "nncase/runtime/util.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nncase/io_utils.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/tensor_serializer.h>

using namespace nncase;
using namespace nncase::runtime;

result<bool> compare_datatype(datatype_t actual, datatype_t expect) {
    if (actual.is_a<prim_type_t>() && expect.is_a<prim_type_t>()) {
        try_var(actual_prim, actual.as<prim_type_t>());
        try_var(expect_prim, expect.as<prim_type_t>());
        return ok(actual_prim->typecode() == expect_prim->typecode());
    } else if (actual.is_a<vector_type_t>() && expect.is_a<vector_type_t>()) {
        try_var(actual_vec, actual.as<vector_type_t>());
        try_var(expect_vec, expect.as<vector_type_t>());
        try_var(result, compare_datatype(actual_vec->elemtype(),
                                         expect_vec->elemtype()));
        return ok(result && (actual_vec->lanes() == expect_vec->lanes()));
    } else if (actual.is_a<pointer_type_t>() && expect.is_a<pointer_type_t>()) {
        try_var(actual_ptr, actual.as<pointer_type_t>());
        try_var(expect_ptr, expect.as<pointer_type_t>());
        return compare_datatype(actual_ptr->elemtype(), expect_ptr->elemtype());
    } else if (actual.is_a<reference_type_t>() &&
               expect.is_a<reference_type_t>()) {
        try_var(actual_ref, actual.as<reference_type_t>());
        try_var(expect_ref, expect.as<reference_type_t>());
        return compare_datatype(actual_ref->elemtype(), expect_ref->elemtype());
    } else if (actual.is_a<value_type_t>() && expect.is_a<value_type_t>()) {
        try_var(actual_val, actual.as<value_type_t>());
        try_var(expect_val, expect.as<value_type_t>());
        return ok(actual_val->uuid() == expect_val->uuid() &&
                  actual_val->size_bytes() == expect_val->size_bytes());
    } else {
        return ok(false);
    }
}

result<typecode_t> get_prim_typecode(datatype_t dtype) {
    if (dtype.is_a<prim_type_t>()) {
        try_var(prim_type, dtype.as<prim_type_t>());
        return ok(prim_type->typecode());
    } else if (dtype.is_a<vector_type_t>()) {
        try_var(vec_type, dtype.as<vector_type_t>());
        return get_prim_typecode(vec_type->elemtype());
    }

    return err(std::errc::invalid_argument);
}

template <typename T> float dot(const T *v1, const T *v2, size_t size) {
    float ret = 0.f;
    for (size_t i = 0; i < size; i++) {
        ret += v1[i] * v2[i];
    }
    return ret;
}

template <typename T> float cosine(std::span<T> actual, std::span<T> expect) {
    std::cout << "actual data size: " << actual.size()
              << " expect data size: " << expect.size() << std::endl;
    std::cout << "Comparation of first 10 (actual, golden):" << std::endl;
    for (size_t i = 0; i < 10; i++) {
        std::cout << "index[" << i << "]: " << actual[i] << " " << expect[i]
                  << std::endl;
    }
    float d0 = dot(actual.data(), expect.data(), actual.size());
    float d1 = dot(actual.data(), actual.data(), actual.size());
    float d2 = dot(expect.data(), expect.data(), actual.size());
    return d0 / sqrtf(d1 * d2);
}

#define COS_DISPATCH(dt_name, type)                                            \
    case typecode_t::dt_name:                                                  \
        cos = cosine(as_span<type>(actual_span), as_span<type>(expect_span));  \
        break

result<bool> compare_tensor(tensor actual, tensor expect) {
    if (actual->shape() != expect->shape()) {
        dbg("shape not match!");
        dbg("actual shape: ", actual->shape());
        dbg("expect shape: ", expect->shape());
        return ok(false);
    }

    try_var(datatype_compared,
            compare_datatype(actual->dtype(), expect->dtype()));

    if (!datatype_compared) {
        dbg("datatype not match!");
        return ok(false);
    }

    try_var(prim_typecode, get_prim_typecode(actual->dtype()));

    try_var(actual_buffer, actual->buffer().as_host());
    try_var(expect_buffer, expect->buffer().as_host());
    try_var(actual_map, actual_buffer.map(nncase::runtime::map_read));
    try_var(expect_map, expect_buffer.map(nncase::runtime::map_read));
    auto actual_span = actual_map.buffer();
    auto expect_span = expect_map.buffer();
    if (actual_span.size_bytes() != expect_span.size_bytes()) {
        printf("size bytes not match: actual %zu, expect %zu\n",
               actual_span.size_bytes(), expect_span.size_bytes());
        return ok(false);
    }
    float cos = 0.0;
    switch (prim_typecode) {
        COS_DISPATCH(dt_int8, int8_t);
        COS_DISPATCH(dt_int16, int16_t);
        COS_DISPATCH(dt_int32, int32_t);
        COS_DISPATCH(dt_int64, int64_t);
        COS_DISPATCH(dt_uint8, uint8_t);
        COS_DISPATCH(dt_uint16, uint16_t);
        COS_DISPATCH(dt_uint32, uint32_t);
        COS_DISPATCH(dt_uint64, uint64_t);
        COS_DISPATCH(dt_float16, half);
        COS_DISPATCH(dt_float32, float);
        COS_DISPATCH(dt_float64, double);
        COS_DISPATCH(dt_bfloat16, bfloat16);
        COS_DISPATCH(dt_float8e4m3, float_e4m3_t);
        COS_DISPATCH(dt_float8e5m2, float_e5m2_t);
    default:
        break;
    };
    std::cout << "cosine similarity: " << cos << std::endl;
    return ok(cos >= 0.999);
}

#undef COS_DISPATCH

result<void> write_tensor_buffer(value_t value, std::ofstream &of) {
    auto v_tensor = value.as<tensor>().expect("not a tensor");
    try_var(v_tensor_buffer, v_tensor->buffer().as_host());
    try_var(v_tensor_span, v_tensor_buffer.map(nncase::runtime::map_read));
    of.write(reinterpret_cast<const char *>(v_tensor_span.buffer().data()),
             v_tensor_span.buffer().size_bytes());
    return ok();
}

result<void> run_core(const std::string &kmodel_path,
                      const std::vector<std::string> &files, size_t loop_count,
                      bool warmup) {
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

    if (entry->parameters_size() > files.size())
        return err(std::errc::argument_list_too_long);
    /* create the input parameters tensor
       note the input tenosr must be contiguous
    */
    std::vector<value_t> parameters;
    for (int i = 0; i < entry->parameters_size(); i++) {
        if (files[i].ends_with(".json")) {
            std::ifstream json_file(files[i]);
            const nlohmann::json &json_data = nlohmann::json::parse(json_file);
            try_var(ts, deserialize_tensor(json_data));
            parameters.push_back(ts);
        } else {
            try_var(type, entry->parameter_type(i));
            try_var(ts_type, type.as<tensor_type>());
            auto input_pool = read_file(files[i]);
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
    if (warmup) {
        try_var(ret, entry->invoke({parameters.data(), parameters.size()}));
    }

    double total_time = 0.0;
    for (size_t i = 0; i < loop_count; i++) {
        auto start_time = std::chrono::steady_clock::now();
        try_var(ret, entry->invoke({parameters.data(), parameters.size()}));
        auto end_time = std::chrono::steady_clock::now();
        total_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count() /
                       1e6);

        if (i == (loop_count - 1) && (parameters.size() < files.size())) {
            auto output_file = files[parameters.size()];
            if (std::filesystem::exists(output_file)) {
                // try compare
                if (output_file.ends_with(".json")) {
                    std::ifstream json_file(output_file);
                    const nlohmann::json &json_data =
                        nlohmann::json::parse(json_file);
                    try_var(ts, deserialize_tensor(json_data));
                    auto ret_tensor = ret.as<tensor>().expect("not a tensor");
                    try_(compare_tensor(ret_tensor, ts));
                } else {
                    std::vector<tensor> ret_tensors;
                    if (ret.is_a<tensor>()) {
                        try_var(t, ret.as<tensor>());
                        ret_tensors.push_back(t);
                    } else {
                        try_var(tp, ret.as<tuple>());
                        for (size_t i = 0; i < tp->fields().size(); i++) {
                            try_var(t, tp->fields()[i].as<tensor>());
                            ret_tensors.push_back(t);
                        }
                    }

                    for (size_t o = 0; o < ret_tensors.size(); o++) {
                        auto ret_tensor = ret_tensors[o];
                        auto output_pool =
                            read_file(files[parameters.size() + o]);
                        std::span<std::byte> output_pool_span = {
                            reinterpret_cast<std::byte *>(output_pool.data()),
                            output_pool.size()};
                        try_var(_, hrt::create(ret_tensor->dtype(),
                                               dims_t(ret_tensor->shape()),
                                               output_pool_span, true));
                        try_(compare_tensor(ret_tensor, _.impl()));
                    }
                }
            } else {
                std::ofstream output_stream(output_file, std::ios::binary);
                try_(write_tensor_buffer(ret, output_stream));
                output_stream.close();
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
int main(int argc, char **argv) {
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__
              << std::endl;

    cxxopts::Options options("nncase-interp", "NNCASE interpreter CLI tool");

    // clang-format off
    options.add_options()
      ("k,kmodel", "Path to kmodel file", cxxopts::value<std::string>())
      ("i,inputs", "Input files (can be multiple), Output files (can be multiple)", cxxopts::value<std::vector<std::string>>())
      ("l,loop", "Number of inference iterations", cxxopts::value<size_t>()->default_value("1"))
      ("w,warmup", "Enable warmup before inference", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
      ("h,help", "Print usage");
    // clang-format on

    options.positional_help("kmodel_path input0 ... inputN output");
    options.parse_positional({"kmodel", "inputs"});

    auto result = options.parse(argc, argv);

    if (result.count("help") || argc < 3) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("kmodel") || !result.count("inputs")) {
        std::cout << "Error: kmodel and input files are required\n"
                  << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    std::string kmodel_bin = result["kmodel"].as<std::string>();
    std::vector<std::string> bins =
        result["inputs"].as<std::vector<std::string>>();
    size_t loop_count = result["loop"].as<size_t>();
    bool warmup = result["warmup"].as<bool>();

    run_core(kmodel_bin, bins, loop_count, warmup).unwrap_or_throw();
    return 0;
}