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
#include <cstring>
#include <iostream>
#include <nncase/io_utils.h>
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;

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

result<void> run_core(const std::string &arg_file_path) {
    std::ifstream arg_file(arg_file_path);
    std::string kmodel_path, input_pool_path, output_pool_path;

    std::getline(arg_file, kmodel_path);
    std::getline(arg_file, input_pool_path);
    std::getline(arg_file, output_pool_path);

    auto input_pool = read_file(input_pool_path);
    std::span<std::byte> input_pool_span = {
        reinterpret_cast<std::byte *>(input_pool.data()), input_pool.size()};
    /* create the input parameters tensor */
    std::vector<value_t> parameters;
    int input_nums;
    arg_file >> input_nums;
    for (int i = 0; i < input_nums; i++) {
        // datatype_t dtype;
        int32_t type_code;
        arg_file >> type_code;
        int32_t rank;
        arg_file >> rank;
        dims_t dims;
        for (size_t i = 0; i < rank; i++) {
            int32_t dim;
            arg_file >> dim;
            dims.push_back(dim);
        }
        size_t start, size;
        arg_file >> start >> size;
        try_var(_, hrt::create((typecode_t)type_code, dims,
                               input_pool_span.subspan(start, size), false));
        parameters.push_back(_.impl());
    }

    auto kmodel = read_file(kmodel_path);
    interpreter *interp = new interpreter();
    auto dump_path =
        std::filesystem::path(arg_file_path).parent_path().string();
    nncase_interp_set_dump_root(interp, dump_path.c_str());
    try_(interp->load_model(
        {reinterpret_cast<const std::byte *>(kmodel.data()), kmodel.size()},
        false));

    try_var(entry, interp->entry_function());

    try_var(ret, entry->invoke({parameters.data(), parameters.size()}));

    std::ofstream output_bin(output_pool_path, std::ios::binary);

    if (ret.is_a<tensor>()) {
        try_(write_tensor_buffer(ret, output_bin));
    } else if (ret.is_a<tuple>()) {
        try_var(tp, ret.as<tuple>());
        for (auto &&ret_v : tp->fields()) {
            try_(write_tensor_buffer(ret_v, output_bin));
        }
    } else {
        return nncase::err(std::errc::bad_message);
    }
    output_bin.close();
    return ok();
}

/**
 * @brief interp cli for the test.
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(NNCASE_UNUSED int argc, char **argv) {
    assert(argc == 2);
    run_core(argv[1]).unwrap_or_throw();
    return 0;
}