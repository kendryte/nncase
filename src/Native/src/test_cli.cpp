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
#include <chrono>
#include <cstring>
#include <iostream>
#include <nncase/io_utils.h>
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
// constexpr size_t loop_count = 10;
constexpr size_t loop_count = 1;

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

result<void> run_core(const std::string &kmodel_path,
                      const std::vector<std::string> &bins) {
    auto kmodel = read_file(kmodel_path);
    interpreter *interp = new interpreter();
    // auto dump_path =
    //     std::filesystem::path(arg_file_path).parent_path().string();
    // nncase_interp_set_dump_root(interp, dump_path.c_str());
    try_(interp->load_model(
        {reinterpret_cast<const gsl::byte *>(kmodel.data()), kmodel.size()},
        false));

    try_var(entry, interp->entry_function());

    if (entry->parameters_size() > bins.size())
        return err(std::errc::argument_list_too_long);
    /* create the input parameters tensor
       note the input tenosr must be contiguous
    */
    std::vector<value_t> parameters;
    for (int i = 0; i < entry->parameters_size(); i++) {
        try_var(type, entry->parameter_type(i));
        try_var(ts_type, type.as<tensor_type>());
        auto input_pool = read_file(bins[i]);
        gsl::span<gsl::byte> input_pool_span = {
            reinterpret_cast<gsl::byte *>(input_pool.data()),
            input_pool.size()};
        try_var(dims, ts_type->shape().as_fixed());
        try_var(_, hrt::create(ts_type->dtype()->typecode(), dims,
                               input_pool_span, true));
        parameters.push_back(_.impl());
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

        if (i == (loop_count - 1)) {
            if (entry->parameters_size() == (bins.size() - 1)) {
                auto output_bin = bins.back();
                std::ofstream output_stream(output_bin, std::ios::binary);
                if (ret.is_a<tensor>()) {
                    try_(write_tensor_buffer(ret, output_stream));
                } else if (ret.is_a<tuple>()) {
                    try_var(tp, ret.as<tuple>());
                    for (auto &&ret_v : tp->fields()) {
                        try_(write_tensor_buffer(ret_v, output_stream));
                    }
                } else {
                    return nncase::err(std::errc::bad_message);
                }
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