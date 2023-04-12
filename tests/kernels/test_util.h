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
#include "nncase/runtime/simple_types.h"
#include "nncase/shape.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <random>
#include <string>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

enum OpType { Opt, Ref };

void print_index(const dims_t &index) {
    for (size_t i = 0; i < index.size(); ++i) {
        std::cout << index[i] << " ";
    }
    std::cout << std::endl;
}

size_t get_last_no_zero_stride(const strides_t &strides, size_t i) {
    for (size_t j = i; j < strides.size(); ++j) {
        if (strides[j] != 0) {
            return strides[j];
        }
    }
    assert(false);
    return 0;
}

// strides bias first value is no effect
// line is contiguous
dims_t get_strides(const dims_t &shape, const strides_t &strides_bias) {
    dims_t strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) {
        const auto line_width = shape[i + 1] + strides_bias[i + 1];
        const auto shape_v = line_width == 1 ? 0 : line_width;
        const auto strides_v = strides[i + 1] == 0
                                   ? get_last_no_zero_stride(strides, i + 1)
                                   : strides[i + 1];
        strides[i] = strides_v * shape_v;
    }
    return strides;
}

uint32_t &get(runtime_tensor &t, const dims_t &index) {
    auto map = std::move(hrt::map(t, hrt::map_read).unwrap_or_throw());
    auto data = map.buffer().as_span<uint32_t>();
    return data[offset(t.strides(), index)];
}

void init_tensor_data(runtime_tensor &tensor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> dis(3, 333);
    // auto *ptr = tensor.data_as<uint32_t>();
    // auto ptr =
    // reinterpret_cast<uint32_t*>(host_runtime_tensor::buffer(tensor).unwrap().begin());
    NNCASE_UNUSED auto res = cpu::reference::apply(
        tensor.shape(), [&](const dims_t &index) -> result<void> {
            // print_index(index);
            // ptr[offset(tensor.strides(), index)] = dis(gen);
            get(tensor, index) = dis(gen);
            //  std::cout << "tensor:" << get(tensor, index) << std::endl;
            return ok();
        });
}

runtime_tensor create_tensor(const dims_t &shape,
                             const strides_t &strides_bias) {
    auto strides = get_strides(shape, strides_bias);
    // std::make_shared<uint32_t>(compute_size(shape, strides))
    return host_runtime_tensor::create(dt_float32, shape, strides).unwrap();
}

runtime_tensor create_input_tensor(const dims_t &shape,
                                   const strides_t &strides_bias) {
    auto tensor = create_tensor(shape, strides_bias);
    init_tensor_data(tensor);
    return tensor;
}

const gsl::byte *get_tensor_cbegin(runtime_tensor &t) {
    auto map = hrt::map(t, hrt::map_read).unwrap_or_throw();
    return map.buffer().cbegin();
}

gsl::byte *get_tensor_begin(runtime_tensor &t) {
    auto map = hrt::map(t, hrt::map_read).unwrap_or_throw();
    return map.buffer().begin();
}

runtime_shape_t shape_sub(dims_t begin, dims_t end) {
    dims_t v;
    for (size_t i = 0; i < begin.size(); ++i) {
        v.push_back(end[i] - begin[i]);
    }
    return v;
}

bool is_same_tensor(runtime_tensor &lhs, runtime_tensor &rhs) {
    if (lhs.shape() != rhs.shape()) {
        return false;
    }
    return cpu::reference::apply(lhs.shape(),
                                 [&](const dims_t &index) -> result<void> {
                                     // print_index(index);
                                     // std::cout << get(index) << " " <<
                                     // rhs.get(index) << std::endl;
                                     if (get(lhs, index) == get(rhs, index)) {
                                         return ok();
                                     } else {
                                         return err(std::errc::not_supported);
                                     }
                                 })
        .is_ok();
}

void print_data(runtime_tensor &data) {
    NNCASE_UNUSED auto res = cpu::reference::apply(
        data.shape(), [&](const dims_t &index) -> result<void> {
            std::cout << get(data, index) << std::endl;
            return ok();
        });
}

namespace fs = std::filesystem;
bool exist_directory(const fs::path &path) {
    std::error_code error;
    auto file_status = std::filesystem::status(path, error);
    if (error) {
        return false;
    }
    return fs::exists(file_status) && fs::is_directory(file_status);
}

void output_data(runtime_tensor &data, std::string name,
                 std::string dir_name = "op_test") {
    if (!exist_directory(dir_name) && dir_name != "") {
        fs::create_directory(dir_name);
    }
    std::ofstream f(dir_name + "/" + name + ".txt");
    // output shape
    auto shape_str = std::accumulate(
        data.shape().begin(), data.shape().end(), std::string(),
        [](std::string s, auto v) { return s + std::to_string(v) + ","; });
    shape_str.pop_back();
    f << "shape(" << shape_str << ")" << std::endl;
    NNCASE_UNUSED auto res = cpu::reference::apply(
        data.shape(), [&](const dims_t &index) -> result<void> {
            f << get(data, index) << std::endl;
            return ok();
        });
    f.close();
}

constexpr auto output_root = "op_test";

void make_output_root_dir(std::string root = output_root) {
    if (exist_directory(root)) {
        fs::remove_all(root);
    }
    if (root != "") {
        fs::create_directory(root);
    }
}

std::string get_index_dir_path(size_t index) {
    std::string s = output_root;
    return s + "/" + std::to_string(index) + "/";
}

static inline size_t output_index = 0;

void output_all_data(runtime_tensor &input, runtime_tensor &output_ref,
                     runtime_tensor &output_opt) {
    make_output_root_dir();
    auto dir_name = get_index_dir_path(output_index);
    output_data(input, "input", dir_name);
    output_data(output_ref, "output_ref", dir_name);
    output_data(output_opt, "output_opt", dir_name);
    ++output_index;
}

void output_all_data(std::vector<runtime_tensor> &inputs,
                     runtime_tensor &output_ref, runtime_tensor &output_opt) {
    make_output_root_dir();
    auto dir_name = get_index_dir_path(output_index);
    for (size_t i = 0; i < inputs.size(); ++i) {
        output_data(inputs[i], "input" + std::to_string(i), dir_name);
    }
    output_data(output_ref, "output_ref", dir_name);
    output_data(output_opt, "output_opt", dir_name);
    ++output_index;
}
