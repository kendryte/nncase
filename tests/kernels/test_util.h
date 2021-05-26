/* Copyright 2020 Canaan Inc.
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
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <random>
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

enum OpType
{
    Opt,
    Ref
};

template <typename T>
class Tensor;

void print_index(const runtime_shape_t& index)
{
    for (size_t i = 0; i < index.size(); ++i)
    {
        std::cout << index[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
T *alloc_tensor_data(const runtime_shape_t &shape, const runtime_shape_t &strides)
{
    auto length = compute_size(shape, strides);
    //auto length = std::accumulate(strides.begin(), strides.end(), 1, [](auto sum, auto cur) {
    //    return sum * (cur == 0 ? 1 : cur);
    //});
    auto *v = new T[length];
    return v;
}

template <typename T>
T *alloc_tensor_data(const Tensor<T>& tensor)
{
    auto length = compute_size(tensor.shape, tensor.strides);
    return new T[length];
}

size_t get_last_no_zero_stride(const runtime_shape_t &strides, size_t i)
{
    for (size_t j = i; j < strides.size(); ++j)
    {
        if (strides[j] != 0)
        {
            return strides[j];
        }
    }
    assert(false);
    return 0;
}

// strides bias first value is no effect
// line is contiguous
runtime_shape_t get_strides(const runtime_shape_t &shape, const runtime_shape_t &strides_bias) 
{
    runtime_shape_t strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--)
    { 
        const auto line_width = shape[i + 1] + strides_bias[i + 1];
        const auto shape_v = line_width == 1 ? 0 : line_width;
        const auto strides_v = strides[i + 1] == 0 ? get_last_no_zero_stride(strides, i + 1) : strides[i + 1];
        strides[i] = strides_v * shape_v;
    }
    return strides;
}

template<typename T>
void init_tensor_data(Tensor<T>& data)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> dis(-11.0, 11.0);
    std::uniform_int_distribution<uint32_t> dis(3, 333);
    NNCASE_UNUSED auto res = cpu::reference::apply(data.shape,
        [&](const runtime_shape_t &index) -> result<void> {
            // print_index(index);
            data.get(index) = dis(gen);
            // std::cout << "data:" << data.get(index) << std::endl;
            return ok();
        });
}

runtime_shape_t shape_sub(runtime_shape_t begin, runtime_shape_t end)
{
    runtime_shape_t v;
    for (size_t i = 0; i < begin.size(); ++i)
    {
        v.push_back(end[i] - begin[i]);
    }
    return v;
}

// memory release and smart ptr
template <typename T = uint32_t>
class Tensor
{
public:
    T *data;
    runtime_shape_t shape, strides;

    Tensor() = default;

    Tensor(runtime_shape_t shape, runtime_shape_t strides_bias)
    {
        this->shape = shape;
        this->strides = get_strides(shape, strides_bias);
        data = alloc_tensor_data<T>(shape, strides);
    }

    Tensor(T* data)
    {
        this->data = data;
    }

    bool operator==(const Tensor &rhs) const
    {
        if (shape != rhs.shape)
        {
            return false;
        }
        auto res = cpu::reference::apply(shape,
            [&](const runtime_shape_t &index) -> result<void> {
                // print_index(index);
                // std::cout << get(index) << " " << rhs.get(index) << std::endl;
                if (get(index) == rhs.get(index))
                {
                    return ok();
                }
                else
                {
                    // TODO:
                    return err(std::errc::not_supported);
                }
            });
        return res.is_ok();
    }
    T &get(const runtime_shape_t &index) const
    {
        return data[offset(strides, index)];
    }

    gsl::byte* gsl_ptr() 
    {
        return reinterpret_cast<gsl::byte *>(data);
    }

    const gsl::byte* gsl_cptr() const
    {
        return reinterpret_cast<const gsl::byte *>(data);
    }

    ~Tensor()
    {
    }
};

template<typename T>
void print_data(const Tensor<T>& data)
{
    NNCASE_UNUSED auto res = cpu::reference::apply(data.shape,
        [&](const runtime_shape_t& index) -> result<void> {
        std::cout << data.get(index) << std::endl;
            return ok();
        });
}

namespace fs = std::filesystem;
bool exist_directory(const fs::path& path)
{
    std::error_code error;
    auto file_status = std::filesystem::status(path, error);
    if (error)
    {
        return false;
    }
    return fs::exists(file_status) && fs::is_directory(file_status);
}

template <typename T>
void output_data(const Tensor<T> &data, std::string name, std::string dir_name = "op_test")
{
    if (!exist_directory(dir_name))
    {
        fs::create_directory(dir_name);
    }
    std::ofstream f(dir_name + "/" + name + ".txt");
    // output shape
    auto shape_str = std::accumulate(data.shape.begin(), data.shape.end(), std::string(), [](std::string s, T v) {
        return s + std::to_string(v) + ",";
    });
    shape_str.pop_back();
    f << "shape(" << shape_str << ")" << std::endl;
    NNCASE_UNUSED auto res = cpu::reference::apply(data.shape,
        [&](const runtime_shape_t &index) -> result<void> {
            f << data.get(index) << std::endl;
            return ok();
        });
    f.close();
}

constexpr auto output_root = "op_test";

void make_output_root_dir(std::string root = output_root)
{
    if (exist_directory(root))
    {
        fs::remove_all(root);
    }
    fs::create_directory(root);
}

std::string get_index_dir_path(size_t index)
{
    std::string s = output_root;
    return s + "/" + std::to_string(index) + "/";
}

static inline size_t output_index = 0;
template<typename T>
void output_all_data(const Tensor<T> &input, const Tensor<T> &output_ref, const Tensor<T> &output_opt)
{
    make_output_root_dir();
    auto dir_name = get_index_dir_path(output_index);
    output_data(input, "input", dir_name);
    output_data(output_ref, "output_ref", dir_name);
    output_data(output_opt, "output_opt", dir_name);
    ++output_index;
}

template <typename T>
void output_all_data(const std::vector<Tensor<T>> &inputs, const Tensor<T> &output_ref, const Tensor<T> &output_opt)
{
    make_output_root_dir();
    auto dir_name = get_index_dir_path(output_index);
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        output_data(inputs[i], "input" + std::to_string(i), dir_name);
    }
    output_data(output_ref, "output_ref", dir_name);
    output_data(output_opt, "output_opt", dir_name);
    ++output_index;
}