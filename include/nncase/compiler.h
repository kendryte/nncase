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
#include "plugin_loader.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace nncase::ir
{
class graph;
}

namespace nncase
{
class target;

struct compile_options
{
    bool dump_ir;
    bool dump_asm;
    bool dump_quant_error;
    bool dump_import_op_range;
    bool is_fpga;
    bool use_dataset_as_input_stat = false;
    bool benchmark_only = false;
    bool preprocess = false;
    bool swapRB = false;
    std::string target;
    std::filesystem::path dump_dir;
    std::string input_type = "default";
    std::string output_type = "float32";
    std::string quant_type = "uint8";
    std::vector<float> mean { 0.f, 0.f, 0.f };
    std::vector<float> std { 1.f, 1.f, 1.f };
    std::vector<float> input_range { 0.f, 1.f };
    std::vector<float> output_range;
    float letterbox_value = 0.f;
    std::vector<int32_t> input_shape {};
    std::string w_quant_type = "uint8";
    bool use_mse_quant_w = false;
    bool split_w_to_act = false;
    std::string input_layout = "NCHW";
    std::string output_layout = "NCHW";
    std::string model_layout;
};

struct import_options
{
    std::span<const std::string> output_arrays;
};

struct ptq_options_base
{
    std::string calibrate_method = "no_clip";
    std::function<void(size_t cnt, size_t total)> progress;
};

struct ptq_dataset_options : ptq_options_base
{
    std::filesystem::path dataset;
    std::string dataset_format;
};

struct ptq_tensor_options : ptq_options_base
{
    std::vector<uint8_t> tensor_data;
    size_t samples_count;
};

struct dump_range_options_base
{
    std::string calibrate_method = "no_clip";
    std::function<void(size_t cnt, size_t total)> progress;
};

struct dump_range_dataset_options : dump_range_options_base
{
    std::filesystem::path dataset;
    std::string dataset_format;
};
struct dump_range_tensor_options : dump_range_options_base
{
    std::vector<uint8_t> tensor_data;
    size_t samples_count;
};

class NNCASE_API compiler
{
public:
    static std::unique_ptr<compiler> create(const compile_options &options);

    virtual ~compiler();
    virtual void import_tflite(std::span<const uint8_t> model, const import_options &options) = 0;
    virtual void import_onnx(std::span<const uint8_t> model, const import_options &options) = 0;
    virtual void import_caffe(std::span<const uint8_t> model, std::span<const uint8_t> prototxt) = 0;
    virtual void import_pnnx(std::string parampath, std::string binpath, const import_options &options) = 0;
    virtual void use_ptq(ptq_dataset_options options) = 0;
    virtual void use_ptq(ptq_tensor_options options) = 0;
    virtual void dump_range_options(dump_range_dataset_options options) = 0;
    virtual void dump_range_options(dump_range_tensor_options options) = 0;
    virtual ir::graph &graph(uint32_t stage) = 0;
    virtual nncase::target &target() = 0;
    virtual void compile() = 0;
    virtual void gencode(std::ostream &output) = 0;
};
}
