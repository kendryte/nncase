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
    bool is_fpga;
    bool use_dataset_as_input_stat = false;
    std::string target;
    std::filesystem::path dump_dir;
    std::string input_type = "default";
    std::string output_type = "float32";
    std::string quant_type = "uint8";
};

struct import_options
{
    std::string input_layout = "NCHW";
    std::string output_layout = "NCHW";
    std::span<const std::string> output_arrays;
};

struct ptq_options_base
{
    std::string calibrate_method = "no_clip";
    std::function<void(size_t cnt, size_t total)> progress;

    float input_mean = 0.f;
    float input_std = 1.f;
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

class NNCASE_API compiler
{
public:
    static std::unique_ptr<compiler> create(const compile_options &options);

    virtual ~compiler();
    virtual void import_tflite(std::span<const uint8_t> model, const import_options &options) = 0;
    virtual void import_onnx(std::span<const uint8_t> model, const import_options &options) = 0;
    virtual void import_caffe(std::span<const uint8_t> model, const import_options &options) = 0;
    virtual void use_ptq(ptq_dataset_options options) = 0;
    virtual void use_ptq(ptq_tensor_options options) = 0;
    virtual ir::graph &graph(uint32_t stage) = 0;
    virtual nncase::target &target() = 0;
    virtual void compile() = 0;
    virtual void gencode(std::ostream &output) = 0;
};
}
