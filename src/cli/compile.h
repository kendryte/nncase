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
#include <lyra/lyra.hpp>
#include <string>

namespace nncase::cli
{
class compile_command
{
public:
    compile_command(lyra::cli &cli);

private:
    void run();

    template <typename T>
    void parser_vector_opt(const std::string &cli_opt, std::vector<T> &opt)
    {
        std::string::const_iterator last_pos, pos;
        last_pos = pos = cli_opt.cbegin();
        do
        {
            pos = std::find(last_pos, cli_opt.cend(), ' ');
            if (last_pos != pos)
            {
                std::string substr = cli_opt.substr(last_pos - cli_opt.begin(), pos - last_pos);
                if constexpr (std::is_same_v<T, int32_t>)
                {
                    opt.push_back(std::stoi(substr));
                }
                else if constexpr (std::is_same_v<T, float>)
                {
                    opt.push_back(std::stof(substr));
                }
                else
                {
                    throw std::runtime_error("not supported arguments type");
                }
            }
            last_pos = pos + 1;
        } while (pos != cli_opt.cend());
    }

private:
    std::string input_filename_;
    std::string output_filename_;
    std::string input_prototxt_;
    std::string input_format_;
    std::string target_name_;
    std::string output_arrays_;
    std::string dump_dir_;
    std::string dataset_;
    std::string dataset_format_ = "image";
    std::string calibrate_method_ = "no_clip";
    std::string input_type_ = "default";
    std::string output_type_ = "float32";
    std::string quant_type_ = "uint8";
    std::string w_quant_type_ = "uint8";
    std::string input_layout_ = "NCHW";
    std::string output_layout_ = "NCHW";
    bool use_mse_quant_w_ = false;
    std::vector<float> mean_ = { 0.f, 0.f, 0.f };
    std::vector<float> std_ = { 1.f, 1.f, 1.f };
    std::vector<float> input_range_;
    float letterbox_value_;
    std::vector<int32_t> input_shape_;

    std::string cli_mean_ = "0. 0. 0.";
    std::string cli_std_ = "1. 1. 1.";
    std::string cli_input_range_;
    std::string cli_input_shape_;

    bool swapRB_ = false;
    bool dump_ir_ = false;
    bool dump_asm_ = false;
    bool dump_quant_error_ = false;
    bool dump_import_op_range_ = false;
    bool is_fpga_ = false;
    bool benchmark_only_ = false;
    bool preprocess_ = false;
};
}
