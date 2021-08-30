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
#include "../pass.h"

namespace nncase::ir::transforms
{
class NNCASE_API pre_process_transform : public graph_pass
{
public:
    pre_process_transform(std::vector<float> mean, std::vector<float> scale, std::vector<float> input_range, std::vector<int32_t> input_shape, std::string image_format, std::string input_layout, std::string input_type, std::string quant_type, std::string real_layout, bool preprocess) noexcept
        : means_(std::move(mean)), scales_(std::move(scale)), input_range_(input_range), input_shape_(input_shape), image_format_(image_format), input_layout_(input_layout), input_type_(input_type), quant_type_(quant_type), real_layout_(real_layout), preprocess_(preprocess) {};
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;

private:
    std::vector<float> means_;
    std::vector<float> scales_;
    std::vector<float> input_range_;
    std::vector<int32_t> input_shape_;
    std::string image_format_;
    std::string input_layout_;
    std::string input_type_;
    std::string quant_type_;
    std::string real_layout_;
    bool preprocess_;
};
}
