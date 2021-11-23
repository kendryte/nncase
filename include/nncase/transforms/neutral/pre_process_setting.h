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
    pre_process_transform(std::vector<float> mean, std::vector<float> std, std::vector<float> input_range, std::vector<int32_t> input_shape, bool swapRB, std::string input_layout, std::string input_type, std::string quant_type, std::string real_inlayout, bool do_letterbox, float letterbox_value) noexcept
        : mean_(std::move(mean)), std_(std::move(std)), input_range_(input_range), input_shape_(input_shape), swapRB_(swapRB), input_layout_(input_layout), input_type_(input_type), quant_type_(quant_type), real_inlayout_(real_inlayout), do_letterbox_(do_letterbox), letterbox_value_(letterbox_value) {};
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<float> input_range_;
    std::vector<int32_t> input_shape_;
    bool swapRB_;
    std::string input_layout_;
    std::string input_type_;
    std::string quant_type_;
    std::string real_inlayout_;
    bool do_letterbox_;
    float letterbox_value_;
};
}
