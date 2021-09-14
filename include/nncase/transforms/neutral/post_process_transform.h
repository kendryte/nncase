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
class NNCASE_API post_process_transform : public graph_pass
{
public:
    post_process_transform(std::string output_layout, std::string output_type, std::string real_outlayout) noexcept
        : output_layout_(output_layout), output_type_(output_type), real_outlayout_(real_outlayout) {};
    using graph_pass::graph_pass;

protected:
    void run_core(graph &graph, nncase::target &target, const run_pass_options &options) override;

private:
    std::string output_layout_;
    std::string output_type_;
    std::string real_outlayout_;
};
}
