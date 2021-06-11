/* Copyright 2019-2020 Canaan Inc.
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
#include "../caffe_importer.h"

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(CtcLoss)
{
    // auto &input_a = *output_tensors_.at(op.bottom(0));
    // auto &input_b = *output_tensors_.at(op.bottom(1));

    // auto &input = input_a.owner().runtime_opcode() == op_input_node ? input_b : input_a;
    // auto tp = graph_.emplace<transpose>(dt_float32, input.shape(), axis_t { 0,1,2,3 });
    // tp->name(op.name() + "/transpose");

    // std::string_view sv = input_a.owner().runtime_opcode() == op_input_node ? op.bottom(1) : op.bottom(0);
    // input_tensors_.emplace(&tp->input(), sv);
    // output_tensors_.emplace(op.top(0), &tp->output());
    auto &input_a = *output_tensors_.at(op.bottom(0));
    auto &input_b = *output_tensors_.at(op.bottom(1));

    auto &input = input_a.owner().runtime_opcode() == op_input_node ? input_b : input_a;
    output_tensors_.emplace(op.top(0), &input);
}
