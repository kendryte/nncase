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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/process_input.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

quant_param_t get_quant_param(datatype_t type)
{
    // TODO: 增加normalize之后需要根据std 和 mean 来计算新的量化参数
    if (type == datatype_t::dt_uint8)
    {
        return { static_cast<int32_t>(128), 1 / 128.f };
    }
    else if (type == datatype_t::dt_int8)
    {
        return { static_cast<int32_t>(1), 1 / 128.f };
    }
    else
        throw std::runtime_error("Not support quant type");
}

bool process_input::on_try_match(node &node, transform_context &context)
{
    if (auto in_node = node_cast<input_node>(node))
    {
        if (in_node->output().type() == dt_float32)
        {
            context.outputs.emplace_back(&in_node->output());
            context.matched_nodes.emplace_back(in_node);
            return true;
        }
    }
    return false;
}

void process_input::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto old_in = node_cast<input_node>(*context.matched_nodes[0]);

    quant_param_t params = get_quant_param(input_type_);
    auto new_in_node = context.graph.emplace<input_node>(input_type_, old_in->output().shape());
    auto deq = context.graph.emplace<dequantize>(new_in_node->output().type(), new_in_node->output().shape(), dt_float32, params);
    deq->input().connect(new_in_node->output());

    new_in_node->name(old_in->name());
    deq->name(new_in_node->name() + "/deq");

    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
