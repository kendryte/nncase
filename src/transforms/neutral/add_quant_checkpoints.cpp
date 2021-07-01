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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/add_quant_checkpoints.h>
#include <xtensor/xstrides.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool add_quant_checkpoints_transform::on_try_match(node &node, transform_context &context)
{
    if (opcodes_.find(node.runtime_opcode()) != opcodes_.end())
    {
        bool not_processed = false;
        if (!node.inputs().empty()
            && std::any_of(node.inputs().begin(), node.inputs().end(), [](input_connector *in) { return (in->connection()->attributes() & cnctr_attr_need_quantize) != cnctr_attr_need_quantize; }))
            not_processed = true;
        if (!not_processed
            && !node.outputs().empty()
            && std::any_of(node.outputs().begin(), node.outputs().end(), [](output_connector *out) { return (out->attributes() & cnctr_attr_need_quantize) != cnctr_attr_need_quantize; }))
            not_processed = true;

        if (not_processed)
        {
            for (auto in : node.inputs())
                context.inputs.emplace_back(in);
            for (auto out : node.outputs())
                context.outputs.emplace_back(out);

            context.matched_nodes.emplace_back(&node);
            return true;
        }
    }

    return false;
}

void add_quant_checkpoints_transform::process(transform_context &context)
{
    auto &node = *context.matched_nodes[0];
    node.attributes(node.attributes() | node_attr_need_quantize);

    for (size_t i = 0; i < node.inputs().size(); i++)
    {
        auto &output = *node.input_at(i).connection();
        output.attributes(output.attributes() | cnctr_attr_need_quantize);
    }

    for (size_t i = 0; i < node.outputs().size(); i++)
    {
        auto &output = node.output_at(i);
        output.attributes(output.attributes() | cnctr_attr_need_quantize);
    }
}
