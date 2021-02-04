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
#include <nncase/ir/evaluator.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/scheduler.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/fold_constant.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;
using namespace nncase::schedule;

namespace
{
std::unordered_set<node_opcode> dontfold_ops {};
}

bool fold_constant_transform::on_try_match(node &node, transform_context &context)
{
    if ((node.attributes() & node_attr_skip_constant_folding) == 0
        && dontfold_ops.find(node.runtime_opcode()) == dontfold_ops.end()
        && node.inputs().size() && std::all_of(node.inputs().begin(), node.inputs().end(), [](input_connector *in) {
               return in->connection()->owner().runtime_opcode() == op_constant;
           }))
    {
        for (auto &in : node.inputs())
            context.inputs.emplace_back(in);
        for (auto out : node.outputs())
            context.outputs.emplace_back(out);

        context.matched_nodes.emplace_back(&node);
        for (auto in : node.inputs())
            context.matched_nodes.emplace_back(&in->connection()->owner());
        return true;
    }

    return false;
}

void fold_constant_transform::process(transform_context &context)
{
    //auto &output = *context.inputs[0]->connection();
    //auto inputs = context.outputs[0]->connections();

    auto &old_op = *context.matched_nodes[0];

    // 1. Construct new eval graph
    graph new_graph;
    std::vector<output_node *> op_outputs;
    std::vector<constant *> output_values;
    for (auto out : old_op.outputs())
    {
        auto node = op_outputs.emplace_back(new_graph.emplace<output_node>(out->type(), out->shape()));
        if (old_op.outputs().size() > 1)
            node->name(out->name() + "_F");
        else
            node->name(out->owner().name());
        node->input().connect(*out);
    }

    // 2. Eval
    {
        scheduler sch(context.target, new_graph, new_graph.outputs());
        auto schr = sch.schedule();
        ir::evaluator eval(schr);
        eval.evaluate();

        for (size_t i = 0; i < op_outputs.size(); i++)
        {
            auto &op_output = *op_outputs[i];
            // TODO: consider of strides
            auto mem = runtime::host_runtime_tensor::buffer(eval.output_at(i)).unwrap_or_throw().as_span<std::byte>();
            auto out_val = context.graph.emplace<constant>(op_output.input().type(), op_output.input().shape(), mem.begin(), mem.end());
            out_val->name(op_output.name());
            output_values.emplace_back(out_val);
        }
    }

    // 3. Clear eval graph connections to main graph
    for (auto &out : op_outputs)
        out->input().clear_connection();

    for (size_t i = 0; i < old_op.outputs().size(); i++)
    {
        auto &out = old_op.outputs()[i];
        for (auto &in : dup(out->connections()))
            in->connect(output_values[i]->output());
    }
}
