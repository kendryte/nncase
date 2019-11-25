/* Copyright 2019 Canaan Inc.
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
#include <ir/evaluator.h>
#include <ir/ops/constant.h>
#include <ir/visitor.h>
#include <scheduler/scheduler.h>
#include <targets/target.h>
#include <transforms/neutral/fold_constant.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;
using namespace nncase::scheduler;

namespace
{
std::unordered_set<node_opcode> dontfold_ops { op_fake_dequantize, op_fake_quantize };
}

bool fold_constant_transform::on_try_match(node &node, transform_context &context)
{
    if (dontfold_ops.find(node.runtime_opcode()) == dontfold_ops.end()
        && node.inputs().size() && std::all_of(node.inputs().begin(), node.inputs().end(), [](input_connector &conn) {
               return conn.connection()->owner().runtime_opcode() == op_constant;
           }))
    {
        for (auto &in : node.inputs())
            context.inputs.emplace_back(&in);
        for (auto &out : node.outputs())
            context.outputs.emplace_back(&out);

        context.matched_nodes.emplace_back(&node);
        for (auto &in : node.inputs())
            context.matched_nodes.emplace_back(&in.connection()->owner());
        return true;
    }

    return false;
}

void fold_constant_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_op = *context.matched_nodes[0];

    // 1. Construct new eval graph
    graph new_graph;
    std::vector<output_node *> op_outputs;
    std::vector<constant *> output_values;
    for (auto &out : old_op.outputs())
    {
        auto node = op_outputs.emplace_back(new_graph.emplace<output_node>(out.type(), out.shape()));
        node->input().connect(out);
    }

    // 2. Eval
    {
        std::vector<std::unique_ptr<memory_allocator>> allocator_holder;
        std::unordered_map<memory_type_t, memory_allocator *> allocators;
        target_.fill_allocators(allocators, allocator_holder);
        allocation_context alloc_ctx(allocators);
        std::vector<node *> compute_sequence;
        schedule(new_graph.outputs(), alloc_ctx, compute_sequence, 0);
        evaluate_context eval_ctx(allocators, alloc_ctx.allocations());
        evaluator eval(eval_ctx, compute_sequence);
        eval.evaluate();

        for (size_t i = 0; i < op_outputs.size(); i++)
        {
            auto &op_output = *op_outputs[i];
            auto mem = eval.output_at<uint8_t>(i);
            auto out_val = context.graph.emplace<constant>(op_output.input().type(), op_output.input().shape(), mem.begin(), mem.end());
            output_values.emplace_back(out_val);
        }
    }

    // 3. Clear eval graph connections to main graph
    for (auto &out : op_outputs)
        out->input().clear_connection();

    for (size_t i = 0; i < old_op.outputs().size(); i++)
    {
        auto &out = old_op.outputs()[i];
        for (auto &in : dup(out.connections()))
            in->connect(output_values[i]->output());
    }
}
