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
#include <ir/ops/reshape.h>
#include <ir/ops/transpose.h>
#include <ir/visitor.h>
#include <transforms/neutral/fold_transpose.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

// Transpose (perm = p1)
//     |
//     v
// Transpose (perm = p2)
//
// y1[i] = x1[p1[i]]
// y2[i] = x2[p2[i]]
// x2 = y1
// =>
// y2[i] = y1[p2[i]] = x1[p1[p2[i]]]

bool fold_transpose_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_transpose)
    {
        auto &tp1 = static_cast<transpose &>(node);
        if (auto tp2 = try_get_direct_child<transpose>(tp1))
        {
            if (tp1.perm().size() == tp2->perm().size())
            {
                context.inputs.emplace_back(&tp1.input());
                context.outputs.emplace_back(&tp2->output());

                context.matched_nodes.emplace_back(&tp1);
                context.matched_nodes.emplace_back(tp2);
                return true;
            }
        }
    }

    return false;
}

void fold_transpose_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &p1 = static_cast<transpose *>(context.matched_nodes[0])->perm();
    auto &p2 = static_cast<transpose *>(context.matched_nodes[1])->perm();

    axis_t perm(p1.size());
    for (size_t i = 0; i < p1.size(); i++)
        perm[i] = p1[p2[i]];

    auto tp = context.graph.emplace<transpose>(output.type(), output.shape(), perm);
    tp->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

// Transpose (perm = p1)
//
// p1[i] = i

bool fold_nop_transpose_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_transpose)
    {
        auto &tp = static_cast<transpose &>(node);

        for (size_t i = 0; i < tp.perm().size(); i++)
        {
            if (tp.perm()[i] != i)
                return false;
        }

        context.inputs.emplace_back(&tp.input());
        context.outputs.emplace_back(&tp.output());

        context.matched_nodes.emplace_back(&tp);
        return true;
    }

    return false;
}

void fold_nop_transpose_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool transpose_to_reshape_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_transpose)
    {
        auto &tp = static_cast<transpose &>(node);

        size_t last_sig_dim = 0;
        for (size_t i = 0; i < tp.perm().size(); i++)
        {
            auto i_dim = tp.perm()[i];
            if (tp.input().shape()[i_dim] != 1)
            {
                if (i_dim < last_sig_dim)
                    return false;
                last_sig_dim = i_dim;
            }
        }

        context.inputs.emplace_back(&tp.input());
        context.outputs.emplace_back(&tp.output());

        context.matched_nodes.emplace_back(&tp);
        return true;
    }

    return false;
}

void transpose_to_reshape_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto rshape = context.graph.emplace<reshape>(output.type(), output.shape(), context.matched_nodes[0]->output_at(0).shape());

    rshape->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rshape->output());
}
