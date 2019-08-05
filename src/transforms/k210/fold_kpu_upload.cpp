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
#include <ir/ops/k210/kpu_data_exchange.h>
#include <ir/visitor.h>
#include <transforms/k210/fold_kpu_upload.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

bool fold_kpu_upload_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_kpu_upload)
    {
        auto &up = static_cast<kpu_upload &>(node);
        if (auto down = try_get_direct_child<kpu_download>(up))
        {
            context.inputs.emplace_back(&up.input());
            context.outputs.emplace_back(&down->output());

            context.matched_nodes.emplace_back(&up);
            context.matched_nodes.emplace_back(down);
            return true;
        }
    }
    else if (node.runtime_opcode() == op_k210_kpu_download)
    {
        auto &down = static_cast<kpu_download &>(node);
        if (auto up = try_get_direct_child<kpu_upload>(down))
        {
            context.inputs.emplace_back(&down.input());
            context.outputs.emplace_back(&up->output());

            context.matched_nodes.emplace_back(&down);
            context.matched_nodes.emplace_back(up);
            return true;
        }
    }

    return false;
}

void fold_kpu_upload_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_input_kpu_upload_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_input_node)
    {
        auto &in = static_cast<input_node &>(node);
        if (auto up = try_get_direct_child<kpu_upload>(in))
        {
            context.outputs.emplace_back(&up->output());

            context.matched_nodes.emplace_back(&in);
            context.matched_nodes.emplace_back(up);
            return true;
        }
    }

    return false;
}

void fold_input_kpu_upload_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto &old_in = static_cast<input_node &>(*context.matched_nodes[0]);

    auto input = context.graph.emplace<input_node>(dt_uint8, old_in.output().shape(), mem_k210_kpu);

    for (auto &in : dup(inputs))
        in->connect(input->output());
}
