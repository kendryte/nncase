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
#include <ir/ops/k210/kpu_conv2d.h>
#include <ir/ops/k210/kpu_data_exchange.h>
#include <ir/visitor.h>
#include <transforms/k210/fuse_kpu_download.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

bool fuse_kpu_download_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_kpu_download)
    {
        auto &download = static_cast<kpu_download &>(node);
        if (auto conv = try_get_direct_parent<kpu_conv2d>(download))
        {
            context.inputs.emplace_back(&conv->input());
            context.outputs.emplace_back(&download.output());

            context.matched_nodes.emplace_back(conv);
            context.matched_nodes.emplace_back(&download);
            return true;
        }
    }

    return false;
}

void fuse_kpu_download_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_conv = static_cast<kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_download = static_cast<kpu_download &>(*context.matched_nodes[1]);

    auto conv = context.graph.emplace<kpu_conv2d>(true, old_conv.input().shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        old_conv.weights(), old_conv.pad_value(), old_conv.arg_x(), old_conv.shift_x(), old_conv.arg_w(), old_conv.shift_w(), old_conv.arg_add(),
        old_conv.batch_norm(), old_conv.activation());

    bool has_other_inputs = false;
    for (auto &out : dup(old_conv.kpu_output().connections()))
    {
        if (out->owner().runtime_opcode() != op_k210_kpu_download)
        {
            out->connect(conv->kpu_output());
            has_other_inputs = true;
        }
    }

    if (!has_other_inputs)
    {
        auto ignore = context.graph.emplace<ignore_node>(conv->kpu_output().type(), conv->kpu_output().shape());
        ignore->input().connect(conv->kpu_output());
    }

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(conv->main_mem_output());
}
