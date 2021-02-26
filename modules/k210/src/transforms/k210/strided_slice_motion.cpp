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
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/transforms/k210/strided_slice_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

bool strided_slice_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_slice)
    {
        auto &slc = static_cast<slice &>(node);
        if (slc.strides() == axis_t { 1, 1, 2, 2 }
            && slc.begin() == axis_t { 0, 0, 0, 0 }
            && slc.end() == axis_t { (int32_t)slc.input().shape()[0], (int32_t)slc.input().shape()[1], (int32_t)slc.input().shape()[2], (int32_t)slc.input().shape()[3] }
            && slc.begin_mask() == 0
            && slc.end_mask() == 0
            && slc.new_axis_mask() == 0
            && slc.shrink_axis_mask() == 0)
        {
            if (auto conv = try_get_direct_child<fake_kpu_conv2d>(slc))
            {
                if (!conv->is_depthwise()
                    && conv->filter_type() == kpu_filter_1x1
                    && conv->pool_type() == kpu_pool_bypass)
                {
                    context.inputs.emplace_back(&slc.input());
                    context.inputs.emplace_back(&conv->weights());
                    context.inputs.emplace_back(&conv->bias());
                    context.outputs.emplace_back(&conv->output());

                    context.matched_nodes.emplace_back(&slc);
                    context.matched_nodes.emplace_back(conv);
                    return true;
                }
            }
        }
    }

    return false;
}

void strided_slice_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<slice &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[1]);

    auto conv = context.graph.emplace<fake_kpu_conv2d>(output.shape(), old_conv.is_depthwise(), old_conv.weights().shape(),
        old_conv.filter_type(), old_conv.pool_type(), old_conv.fused_activation());
    conv->name(old_conv.name());
    conv->weights().connect(weights);
    conv->bias().connect(bias);
    auto slc = context.graph.emplace<slice>(dt_float32, conv->output().shape(), axis_t { 0, 0, 0, 0 }, axis_t { 0, 0, 0, 0 }, old_slice.strides(), 15, 15, 0, 0, 0);
    slc->input().connect(conv->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(slc->output());
}
