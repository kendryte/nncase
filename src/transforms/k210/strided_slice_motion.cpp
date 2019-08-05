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
#include <ir/ops/conv2d.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <ir/ops/pad.h>
#include <ir/ops/strided_slice.h>
#include <ir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <transforms/k210/strided_slice_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

bool strided_slice_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_strided_slice)
    {
        auto &slice = static_cast<strided_slice &>(node);
        if (slice.strides() == axis_t { 1, 1, 2, 2 }
            && slice.begin() == axis_t { 0, 0, 0, 0 }
            && slice.end() == axis_t { (int32_t)slice.input().shape()[0], (int32_t)slice.input().shape()[1], (int32_t)slice.input().shape()[2], (int32_t)slice.input().shape()[3] }
            && slice.begin_mask() == 0
            && slice.end_mask() == 0
            && slice.new_axis_mask() == 0
            && slice.shrink_axis_mask() == 0)
        {
            if (auto conv = try_get_direct_child<fake_kpu_conv2d>(slice))
            {
                if (!conv->is_depthwise()
                    && conv->filter_type() == kpu_filter_1x1
                    && conv->pool_type() == kpu_pool_bypass)
                {
                    context.inputs.emplace_back(&slice.input());
                    context.outputs.emplace_back(&conv->output());

                    context.matched_nodes.emplace_back(&slice);
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
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<strided_slice &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[1]);

    auto conv = context.graph.emplace<fake_kpu_conv2d>(output.shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        old_conv.weights(), old_conv.bias(), old_conv.fused_activation());
    auto slice = context.graph.emplace<strided_slice>(dt_float32, conv->output().shape(), axis_t { 0, 0, 0, 0 }, axis_t { 0, 0, 0, 0 }, old_slice.strides(), 15, 15, 0, 0, 0);
    slice->input().connect(conv->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(slice->output());
}
