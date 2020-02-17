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
#include <hlir/ops/constant.h>
#include <hlir/ops/k210/fake_kpu_conv2d.h>
#include <hlir/ops/matmul.h>
#include <hlir/ops/pad.h>
#include <hlir/transforms/k210/matmul_to_fake_kpu_conv2d.h>
#include <hlir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::hlir::transforms;
using namespace nncase::hlir::transforms::k210;

#define KPU_MATMUL_THRESHOLD (10 * 1024)

bool matmul_to_fake_kpu_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto mm = node_cast<matmul>(node))
    {
        if (auto w = try_get_direct_parent<constant>(*mm, 1))
        {
            if (xt::compute_size(w->output().shape()) <= KPU_MATMUL_THRESHOLD)
            {
                context.inputs.emplace_back(&mm->input_a());
                context.outputs.emplace_back(&mm->output());

                context.matched_nodes.emplace_back(mm);
                context.matched_nodes.emplace_back(w);
                return true;
            }
        }
    }

    return false;
}

void matmul_to_fake_kpu_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_mm = static_cast<matmul &>(*context.matched_nodes[0]);
    auto &old_w = static_cast<constant &>(*context.matched_nodes[1]);

    xt::svector<padding> pre_paddings { padding::zero(), padding::zero(), { 0, 3 }, { 0, 3 } };
    xt::svector<padding> sur_paddings { padding::zero(), padding::zero(), { 0, -3 }, { 0, -3 } };

    auto pre_pad = context.graph.emplace<pad>(dt_float32, output.shape(), pre_paddings, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(output.shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        old_conv.weights(), old_conv.bias(), old_conv.fused_activation());
    auto slice = context.graph.emplace<strided_slice>(dt_float32, conv->output().shape(), axis_t { 0, 0, 0, 0 }, axis_t { 0, 0, 0, 0 }, old_slice.strides(), 15, 15, 0, 0, 0);
    slice->input().connect(conv->output());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(slice->output());
}
