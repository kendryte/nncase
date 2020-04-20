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
#include <hlir/ops/constant.h>
#include <hlir/ops/k210/fake_kpu_conv2d.h>
#include <hlir/ops/matmul.h>
#include <hlir/ops/pad.h>
#include <hlir/ops/reshape.h>
#include <hlir/quantizer.h>
#include <hlir/transforms/k210/kpu_utils.h>
#include <hlir/transforms/k210/matmul_to_fake_kpu_conv2d.h>
#include <hlir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <targets/target.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::hlir::transforms;
using namespace nncase::hlir::transforms::k210;

bool matmul_to_fake_kpu_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto mm = node_cast<matmul>(node))
    {
        if (auto w = try_get_direct_parent<constant>(*mm, 1))
        {
            {
                auto w_beg = reinterpret_cast<const float *>(w->data().data());
                auto w_end = w_beg + w->data().size() / sizeof(float);

                auto total_range = quantizer::fixup_range(quantizer::get_range(w_beg, w_end));
                if (total_range.max - total_range.min > context.target.options().weights_quantize_threshold)
                    return false;
            }

            if (xt::compute_size(w->output().shape()) >= context.target.options().output_quantize_threshold
                && w->output().shape()[0] <= 1024
                && w->output().shape()[1] <= 1024)
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

    shape_t pre_shape { output.shape()[0], output.shape()[1], 1, 1 };
    shape_t sur_shape { output.shape()[0], old_mm.input_b().shape()[1] };
    shape_t w_shape { old_mm.input_b().shape()[0], old_mm.input_b().shape()[1], 1, 1 };
    xt::svector<padding> pre_paddings { padding::zero(), padding::zero(), { 0, 3 }, { 0, 3 } };
    xt::svector<padding> sur_paddings { padding::zero(), padding::zero(), { 0, -3 }, { 0, -3 } };
    xt::xtensor<float, 4> mm_w(xt::adapt(reinterpret_cast<const float *>(old_w.data().data()), w_shape));
    auto w = xt::transpose(mm_w, { 1, 0, 2, 3 });

    auto pre_reshape = context.graph.emplace<reshape>(dt_float32, output.shape(), pre_shape);
    auto pre_pad = context.graph.emplace<pad>(dt_float32, pre_reshape->output().shape(), pre_paddings, 0.f);
    auto conv = context.graph.emplace<fake_kpu_conv2d>(pre_pad->output().shape(), false, kpu_filter_1x1, kpu_pool_bypass,
        w, old_mm.bias(), old_mm.fused_activation());
    auto sur_pad = context.graph.emplace<pad>(dt_float32, conv->output().shape(), sur_paddings, 0.f);
    auto sur_reshape = context.graph.emplace<reshape>(dt_float32, sur_pad->output().shape(), sur_shape);
    pre_pad->input().connect(pre_reshape->output());
    conv->input().connect(pre_pad->output());
    sur_pad->input().connect(conv->output());
    sur_reshape->input().connect(sur_pad->output());

    pre_reshape->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(sur_reshape->output());
}
