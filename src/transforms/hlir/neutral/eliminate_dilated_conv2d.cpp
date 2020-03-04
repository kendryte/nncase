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
#include <hlir/ops/conv2d.h>
#include <hlir/ops/pad.h>
#include <hlir/ops/reshape.h>
#include <hlir/ops/transpose.h>
#include <hlir/transforms/neutral/eliminate_dilated_conv2d.h>
#include <hlir/visitor.h>
#include <llir/evaluator.h>
#include <scheduler/scheduler.h>
#include <targets/target.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;
using namespace nncase::scheduler;

namespace
{
auto get_pad_crop(conv2d &conv)
{
    xt::svector<padding> new_paddings;
    // batch
    new_paddings.push_back(padding::zero());
    // channel
    new_paddings.push_back(padding::zero());
    // spatial
    new_paddings.push_back(conv.padding_h());
    new_paddings.push_back(conv.padding_w());

    auto pad_end_extra_h = (conv.dilation_h() - ((int32_t)conv.input().shape()[2] + new_paddings[2].sum()) % conv.dilation_h()) % conv.dilation_h();
    auto pad_end_extra_w = (conv.dilation_w() - ((int32_t)conv.input().shape()[3] + new_paddings[3].sum()) % conv.dilation_w()) % conv.dilation_w();
    new_paddings[2].after += pad_end_extra_h;
    new_paddings[3].after += pad_end_extra_w;

    xt::svector<padding> new_crops;
    // batch
    new_crops.push_back(padding::zero());
    // channel
    new_crops.push_back(padding::zero());
    // spatial
    new_crops.push_back(padding::zero());
    new_crops.push_back(padding::zero());
    new_crops[2].after -= pad_end_extra_h;
    new_crops[3].after -= pad_end_extra_w;

    return std::make_pair(new_paddings, new_crops);
}

auto space_to_batch(conv2d &conv, const xt::svector<padding> &paddings, graph &graph)
{
    auto p = graph.emplace<pad>(dt_float32, conv.input().shape(), paddings, 0.f);

    auto padded_shape = p->output().shape();
    shape_t reshapped_shape;
    // batch = 1, skipped
    assert(padded_shape[0] == 1);
    // channel with h
    reshapped_shape.push_back(padded_shape[1] * (padded_shape[2] / conv.dilation_h()));
    // spatial
    reshapped_shape.push_back(conv.dilation_h());
    reshapped_shape.push_back(padded_shape[3] / conv.dilation_w());
    reshapped_shape.push_back(conv.dilation_w());

    axis_t perm;
    // block shape
    perm.push_back(1);
    perm.push_back(3);
    // channel with h
    perm.push_back(0);
    // spatial
    perm.push_back(2);

    shape_t reshapped_shape2;
    // block shape
    reshapped_shape2.push_back(conv.dilation_h() * conv.dilation_w());
    // channel
    reshapped_shape2.push_back(padded_shape[1]);
    // spatial
    reshapped_shape2.push_back(padded_shape[2] / conv.dilation_h());
    reshapped_shape2.push_back(padded_shape[3] / conv.dilation_w());

    auto rshape = graph.emplace<reshape>(dt_float32, p->output().shape(), reshapped_shape);
    auto tp = graph.emplace<transpose>(dt_float32, rshape->output().shape(), perm);
    auto rshape2 = graph.emplace<reshape>(dt_float32, tp->output().shape(), reshapped_shape2);
    rshape->input().connect(p->output());
    tp->input().connect(rshape->output());
    rshape2->input().connect(tp->output());

    return std::make_pair(p, rshape2);
}

auto batch_to_space(conv2d &conv1, conv2d &conv2, const xt::svector<padding> &crops, graph &graph)
{
    shape_t reshapped_shape;
    // block shape
    reshapped_shape.push_back(conv1.dilation_h());
    reshapped_shape.push_back(conv1.dilation_w());
    // channel with h
    reshapped_shape.push_back(conv2.output_channels() * conv2.output().shape()[2]);
    // spatial
    reshapped_shape.push_back(conv2.output().shape()[3]);

    axis_t perm;
    // channel with h
    perm.push_back(2);
    // block h
    perm.push_back(0);
    // w
    perm.push_back(3);
    // block w
    perm.push_back(1);

    shape_t reshapped_shape2;
    // batch
    reshapped_shape2.push_back(1);
    // channel
    reshapped_shape2.push_back(conv2.output_channels());
    // spatial
    reshapped_shape2.push_back(conv2.output().shape()[2] * conv1.dilation_h());
    reshapped_shape2.push_back(conv2.output().shape()[3] * conv1.dilation_w());

    auto rshape = graph.emplace<reshape>(dt_float32, conv2.output().shape(), reshapped_shape);
    auto tp = graph.emplace<transpose>(dt_float32, rshape->output().shape(), perm);
    auto rshape2 = graph.emplace<reshape>(dt_float32, tp->output().shape(), reshapped_shape2);
    auto p = graph.emplace<pad>(dt_float32, rshape2->output().shape(), crops, 0.f);
    rshape->input().connect(conv2.output());
    tp->input().connect(rshape->output());
    rshape2->input().connect(tp->output());
    p->input().connect(rshape2->output());

    return p;
}
}

bool eliminate_dilated_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto conv = node_cast<conv2d>(node))
    {
        if ((conv->dilation_h() != 1 || conv->dilation_w() != 1)
            && conv->stride_h() == 1 && conv->stride_w() == 1
            && conv->input().shape()[0] == 1)
        {
            context.inputs.emplace_back(&conv->input());
            context.outputs.emplace_back(&conv->output());

            context.matched_nodes.emplace_back(conv);
            return true;
        }
    }

    return false;
}

void eliminate_dilated_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[0]);

    auto [paddings, crops] = get_pad_crop(old_conv);
    auto [p1, rshape1] = space_to_batch(old_conv, paddings, context.graph);
    auto conv = context.graph.emplace<conv2d>(rshape1->output().shape(), old_conv.weights(), old_conv.bias(), old_conv.groups(),
        padding::zero(), padding::zero(), old_conv.stride_h(), old_conv.stride_w(), 1, 1, old_conv.fused_activation());
    conv->name(old_conv.name());
    auto p2 = batch_to_space(old_conv, *conv, crops, context.graph);
    conv->input().connect(rshape1->output());

    p1->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(p2->output());
}
