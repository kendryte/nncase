/* Copyright 2019-2021 Canaan Inc.
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
#include "nncase/runtime/datatypes.h"
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/space_to_batch_transform.h>
#include <nncase/transforms/pass.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool space_to_batch_to_pad::on_try_match(node &node, transform_context &context)
{
    space_to_batch *s2b;

    if ((s2b = node_cast<space_to_batch>(node))
        && s2b->block_size_h() == 1 && s2b->block_size_w() == 1)
    {
        context.inputs.emplace_back(&s2b->input());
        context.outputs.emplace_back(&s2b->output());
        context.matched_nodes.emplace_back(s2b);

        return true;
    }
    return false;
}

void space_to_batch_to_pad::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &s2b = static_cast<space_to_batch &>(*context.matched_nodes[0]);

    xt::svector<padding> paddings { { 0, 0 }, { 0, 0 }, s2b.padding_h(), s2b.padding_w() };
    auto p = context.graph.emplace<pad>(s2b.input().type(), output.shape(), paddings, pad_mode_t::pad_constant, s2b.pad_value());
    p->name(s2b.name());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(p->output());
}
