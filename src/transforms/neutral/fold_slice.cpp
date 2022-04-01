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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_slice.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

static inline bool no_slice(slice *rp, size_t i)
{
    return (rp->begin()[i] == 0) && (rp->end()[i] == rp->input().shape()[i]) && (rp->strides()[i] == 1);
}

static inline bool can_merge(slice *rp1, slice *rp2)
{
    if (std::any_of(rp1->strides().begin(), rp1->strides().end(), [](int32_t dim) { return dim < 0; }) || std::any_of(rp2->strides().begin(), rp2->strides().end(), [](int32_t dim) { return dim < 0; }))
        return false;

    bool ret = true;
    for (size_t i = 0; i < rp1->strides().size(); i++)
    {
        ret &= (no_slice(rp1, i) || no_slice(rp2, i));
        if (not ret)
            break;
    }
    return ret;
}

bool fold_slice_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto rp1 = node_cast<slice>(node))
    {
        if (auto rp2 = try_get_direct_child<slice>(*rp1))
        {
            if (can_merge(rp1, rp2))
            {
                context.inputs.emplace_back(&rp1->input());
                context.outputs.emplace_back(&rp2->output());

                context.matched_nodes.emplace_back(rp1);
                context.matched_nodes.emplace_back(rp2);

                return true;
            }
        }
    }

    return false;
}

void fold_slice_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto rp1 = node_cast<slice>(*context.matched_nodes[0]);
    auto rp2 = node_cast<slice>(*context.matched_nodes[1]);

    axis_t new_begin(rp1->begin());
    axis_t new_end(rp1->begin());
    axis_t new_strides(rp1->strides());
    for (auto i = 0; i < int64_t(rp1->begin().size()); ++i)
    {
        new_begin[i] = no_slice(rp1, i) ? rp2->begin()[i] : rp1->begin()[i];
        new_end[i] = no_slice(rp1, i) ? rp2->end()[i] : rp1->end()[i];
        new_strides[i] = no_slice(rp1, i) ? rp2->strides()[i] : rp1->strides()[i];
    }

    auto new_rp = context.graph.emplace<slice>(
        output.type(), output.shape(), new_begin, new_end, new_strides, 0, 0, 0, 0, 0);
    new_rp->name(rp2->name() + "_F");

    new_rp->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(new_rp->output());
}

bool fold_nop_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto rp1 = node_cast<slice>(node))
    {
        if (rp1->input().shape() == rp1->output().shape() && rp1->strides() == axis_t { 1, 1, 1, 1 })
        {
            context.inputs.emplace_back(&rp1->input());
            context.outputs.emplace_back(&rp1->output());

            return true;
        }
    }

    return false;
}

void fold_nop_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
