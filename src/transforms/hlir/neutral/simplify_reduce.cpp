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
#include <hlir/ops/reduce.h>
#include <hlir/ops/reshape.h>
#include <hlir/transforms/neutral/simplify_reduce.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool simplify_reduce_transform::on_try_match(node &node, transform_context &context)
{
    if (auto r = node_cast<reduce>(node))
    {
        if (r->axis().size() == 1 && !r->keep_dims())
        {
            auto axis = r->axis()[0];
            if (r->input().shape().size() > axis + 1
                && r->input().shape()[axis + 1] == 1)
            {
                context.inputs.emplace_back(&r->input());
                context.outputs.emplace_back(&r->output());

                context.matched_nodes.emplace_back(r);
                return true;
            }
        }
    }

    return false;
}

void simplify_reduce_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_r = *static_cast<reduce *>(context.matched_nodes[0]);

    auto axis = old_r.axis()[0];
    auto in_shape = old_r.input().shape();
    shape_t new_shape;
    new_shape.reserve(in_shape.size() - 1);
    for (size_t i = 0; i < axis + 1; i++)
        new_shape.push_back(in_shape[i]);
    for (size_t i = axis + 2; i < in_shape.size(); i++)
        new_shape.push_back(old_r.input().shape()[i]);

    auto rp = context.graph.emplace<reshape>(output.type(), output.shape(), new_shape);
    auto r = context.graph.emplace<reduce>(old_r.reduce_op(), rp->output().shape(), old_r.axis(), old_r.init_value(), true);
    r->name(old_r.name());
    r->input().connect(rp->output());

    rp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(r->output());
}
