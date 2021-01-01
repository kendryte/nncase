/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_clamp.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_nop_clamp_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cl = node_cast<clamp>(node))
    {
        if (auto low = try_get_direct_parent<constant>(*cl, 1))
        {
            if (auto high = try_get_direct_parent<constant>(*cl, 2))
            {
                if (xt::compute_size(low->output().shape()) == 1
                    && xt::compute_size(high->output().shape()) == 1)
                {
                    auto low_v = reinterpret_cast<const float *>(low->data().data())[0];
                    auto high_v = reinterpret_cast<const float *>(high->data().data())[0];

                    if (value_range<float>::full() == value_range<float> { low_v, high_v })
                    {
                        context.inputs.emplace_back(&cl->input());
                        context.outputs.emplace_back(&cl->output());

                        context.matched_nodes.emplace_back(low);
                        context.matched_nodes.emplace_back(high);
                        context.matched_nodes.emplace_back(cl);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void fold_nop_clamp_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
