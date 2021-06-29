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

// static bool is_strided_slice(nncase::ir::slice& node){
//     return true;
// }

bool fold_slice_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto rp1 = node_cast<slice>(node))
    {
        // if(is_strided_slice(*rp1)){            return false;        }
        if (auto rp2 = try_get_direct_child<slice>(*rp1))
        {
            // if(is_strided_slice(*rp2)){return false;}
            context.inputs.emplace_back(&rp1->input());
            context.outputs.emplace_back(&rp2->output());

            context.matched_nodes.emplace_back(rp1);
            context.matched_nodes.emplace_back(rp2);
            return true;
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
    for (auto i = 0; i < int64_t(rp1->begin().size()); ++i)
    {
        // new_begin.emplace_back(rp1->begin()[i] + rp2->begin()[i]);
        // new_end.emplace_back(rp1->end()[i] + rp2->end()[i]);
        new_begin[i] = rp1->begin()[i] + rp2->begin()[i];
        new_end[i] = rp1->begin()[i] + rp2->end()[i];
    }
    // auto begin = rp1->begin() + rp2->begin();
    // auto end = rp1->end()+ rp2->end();

    auto new_rp = context.graph.emplace<slice>(
        output.type(), output.shape(), new_begin, new_end);
    new_rp->name(rp2->name() + "_F");

    new_rp->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(new_rp->output());
}
