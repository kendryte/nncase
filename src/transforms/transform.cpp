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
#include "nncase/ir/quantizer.h"
#include <algorithm>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/transform.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool transform::try_match(node &node, transform_context &context)
{
    if (on_try_match(node, context))
    {
        if (!skip_self_contained_check())
        {
            for (auto &&node : context.matched_nodes)
            {
                // there exist input connectors out of the subgraph
                for (auto in : node->inputs())
                {
                    if (in->connection())
                    {
                        if (std::find(std::begin(context.inputs), std::end(context.inputs), in) == std::end(context.inputs)
                            && std::find(std::begin(context.matched_nodes), std::end(context.matched_nodes), &in->connection()->owner()) == std::end(context.matched_nodes))
                            return false;
                    }
                }

                // there exist output connectors out of the subgraph
                for (auto out : node->outputs())
                {
                    for (auto conn : out->connections())
                    {
                        if (std::find(std::begin(context.outputs), std::end(context.outputs), out) == std::end(context.outputs)
                            && std::find(std::begin(context.matched_nodes), std::end(context.matched_nodes), &conn->owner()) == std::end(context.matched_nodes))
                            return false;
                    }
                }
            }
        }

        return true;
    }

    return false;
}

bool transform::skip_self_contained_check() const noexcept
{
    return false;
}

namespace
{
class transform_apply_visitor : public dfs_ir_visitor
{
public:
    using dfs_ir_visitor::visit;
    ir::graph *graph;
    nncase::target *target;
    bool need_retry = false;
    transforms::transform *transform;

protected:
    bool visit(node &node) override
    {
        transform_context context { *graph, *target };
        if (transform->try_match(node, context))
        {
            transform->process(context);
            need_retry = true;
            return true;
        }

        return false;
    }
};
}

void nncase::ir::transforms::link(ir::output_connector &old_c, ir::output_connector &new_c, [[maybe_unused]] ir::quantizer *quantizer)
{
    new_c.attributes(old_c.attributes());

    if (old_c.attributes() & ir::cnctr_attr_need_quantize && quantizer)
        quantizer->set(new_c, quantizer->get(old_c));
}
