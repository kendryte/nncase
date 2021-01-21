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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/call.h>
#include <nncase/ir/visitor.h>

using namespace nncase;
using namespace nncase::ir;

call::call(graph &target)
    : target_(target)
{
    size_t i = 0;
    for (auto &in : target_.inputs())
        add_input(in->name(), in->output().type(), in->output().shape());

    i = 0;
    for (auto &out : target_.outputs())
        add_output(out->name(), out->input().type(), out->input().shape());
}

input_connector &call::outer_connector(input_node &subgraph_input)
{
    auto it = std::find(target_.inputs().begin(), target_.inputs().end(), &subgraph_input);
    if (it == target_.inputs().end())
        throw std::invalid_argument("Cannot find input node in subgraph");
    return input_at(std::distance(target_.inputs().begin(), it));
}

input_connector &call::outer_connector(input_connector &subgraph_input)
{
    auto &owner = subgraph_input.connection()->owner();
    if (auto in = node_cast<input_node>(owner))
        return outer_connector(*in);
    else
        throw std::invalid_argument("Input connector is not exported");
}

output_connector &call::outer_connector(output_node &subgraph_output)
{
    auto it = std::find(target_.outputs().begin(), target_.outputs().end(), &subgraph_output);
    if (it == target_.outputs().end())
        throw std::invalid_argument("Cannot find output node in subgraph");
    return output_at(std::distance(target_.outputs().begin(), it));
}

output_connector &call::outer_connector(output_connector &subgraph_input)
{
    assert(subgraph_input.connections().size() == 1);
    auto &owner = subgraph_input.connections()[0]->owner();
    if (auto out = node_cast<output_node>(owner))
        return outer_connector(*out);
    else
        throw std::invalid_argument("Output connector is not exported");
}

bool call::properties_equal(node &other) const
{
    auto &r = static_cast<call &>(other);
    return &target() == &r.target();
}
