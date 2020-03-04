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
#include <llir/ops/memory_copy.h>
#include <llir/transforms/neutral/fold_memorycopy.h>
#include <llir/visitor.h>

using namespace nncase;
using namespace nncase::llir;
using namespace nncase::llir::transforms;

bool fold_memorycopy_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_memory_copy)
    {
        auto &mc = static_cast<memory_copy &>(node);

        context.inputs.emplace_back(&mc.input());
        context.outputs.emplace_back(&mc.output());

        context.matched_nodes.emplace_back(&mc);
        return true;
    }

    return false;
}

void fold_memorycopy_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
