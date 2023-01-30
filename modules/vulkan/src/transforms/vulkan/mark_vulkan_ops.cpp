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
#include <nncase/ir/visitor.h>
#include <nncase/runtime/vulkan/runtime_module.h>
#include <nncase/transforms/vulkan/mark_vulkan_ops.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::runtime::vulkan;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::vulkan;

namespace {
std::unordered_set<node_opcode> supported_opcodes{op_unary};
}

bool mark_vulkan_ops_transform::on_try_match(node &node,
                                             transform_context &context) {
    if (supported_opcodes.contains(node.runtime_opcode()) &&
        node.module_type() != vulkan_module_type) {
        for (auto in : node.inputs())
            context.inputs.emplace_back(in);
        for (auto out : node.outputs())
            context.outputs.emplace_back(out);
        context.matched_nodes.emplace_back(&node);
        return true;
    }

    return false;
}

void mark_vulkan_ops_transform::process(transform_context &context) {
    context.matched_nodes.front()->module_type(vulkan_module_type);
}
