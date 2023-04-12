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
#include "../module_builder.h"
#include <nncase/runtime/vulkan/runtime_types.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::vulkan;
using namespace nncase::ir;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;
using namespace nlohmann;

void vulkan_module_builder::emit(unary &node) {
    auto len = xt::compute_size(node.output().shape());
    json ctx;
    ctx["length"] = len;
    ctx["unary_op"] = unary_op_to_string(node.unary_op());
    auto shader = compile_shader(node, "unary.comp", ctx);

    ldbuf(allocation(node.input()).runtime_type());
    ldbuf(allocation(node.output()).runtime_type());

    ldpipeline_op_t ldp_op;
    ldp_op.buffers = 2;
    ldp_op.shader_type = shader_type_t::compute;
    ldpipeline(node, 0, ldp_op, shader);

    dispatch_op_t dop;
    dop.x = (uint32_t)len;
    dop.y = 1;
    dop.z = 1;
    text_writer().write(dop);

    descriptor_sets_ += 1;
    descriptors_ += 2;
}
