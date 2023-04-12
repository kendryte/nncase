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
#include "module_builder.h"
#include "templates/template.h"
#include <vulkan/vulkan.hpp>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::vulkan;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

std::unique_ptr<module_builder>
codegen::create_vulkan_module_builder(std::string_view module_name,
                                      const module_builder_params &params) {
    return std::make_unique<vulkan_module_builder>(module_name, params);
}

vulkan_module_builder::vulkan_module_builder(
    std::string_view module_name, const module_builder_params &params)
    : module_builder(256, module_name, params) {}

module_type_t vulkan_module_builder::module_type() const noexcept {
    return vulkan_module_type;
}

uint32_t vulkan_module_builder::module_version() const noexcept {
    return vulkan_module_version;
}

section_writer &vulkan_module_builder::text_writer() { return writer(".text"); }

section_writer &vulkan_module_builder::shader_writer() {
    return writer(SHADER_SECTION_NAME);
}

std::vector<uint32_t>
vulkan_module_builder::compile_shader(ir::node &node,
                                      const std::string &template_name,
                                      const nlohmann::json &context) {
    compile_options options{context};
    options.dump_asm = dump_asm_;
    options.dump_dir = dump_dir_;
    options.function_name = node.escaped_name();
    return render_and_compile(template_name, options);
}

void vulkan_module_builder::ldbuf(const memory_range &range) {
    ldbuf_op_t op;
    op.memory = range;
    text_writer().write(op);
}

void vulkan_module_builder::ldpipeline(ir::node &node, size_t shader_index,
                                       ldpipeline_op_t &op,
                                       const std::vector<uint32_t> &shader) {
    auto &tw = text_writer();
    for (auto in : node.inputs()) {
        ldbufbarrier_op_t op{};
        op.src_access_mask = (uint32_t)vk::AccessFlagBits::eShaderRead;
        op.dest_access_mask = 0;
        op.memory = allocation(*in).runtime_type();
        tw.write(op);
    }

    for (auto out : node.outputs()) {
        ldbufbarrier_op_t op{};
        op.src_access_mask = 0;
        op.dest_access_mask = (uint32_t)vk::AccessFlagBits::eShaderWrite;
        op.memory = allocation(*out).runtime_type();
        tw.write(op);
    }

    barrier_op_t bop{};
    bop.src_stage = (uint32_t)vk::PipelineStageFlagBits::eComputeShader;
    bop.dest_stage = (uint32_t)vk::PipelineStageFlagBits::eComputeShader;
    bop.buffer_barriers =
        (uint32_t)(node.inputs().size() + node.outputs().size());
    tw.write(bop);

    op.shader_start = 0;
    op.shader_size = static_cast<uint32_t>(shader.size() * sizeof(uint32_t));

    auto symbol = node.name() + ".shader" + std::to_string(shader_index);
    auto &sw = shader_writer();
    sw.add_symbol(symbol);
    sw.write_array<uint32_t>(shader);

    tw.add_symbol_ref(offsetof(ldpipeline_op_t, shader_start) * 8,
                      sizeof(op.shader_start) * 8, symbol);
    tw.write(op);
}

void vulkan_module_builder::begin_emit_function(
    [[maybe_unused]] const schedule::function_schedule_result &function) {
    set_current_entry_point(text_writer().position());
}

void vulkan_module_builder::end_emit_function(
    [[maybe_unused]] const schedule::function_schedule_result &function) {
    set_current_function_text_end(text_writer().position());
}

void vulkan_module_builder::emit(ir::node &node) {
#define DEFINE_OP(op)                                                          \
    if (node.runtime_opcode() == ir::op::opcode())                             \
        return emit(static_cast<op &>(node));
#include "ops.def"
#undef DEFINE_OP
    module_builder::emit(node);
}

void vulkan_module_builder::end_emit_module() {
    auto &sw = writer(DESCRIPTORS_SECTION_NAME);
    sw.write(descriptor_sets_);
    sw.write(descriptors_);
}
