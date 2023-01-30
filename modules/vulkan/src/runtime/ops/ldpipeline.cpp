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
#include "../runtime_function.h"
#include "../vulkan_error.h"
#include <vulkan/vulkan.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void>
vulkan_runtime_function::visit(const ldpipeline_op_t &op) noexcept {
    auto code = module()
                    .shader()
                    .subspan(op.shader_start, op.shader_size)
                    .as_span<const uint32_t>();
    vk::ShaderModuleCreateInfo shader_cinfo({}, op.shader_size, code.data());
    try_var(shader,
            vk::to_result(module().device().createShaderModule(shader_cinfo)));

    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings(
        (size_t)op.buffers);
    for (int32_t i = (int32_t)op.buffers - 1; i >= 0; i--) {
        auto &binding = layout_bindings[i];
        binding.setBinding((uint32_t)i);
        binding.setDescriptorCount(1);
        binding.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo desc_layout_cinfo({}, layout_bindings);
    try_var(desc_layout,
            vk::to_result(module().device().createDescriptorSetLayout(
                desc_layout_cinfo)));

    vk::PipelineLayoutCreateInfo ppl_layout_cinfo({}, desc_layout);
    try_var(ppl_layout, vk::to_result(module().device().createPipelineLayout(
                            ppl_layout_cinfo)));

    if (op.shader_type != shader_type_t::compute)
        return err(std::errc::not_supported);

    vk::ComputePipelineCreateInfo comp_ppl_cinfo(
        {}, {{}, vk::ShaderStageFlagBits::eCompute, shader, "main"},
        ppl_layout);
    try_var(pipeline, vk::to_result(module().device().createComputePipeline(
                          {}, comp_ppl_cinfo)));
    module().device().destroyShaderModule(shader);

    vk::DescriptorSetAllocateInfo desc_alloc_info(module().buffer_desc_pool(),
                                                  desc_layout);
    try_var(desc_sets, vk::to_result(module().device().allocateDescriptorSets(
                           desc_alloc_info)));

    std::vector<vk::DescriptorBufferInfo> buffer_infos((size_t)op.buffers);
    std::vector<vk::WriteDescriptorSet> write_descs(buffer_infos.size());
    std::vector<vk::BufferMemoryBarrier> bm_barriers(buffer_infos.size());
    for (int32_t i = (int32_t)op.buffers - 1; i >= 0; i--) {
        auto &info = buffer_infos[i];
        try_var(buffer_ref, pop_buffer_ref());
        info.setBuffer(buffer_ref.buffer);
        info.setOffset(buffer_ref.start);
        info.setRange(buffer_ref.size);

        auto &write_desc = write_descs[i];
        write_desc.setBufferInfo(info);
        write_desc.setDescriptorCount(1);
        write_desc.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        write_desc.setDstArrayElement(0);
        write_desc.setDstBinding((uint32_t)i);
        write_desc.setDstSet(desc_sets[0]);
    }

    module().device().updateDescriptorSets(write_descs, {});

    try_(module().add_pipeline(pipeline, ppl_layout, desc_layout));
    cmd_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ppl_layout,
                                   0, desc_sets, {});
    return ok();
}
