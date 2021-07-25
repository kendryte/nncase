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
#include "../runtime_module.h"
#include "../vulkan_error.h"
#include <vulkan/vulkan.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> vulkan_runtime_module::visit(const ldpipeline_op_t &op) noexcept
{
    auto code = shader_.subspan(op.shader_start, op.shader_size).as_span<const uint32_t>();
    vk::ShaderModuleCreateInfo shader_cinfo({}, op.shader_size, code.data());
    try_var(shader, vk::to_result(device_.createShaderModule(shader_cinfo)));

    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings((size_t)op.buffers);
    for (int32_t i = (int32_t)op.buffers - 1; i >= 0; i--)
    {
        auto &binding = layout_bindings[i];
        binding.setBinding((uint32_t)i);
        binding.setDescriptorCount(1);
        binding.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo desc_layout_cinfo({}, layout_bindings);
    try_var(desc_layout, vk::to_result(device_.createDescriptorSetLayout(desc_layout_cinfo)));

    vk::PipelineLayoutCreateInfo ppl_layout_cinfo({}, desc_layout);
    try_var(ppl_layout, vk::to_result(device_.createPipelineLayout(ppl_layout_cinfo)));

    if (op.shader_type != shader_type_t::compute)
        return err(std::errc::not_supported);

    vk::ComputePipelineCreateInfo comp_ppl_cinfo({}, { {}, vk::ShaderStageFlagBits::eCompute, shader, "main" }, ppl_layout);
    try_var(pipeline, vk::to_result(device_.createComputePipeline({}, comp_ppl_cinfo)));

    vk::DescriptorSetAllocateInfo desc_alloc_info(buffer_desc_pool_, desc_layout);
    try_var(desc_sets, vk::to_result(device_.allocateDescriptorSets(desc_alloc_info)));

    std::vector<vk::DescriptorBufferInfo> buffer_infos((size_t)op.buffers);
    std::vector<vk::WriteDescriptorSet> write_descs(buffer_infos.size());
    for (int32_t i = (int32_t)op.buffers - 1; i >= 0; i--)
    {
        auto &info = buffer_infos[i];
        try_var(buffer, pop_buffer());
        info.setBuffer(std::move(buffer));
        info.setOffset(0);
        info.setRange(VK_WHOLE_SIZE);

        auto &write_desc = write_descs[i];
        write_desc.setBufferInfo(info);
        write_desc.setDescriptorCount(1);
        write_desc.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        write_desc.setDstArrayElement(0);
        write_desc.setDstBinding((uint32_t)i);
        write_desc.setDstSet(desc_sets[0]);
    }

    device_.updateDescriptorSets(write_descs, {});

    cmd_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ppl_layout, 0, desc_sets, {});
    return ok();
}
