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
#include "k210_target.h"
#include <nncase/codegen/k210/module_builder.h>
#include <nncase/schedule/k210/kpu_buffer_allocator.h>
#include <nncase/transforms/k210/fake_kpu_conv2d.h>
#include <nncase/transforms/k210/fold_kpu_upload.h>
#include <nncase/transforms/k210/fuse_kpu_conv2d_pool.h>
#include <nncase/transforms/k210/fuse_kpu_download.h>
#include <nncase/transforms/k210/kpu_conv2d.h>
#include <nncase/transforms/k210/strided_slice_motion.h>
#include <nncase/transforms/neutral/add_quant_checkpoints.h>
#include <nncase/transforms/neutral/eliminate_dilated_conv2d.h>
#include <nncase/transforms/neutral/fold_pad.h>
#include <nncase/transforms/neutral/fold_quantize.h>
#include <nncase/transforms/neutral/fold_transpose.h>
#include <nncase/transforms/neutral/fuse_pad.h>
#include <nncase/transforms/neutral/transpose_motion.h>
#include <nncase/transforms/pass.h>

#if defined(_MSC_VER)
#define K210_TARGET_API __declspec(dllexport)
#else
#define K210_TARGET_API
#endif

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::schedule::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;
using namespace nncase::targets;
using namespace nncase::runtime;

extern "C"
{
    K210_TARGET_API target *create_target()
    {
        return new k210_target();
    }
}

std::unique_ptr<codegen::module_builder> k210_target::create_module_builder(const module_type_t &type, std::string_view module_name, const codegen::module_builder_params &params)
{
    if (type == runtime::k210::k210_module_type)
        return codegen::create_k210_module_builder(module_name, params);
    return neutral_target::create_module_builder(type, module_name, params);
}

void k210_target::register_allocators(const module_type_t &type, allocator_map_t &allocators, std::vector<std::shared_ptr<buffer_allocator>> &allocator_holders)
{
    if (type == runtime::k210::k210_module_type)
    {
        allocators.emplace(mem_input, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_output, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_rdata, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_data, allocator_holders.emplace_back(std::make_shared<kpu_buffer_allocator>()).get());
    }
    else
    {
        neutral_target::register_allocators(type, allocators, allocator_holders);
    }
}

void k210_target::register_quantize_annotation_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr)
{
    pass p("annotate_kpu");
    p.emplace<eliminate_dilated_conv2d_transform>();
    p.emplace<fake_kpu_conv2d_transform>();
    p.emplace<strided_slice_motion_transform>();
    p.emplace<fuse_fake_kpu_conv2d_strided_slice_transform>();
    add_default_transforms(p);
    pass_mgr.add_pass(std::move(p));
}

void k210_target::register_target_dependent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr)
{
}
