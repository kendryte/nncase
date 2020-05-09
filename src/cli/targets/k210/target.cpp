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
#include "target.h"
#include <hlir/transforms/k210/fake_kpu_conv2d.h>
#include <hlir/transforms/k210/fold_kpu_upload.h>
#include <hlir/transforms/k210/fuse_kpu_download.h>
#include <hlir/transforms/k210/kpu_conv2d.h>
#include <hlir/transforms/k210/matmul_to_fake_kpu_conv2d.h>
#include <hlir/transforms/k210/strided_slice_motion.h>
#include <hlir/transforms/neutral/add_quant_checkpoints.h>
#include <hlir/transforms/neutral/eliminate_dilated_conv2d.h>
#include <hlir/transforms/neutral/fold_pad.h>
#include <hlir/transforms/neutral/fold_quantize.h>
#include <hlir/transforms/neutral/fold_transpose.h>
#include <hlir/transforms/neutral/fuse_pad.h>
#include <hlir/transforms/neutral/transpose_motion.h>
#include <hlir/transforms/k210/fuse_kpu_conv2d_pool.h>
#include <scheduler/k210/kpu_memory_allocator.h>
#include <scheduler/main_memory_allocator.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::scheduler;
using namespace nncase::scheduler::k210;
using namespace nncase::hlir::transforms;
using namespace nncase::hlir::transforms::k210;

namespace nncase
{
namespace codegen
{
    void register_k210_emitters();
}
}

namespace nncase
{
namespace llir
{
    void register_k210_evaluators();
}
}

void nncase::k210_target::fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<memory_allocator>> &allocator_holders)
{
    cpu_target::fill_allocators(allocators, allocator_holders);
    allocators.emplace(mem_k210_kpu, allocator_holders.emplace_back(std::make_unique<kpu_memory_allocator>()).get());
}

void nncase::k210_target::registry_codegen_ops()
{
    using namespace nncase::codegen;

    cpu_target::registry_codegen_ops();
    register_k210_emitters();
}

void nncase::k210_target::registry_evaluator_ops()
{
    using namespace nncase::llir;

    cpu_target::registry_evaluator_ops();
    register_k210_evaluators();
}

void nncase::k210_target::optimize_target_independent(hlir::transforms::pass_manager &pass_mgr)
{
    cpu_target::optimize_target_independent(pass_mgr);
}

void nncase::k210_target::optimize_target_dependent(hlir::transforms::pass_manager &pass_mgr)
{
    pass p;
    p.emplace<eliminate_dilated_conv2d_transform>();
    p.emplace<fake_kpu_conv2d_transform>();
    p.emplace<matmul_to_fake_kpu_conv2d_transform>();
    p.emplace<strided_slice_motion_transform>();
    p.emplace<fuse_fake_kpu_conv2d_strided_slice_transform>();
    add_default_transforms(p);
    pass_mgr.add_pass(std::move(p));
}

void nncase::k210_target::add_quantization_checkpoints(hlir::transforms::pass_manager &pass_mgr)
{
    pass p;
    p.emplace<add_quant_checkpoints_transform>(op_k210_fake_kpu_conv2d);
    pass_mgr.add_pass(std::move(p));
    cpu_target::add_quantization_checkpoints(pass_mgr);
}

void nncase::k210_target::optimize_quantize(hlir::quantizer &quantizer, hlir::transforms::pass_manager &pass_mgr)
{
    {
        pass p;
        p.emplace<kpu_conv2d_transform>(quantizer);
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        pass p;
        p.emplace<fuse_kpu_conv2d_pool_transform>();
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        pass p;
        p.emplace<fold_kpu_upload_transform>();
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        cpu_target::optimize_quantize(quantizer, pass_mgr);

        pass p;
        p.emplace<fuse_kpu_download_transform>();
        p.emplace<fold_input_kpu_upload_transform>();
        add_default_transforms(p);
        pass_mgr.add_pass(std::move(p));
    }
}
