/* Copyright 2019 Canaan Inc.
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
#include <scheduler/k210/kpu_memory_allocator.h>
#include <scheduler/main_memory_allocator.h>
#include <transforms/k210/fake_kpu_conv2d.h>
#include <transforms/k210/fake_piecewise_linear.h>
#include <transforms/k210/fold_kpu_upload.h>
#include <transforms/k210/fuse_kpu_download.h>
#include <transforms/k210/kpu_conv2d.h>
#include <transforms/k210/strided_slice_motion.h>
#include <transforms/neutral/add_quant_checkpoints.h>
#include <transforms/neutral/fold_pad.h>
#include <transforms/neutral/fold_quantize.h>
#include <transforms/neutral/fold_transpose.h>
#include <transforms/neutral/fuse_pad.h>
#include <transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::scheduler;
using namespace nncase::scheduler::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

namespace nncase
{
namespace codegen
{
    void register_k210_emitters();
}
}

namespace nncase
{
namespace ir
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
    using namespace nncase::ir;

    cpu_target::registry_evaluator_ops();
    register_k210_evaluators();
}

void nncase::k210_target::add_default_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    cpu_target::add_default_transforms(transforms);
}

void nncase::k210_target::add_optimize1_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    cpu_target::add_optimize1_transforms(transforms);
}

void nncase::k210_target::add_optimize2_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    cpu_target::add_optimize2_transforms(transforms);
    transforms.emplace_back(new fake_kpu_conv2d_transform());
    transforms.emplace_back(new strided_slice_motion_transform());
    transforms.emplace_back(new fuse_fake_kpu_conv2d_strided_slice_transform());
    transforms.emplace_back(new fuse_fake_kpu_conv2d_reduce_window2d_transform());
    transforms.emplace_back(new binary_to_fake_piecewise_linear_transform());
    transforms.emplace_back(new fake_piecewise_linear_binary_transform());
    transforms.emplace_back(new fuse_fake_kpu_conv2d_piecewise_linear_transform());
}

void nncase::k210_target::add_quantization_checkpoint_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    cpu_target::add_quantization_checkpoint_transforms(transforms);
    transforms.emplace_back(new revert_piecewise_linear_transform());
    transforms.emplace_back(new add_quant_checkpoints_transform({ op_k210_fake_kpu_conv2d }));
}

void nncase::k210_target::add_quantization_transforms(ir::quantizer &quantizer, std::vector<std::unique_ptr<transform>> &transforms)
{
    transforms.emplace_back(new kpu_conv2d_transform(quantizer));
    transforms.emplace_back(new fold_kpu_upload_transform());
    transforms.emplace_back(new fuse_kpu_download_transform());
    transforms.emplace_back(new fold_input_kpu_upload_transform());
    cpu_target::add_quantization_transforms(quantizer, transforms);
}
