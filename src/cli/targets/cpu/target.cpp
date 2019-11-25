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
#include <ir/opcode.h>
#include <scheduler/main_memory_allocator.h>
#include <transforms/neutral/add_quant_checkpoints.h>
#include <transforms/neutral/dequantize_motion.h>
#include <transforms/neutral/fold_constant.h>
#include <transforms/neutral/fold_pad.h>
#include <transforms/neutral/fold_quantize.h>
#include <transforms/neutral/fold_reshape.h>
#include <transforms/neutral/fold_transpose.h>
#include <transforms/neutral/fuse_pad.h>
#include <transforms/neutral/quantized_binary.h>
#include <transforms/neutral/quantized_conv2d.h>
#include <transforms/neutral/quantized_matmul.h>
#include <transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::scheduler;
using namespace nncase::transforms;

namespace nncase
{
namespace codegen
{
    void register_netural_emitters();
}
}

namespace nncase
{
namespace ir
{
    void register_neutral_evaluators();
}
}

void nncase::cpu_target::fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<memory_allocator>> &allocator_holders)
{
    allocators.emplace(mem_const, allocator_holders.emplace_back(std::make_unique<main_memory_allocator>()).get());
    allocators.emplace(mem_main, allocator_holders.emplace_back(std::make_unique<main_memory_allocator>()).get());
}

void nncase::cpu_target::registry_codegen_ops()
{
    using namespace nncase::codegen;

    register_netural_emitters();
}

void nncase::cpu_target::registry_evaluator_ops()
{
    using namespace nncase::ir;

    register_neutral_evaluators();
}

void nncase::cpu_target::add_default_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    transforms.emplace_back(new fold_constant_transform(*this));
    transforms.emplace_back(new fold_nop_pad_transform());
    transforms.emplace_back(new fold_nop_reshape_transform());
    transforms.emplace_back(new fold_nop_transpose_transform());
    transforms.emplace_back(new fold_pad_strided_slice_transform());
    transforms.emplace_back(new fold_quantize_transform());
    transforms.emplace_back(new fold_reshape_transform());
    transforms.emplace_back(new fold_transpose_transform());
    transforms.emplace_back(new fuse_pad_conv2d_transform());
    transforms.emplace_back(new strided_slice_to_pad_transform());
    transforms.emplace_back(new transpose_binary_motion_transform());
    transforms.emplace_back(new transpose_constant_binary_motion_transform());
    transforms.emplace_back(new transpose_concat_motion_transform());
    transforms.emplace_back(new transpose_pad_motion_transform());
    transforms.emplace_back(new transpose_reduce_motion_transform());
    transforms.emplace_back(new transpose_unary_motion_transform());
    transforms.emplace_back(new transpose_to_reshape_transform());
}

void nncase::cpu_target::add_optimize1_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
}

void nncase::cpu_target::add_optimize2_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
}

void nncase::cpu_target::add_quantization_checkpoint_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    transforms.emplace_back(new add_quant_checkpoints_transform({ op_conv2d, op_matmul, op_binary }));
}

void nncase::cpu_target::add_quantization_transforms(ir::quantizer &quantizer, std::vector<std::unique_ptr<transform>> &transforms)
{
    if (options_.input_type == "uint8")
        transforms.emplace_back(new fold_input_quantize_transform(quantizer));
    transforms.emplace_back(new dequantize_transpose_motion_transform());
    transforms.emplace_back(new dequantize_pad_motion_transform());
    transforms.emplace_back(new dequantize_strided_slice_motion_transform());
    transforms.emplace_back(new dequantize_resize_image_motion_transform());
    transforms.emplace_back(new quantized_conv2d_transform(quantizer));
    transforms.emplace_back(new quantized_matmul_transform(quantizer));
    transforms.emplace_back(new quantized_binary_transform(quantizer));
}

void nncase::cpu_target::add_quantization_broadcast(std::unordered_set<ir::node_opcode> &opcodes)
{
    using namespace ir;
    opcodes.emplace(op_input_node);
    opcodes.emplace(op_fake_quantize);
    opcodes.emplace(op_fake_dequantize);
    opcodes.emplace(op_concat);
    opcodes.emplace(op_reshape);
    opcodes.emplace(op_transpose);
    opcodes.emplace(op_pad);
    opcodes.emplace(op_strided_slice);
    opcodes.emplace(op_resize_image);
}
