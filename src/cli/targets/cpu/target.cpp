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
#include <hlir/opcode.h>
#include <hlir/transforms/neutral/add_quant_checkpoints.h>
#include <hlir/transforms/neutral/dequantize_motion.h>
#include <hlir/transforms/neutral/fold_constant.h>
#include <hlir/transforms/neutral/fold_pad.h>
#include <hlir/transforms/neutral/fold_quantize.h>
#include <hlir/transforms/neutral/fold_reshape.h>
#include <hlir/transforms/neutral/fold_transpose.h>
#include <hlir/transforms/neutral/fuse_clamp.h>
#include <hlir/transforms/neutral/fuse_pad.h>
#include <hlir/transforms/neutral/fuse_unary.h>
#include <hlir/transforms/neutral/fused_unary_to_lookup1d.h>
#include <hlir/transforms/neutral/quantized_binary.h>
#include <hlir/transforms/neutral/quantized_conv2d.h>
#include <hlir/transforms/neutral/quantized_matmul.h>
#include <hlir/transforms/neutral/simplify_reduce.h>
#include <hlir/transforms/neutral/transpose_motion.h>
#include <llir/transforms/neutral/fold_memorycopy.h>
#include <scheduler/main_memory_allocator.h>

using namespace nncase;
using namespace nncase::scheduler;

namespace nncase
{
namespace codegen
{
    void register_netural_emitters();
}
}

namespace nncase
{
namespace llir
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
    using namespace nncase::llir;

    register_neutral_evaluators();
}

void cpu_target::add_default_transforms(hlir::transforms::pass &pass)
{
    using namespace nncase::hlir;
    using namespace nncase::hlir::transforms;

    pass.emplace<fold_constant_transform>();
    pass.emplace<fold_nop_pad_transform>();
    pass.emplace<fold_nop_reshape_transform>();
    pass.emplace<fold_nop_transpose_transform>();
    pass.emplace<fold_pad_pad_transform>();
    pass.emplace<fold_pad_strided_slice_transform>();
    pass.emplace<fold_quantize_transform>();
    pass.emplace<fold_reshape_transform>();
    pass.emplace<fold_transpose_transform>();
    pass.emplace<fuse_pad_conv2d_transform>();
    pass.emplace<fuse_clamp_conv2d_transform>();
    pass.emplace<fuse_clamp_binary_transform>();
    pass.emplace<strided_slice_to_pad_transform>();
    pass.emplace<transpose_binary_motion_transform>();
    pass.emplace<transpose_constant_binary_motion_transform>();
    pass.emplace<transpose_concat_motion_transform>();
    pass.emplace<transpose_pad_motion_transform>();
    pass.emplace<transpose_reduce_motion_transform>();
    pass.emplace<transpose_unary_motion_transform>();
    pass.emplace<transpose_to_reshape_transform>();
    pass.emplace<simplify_reduce_transform>();

    if (options_.inference_type == "uint8")
    {
        pass.emplace<fuse_one_unary_transform>();
        pass.emplace<fuse_one_binary_transform>();
        pass.emplace<fuse_two_fused_unary_transform>();
        pass.emplace<fuse_one_fused_unary_with_binary_transform>();
        pass.emplace<fuse_two_fused_unary_with_binary_transform>();
    }
}

void nncase::cpu_target::optimize_target_independent(hlir::transforms::pass_manager &pass_mgr)
{
    using namespace nncase::hlir;
    using namespace nncase::hlir::transforms;

    pass p;
    add_default_transforms(p);
    pass_mgr.add_pass(std::move(p));
}

void nncase::cpu_target::optimize_target_dependent(hlir::transforms::pass_manager &pass_mgr)
{
}

void nncase::cpu_target::add_quantization_checkpoints(hlir::transforms::pass_manager &pass_mgr)
{
    using namespace nncase::hlir;
    using namespace nncase::hlir::transforms;

    {
        pass p;
        add_default_transforms(p);
        pass_mgr.add_pass(std::move(p));
    }
    {
        pass p;
        if (options().quantize_binary)
            p.emplace<add_quant_checkpoints_transform>(add_quant_checkpoints_transform { op_conv2d, op_matmul, op_binary, op_fused_unary });
        else
            p.emplace<add_quant_checkpoints_transform>(add_quant_checkpoints_transform { op_conv2d, op_matmul, op_fused_unary });
        pass_mgr.add_pass(std::move(p));
    }
}

void nncase::cpu_target::optimize_quantize(hlir::quantizer &quantizer, hlir::transforms::pass_manager &pass_mgr)
{
    using namespace nncase::hlir;
    using namespace nncase::hlir::transforms;

    pass p;
    if (options_.input_type == "uint8")
        p.emplace<fold_input_quantize_transform>(quantizer);
    p.emplace<dequantize_reshape_motion_transform>();
    p.emplace<dequantize_transpose_motion_transform>();
    p.emplace<dequantize_pad_motion_transform>();
    p.emplace<dequantize_strided_slice_motion_transform>();
    p.emplace<dequantize_resize_image_motion_transform>();
    p.emplace<quantized_conv2d_transform>(quantizer);
    p.emplace<quantized_matmul_transform>(quantizer);

    if (options().quantize_binary)
        p.emplace<quantized_binary_transform>(quantizer);

    p.emplace<fused_unary_to_lookup1d_transform>(quantizer);
    add_default_transforms(p);
    pass_mgr.add_pass(std::move(p));
}

void nncase::cpu_target::add_quantization_broadcast(std::unordered_set<hlir::node_opcode> &opcodes)
{
    using namespace hlir;
    opcodes.emplace(op_input_node);
    opcodes.emplace(op_concat);
    opcodes.emplace(op_reshape);
    opcodes.emplace(op_transpose);
    opcodes.emplace(op_pad);
    opcodes.emplace(op_strided_slice);
    opcodes.emplace(op_resize_image);
    opcodes.emplace(op_reduce_window2d);
}

void nncase::cpu_target::optimize_llir(llir::transforms::pass_manager &pass_mgr)
{
    using namespace nncase::llir;
    using namespace nncase::llir::transforms;

    pass p;
    p.emplace<fold_memorycopy_transform>();
    pass_mgr.add_pass(std::move(p));
}
