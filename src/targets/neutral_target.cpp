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
#include <nncase/ir/opcode.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <nncase/schedule/buffer_allocator.h>
#include <nncase/targets/neutral_target.h>
#include <nncase/transforms/neutral/add_quant_checkpoints.h>
#include <nncase/transforms/neutral/add_quant_motion.h>
#include <nncase/transforms/neutral/binary_motion.h>
#include <nncase/transforms/neutral/bitcast_motion.h>
#include <nncase/transforms/neutral/dequantize_motion.h>
#include <nncase/transforms/neutral/fix_output_shape.h>
#include <nncase/transforms/neutral/fix_tflite_error_shape.h>
#include <nncase/transforms/neutral/fold_bitcast.h>
#include <nncase/transforms/neutral/fold_constant.h>
#include <nncase/transforms/neutral/fold_conv2d_binary.h>
#include <nncase/transforms/neutral/fold_convert.h>
#include <nncase/transforms/neutral/fold_dilated_conv2d.h>
#include <nncase/transforms/neutral/fold_instancenorm.h>
#include <nncase/transforms/neutral/fold_layernorm.h>
#include <nncase/transforms/neutral/fold_matmul_add.h>
#include <nncase/transforms/neutral/fold_pad.h>
#include <nncase/transforms/neutral/fold_quantize.h>
#include <nncase/transforms/neutral/fold_slice.h>
#include <nncase/transforms/neutral/fold_transpose.h>
#include <nncase/transforms/neutral/fuse_clamp.h>
#include <nncase/transforms/neutral/fuse_pad.h>
#include <nncase/transforms/neutral/fuse_unary.h>
#include <nncase/transforms/neutral/fused_unary_to_lookup1d.h>
#include <nncase/transforms/neutral/global_reduce_window_to_reduce.h>
#include <nncase/transforms/neutral/matmul_to_conv2d.h>
#include <nncase/transforms/neutral/quantize_motion.h>
#include <nncase/transforms/neutral/remove_binary.h>
#include <nncase/transforms/neutral/simplify_reduce.h>
#include <nncase/transforms/neutral/space_to_batch_transform.h>
#include <nncase/transforms/neutral/split_to_slice.h>
#include <nncase/transforms/neutral/take_to_slice.h>
#include <nncase/transforms/neutral/transpose_motion.h>
#include <nncase/transforms/pass.h>

using namespace nncase;
using namespace nncase::targets;
using namespace nncase::ir::transforms;
using namespace nncase::schedule;

namespace nncase::codegen
{
void register_neutral_emitters();
}

namespace nncase::ir
{
void register_neutral_evaluators();
}

void neutral_target::register_allocators(const module_type_t &type, allocator_map_t &allocators, std::vector<std::shared_ptr<buffer_allocator>> &allocator_holders)
{
    if (type == runtime::stackvm::stackvm_module_type)
    {
        allocators.emplace(mem_input, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_output, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_rdata, allocator_holders.emplace_back(std::make_shared<linear_buffer_allocator>()).get());
        allocators.emplace(mem_data, allocator_holders.emplace_back(std::make_shared<first_fit_allocator>()).get());
    }
    else
    {
        throw std::runtime_error(std::string("Allocators for module ") + "[" + type.data() + "] is not found");
    }
}

void neutral_target::register_evaluator_ops()
{
    using namespace nncase::ir;

    register_neutral_evaluators();
}

void neutral_target::add_default_transforms(ir::transforms::transform_pass &pass, [[maybe_unused]] bool add_constant_folding)
{
    using namespace nncase::ir;
    using namespace nncase::ir::transforms;

    if (add_constant_folding)
        pass.emplace<fold_constant_transform>();
    pass.emplace<dequantize_transbin_motion_transform>();
    // pass.emplace<dequantize_transpose_motion_transform>();
    pass.emplace<dequantize_bitcast_motion_transform>();
    pass.emplace<dequantize_reshape_motion_transform>();
    // pass.emplace<dequantize_slice_motion_transform>();
    // pass.emplace<dequantize_pad_motion_transform>();
    pass.emplace<quantize_pad_motion_transform>();
    //    pass.emplace<quantize_transbin_motion_transform>();
    pass.emplace<quantize_transpose_motion_transform>();
    pass.emplace<quantize_bitcast_motion_transform>();
    pass.emplace<quantize_reshape_motion_transform>();
    pass.emplace<quantize_slice_motion_transform>();

    pass.emplace<fold_nop_pad_transform>();
    pass.emplace<fold_nop_bitcast_transform>();
    pass.emplace<fold_slice_slice_transform>();
    pass.emplace<fold_pad_pad_transform>();
    pass.emplace<fold_pad_strided_slice_transform>();

    pass.emplace<fold_layernorm_pattern1_transform>();
    pass.emplace<fold_layernorm_pattern2_transform>();
    pass.emplace<fold_layernorm_pattern3_transform>();
    pass.emplace<convert_layernorm_to_channel_last>();

    pass.emplace<fold_bitcast_transform>();

    pass.emplace<fold_convert_transform>();
    pass.emplace<fold_nop_convert_transform>();

    pass.emplace<remove_nonsense_binary>();
    pass.emplace<strided_slice_to_pad_transform>();

    pass.emplace<transpose_binary_motion_transform>();
    pass.emplace<transpose_constant_binary_motion_transform>();
    pass.emplace<transpose_concat_motion_transform>();
    pass.emplace<transpose_pad_motion_transform>();
    pass.emplace<transpose_clamp_motion_transform>();
    pass.emplace<transpose_sigmoid_motion_transform>();
    pass.emplace<bitcast_clamp_motion_transform>();

    pass.emplace<fuse_clamp_conv2d_transform>();
    pass.emplace<fuse_clamp_conv2d_transpose_transform>();
    pass.emplace<fuse_clamp_binary_transform>();

    pass.emplace<binary_reduce_window2d_motion_up_transform>();

    pass.emplace<fold_conv2d_biasadd_transform>();
    pass.emplace<fold_conv2d_mul_transform>();
    pass.emplace<transpose_reduce_motion_transform>();
    pass.emplace<transpose_unary_motion_transform>();
    pass.emplace<simplify_reduce_transform>();
    pass.emplace<global_reduce_window_to_reduce_transform>();
    pass.emplace<transpose_to_reshape_transform>();
    pass.emplace<take_to_slice_transform>();
    pass.emplace<split_to_slice_transform>();
    pass.emplace<fold_transpose_transform>();
    pass.emplace<fold_nop_transpose_transform>();
    pass.emplace<fold_nop_slice_transform>();
    pass.emplace<fold_matmul_add_transform>();
}

void neutral_target::fold_pad_conv_transform(ir::transforms::transform_pass &pass, [[maybe_unused]] bool add_constant_folding)
{
    using namespace nncase::ir;
    using namespace nncase::ir::transforms;
    if (add_constant_folding)
        pass.emplace<fold_constant_transform>();
    // pass.emplace<dequantize_pad_motion_transform>();
    pass.emplace<transpose_pad_motion_transform>();
    pass.emplace<fold_transpose_transform>();
    pass.emplace<fold_nop_transpose_transform>();
    pass.emplace<space_to_batch_to_pad>();
    pass.emplace<fold_pad_pad_transform>();
    pass.emplace<fuse_pad_conv2d_transform>();
    pass.emplace<fold_nop_pad_transform>();
    // pass.emplace<pad_to_maxpool_transform>();
}

void neutral_target::fold_dilated_conv_transform(ir::transforms::transform_pass &pass, [[maybe_unused]] bool add_constant_folding)
{
    using namespace nncase::ir;
    using namespace nncase::ir::transforms;
    if (add_constant_folding)
        pass.emplace<fold_constant_transform>();
    pass.emplace<transpose_binary_motion_transform>();
    pass.emplace<quantize_transpose_motion_transform>();
    // pass.emplace<dequantize_transpose_motion_transform>();
    pass.emplace<fold_transpose_transform>();
    pass.emplace<fold_nop_transpose_transform>();
    pass.emplace<dequantize_s2b_motion_transform>();
    pass.emplace<quantize_b2s_motion_transform>();
    pass.emplace<fold_dilated_conv2d>();
}

void neutral_target::register_target_independent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr)
{
    using namespace nncase::ir;
    using namespace nncase::ir::transforms;

    {
        transform_pass p("fold_instancenorm");
        p.emplace<fold_instancenorm_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    // fix tflite_detection_postprocess shape error in tflite
    {
        transform_pass p("fix_shape_tdp");
        p.emplace<tflite_detection_postprocess_transform>();
        pass_mgr.add_pass(std::move(p));
    }

    // fold quant node in source model
    {
        transform_pass p("fold_quantize_in_source_model");
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    // split to slice
    {
        transform_pass p("split_to_slice");
        p.emplace<split_to_slice_transform>();
        pass_mgr.add_pass(std::move(p));
    }

    if (type == runtime::stackvm::stackvm_module_type)
    {
        // fold_pad_conv
        {
            transform_pass p("fold_pad_conv");
            fold_pad_conv_transform(p, true);
            pass_mgr.add_pass(std::move(p));
        }
        // fold_dilated_conv
        {
            transform_pass p("fold_dilated_conv");
            p.emplace<fix_dilated_conv_transform>();
            fold_dilated_conv_transform(p, true);
            pass_mgr.add_pass(std::move(p));
        }

        // target_independent_pass
        {
            transform_pass p("target_independent_pass");
            add_default_transforms(p, true);
            pass_mgr.add_pass(std::move(p));
        }

        // pad to slice
        {
            transform_pass p("pad_to_slice");
            p.emplace<pad_to_slice_transform>();
            pass_mgr.add_pass(std::move(p));
        }
    }
}

void neutral_target::register_target_dependent_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr, [[maybe_unused]] bool use_ptq, [[maybe_unused]] bool split_w_to_act)
{
}

void neutral_target::register_quantize_annotation_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr)
{
    {
        transform_pass p("annotate_neutral_quantize");
        p.emplace<add_quant_checkpoints_transform>(std::in_place, ir::op_fused_unary, ir::op_bitcast, ir::op_dequantize, ir::op_binary, ir::op_output_node);
        pass_mgr.add_pass(std::move(p));
    }
}

void neutral_target::register_quantize_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr, [[maybe_unused]] datatype_t quant_type, [[maybe_unused]] std::string_view w_quant_type, [[maybe_unused]] bool use_mse_quant_w, [[maybe_unused]] datatype_t output_type, [[maybe_unused]] quant_param_t &output_quant_param, [[maybe_unused]] std::vector<float> output_range)
{
    {
        transform_pass p("fused_unary_to_lut");
        p.emplace<fused_unary_to_lookup1d_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        transform_pass p("fold_quantize");
        add_default_transforms(p);
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        transform_pass p("change_output_type");
        p.emplace<add_output_quantize_transform>(output_type, output_quant_param, output_range);
        pass_mgr.add_pass(std::move(p));
    }
}

void neutral_target::register_allocation_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

std::unique_ptr<target_options> neutral_target::on_create_options()
{
    return std::make_unique<target_options>();
}

void neutral_target::add_quantization_broadcast([[maybe_unused]] std::unordered_set<ir::node_opcode> &opcodes)
{
    using namespace ir;
    opcodes.emplace(op_input_node);
    opcodes.emplace(op_output_node);
    opcodes.emplace(op_transpose);
    opcodes.emplace(op_pad);
    opcodes.emplace(op_resize_image);
    opcodes.emplace(op_bitcast);
    opcodes.emplace(op_slice);
    opcodes.emplace(op_reduce_window2d);
}
