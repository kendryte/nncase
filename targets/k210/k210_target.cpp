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
#include "k210_target.h"
#include <nncase/codegen/k210/module_builder.h>
#include <nncase/ir/ops/k210/k210_evaluators.h>
#include <nncase/ir/ops/k210/opcode.h>
#include <nncase/runtime/k210/runtime_types.h>
#include <nncase/schedule/k210/kpu_buffer_allocator.h>
#include <nncase/transforms/k210/conv2d_transpose_transform.h>
#include <nncase/transforms/k210/fake_kpu_conv2d.h>
#include <nncase/transforms/k210/fold_kpu_upload.h>
#include <nncase/transforms/k210/fuse_kpu_conv2d_pool.h>
#include <nncase/transforms/k210/fuse_kpu_download.h>
#include <nncase/transforms/k210/fused_unary_motion.h>
#include <nncase/transforms/k210/kpu_conv2d.h>
#include <nncase/transforms/k210/strided_slice_motion.h>
#include <nncase/transforms/neutral/add_quant_checkpoints.h>
#include <nncase/transforms/neutral/add_to_conv2d.h>
#include <nncase/transforms/neutral/eliminate_dilated_conv2d.h>
#include <nncase/transforms/neutral/fold_pad.h>
#include <nncase/transforms/neutral/fold_quantize.h>
#include <nncase/transforms/neutral/fold_transpose.h>
#include <nncase/transforms/neutral/fuse_pad.h>
#include <nncase/transforms/neutral/split_sigmoid.h>
#include <nncase/transforms/neutral/transpose_motion.h>
#include <nncase/transforms/pass.h>

#if defined(_MSC_VER)
#define K210_TARGET_API __declspec(dllexport)
#else
#define K210_TARGET_API __attribute__((visibility("default")))
#endif

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::schedule::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;
using namespace nncase::targets;
using namespace nncase::runtime;

extern "C" {
K210_TARGET_API target *create_target() { return new k210_target(); }
}

std::unique_ptr<codegen::module_builder> k210_target::create_module_builder(
    const module_type_t &type, std::string_view module_name,
    const codegen::module_builder_params &params) {
    if (type == runtime::k210::k210_module_type)
        return codegen::create_k210_module_builder(module_name, params);
    return neutral_target::create_module_builder(type, module_name, params);
}

void k210_target::register_allocators(
    const module_type_t &type, allocator_map_t &allocators,
    std::vector<std::shared_ptr<buffer_allocator>> &allocator_holders) {
    if (type == runtime::k210::k210_module_type) {
        allocators.emplace(
            mem_input,
            allocator_holders
                .emplace_back(std::make_shared<linear_buffer_allocator>())
                .get());
        allocators.emplace(
            mem_output,
            allocator_holders
                .emplace_back(std::make_shared<linear_buffer_allocator>())
                .get());
        allocators.emplace(
            mem_rdata,
            allocator_holders
                .emplace_back(std::make_shared<linear_buffer_allocator>())
                .get());
        allocators.emplace(
            mem_data,
            allocator_holders
                .emplace_back(std::make_shared<linear_buffer_allocator>())
                .get());
        allocators.emplace(
            runtime::k210::mem_kpu,
            allocator_holders
                .emplace_back(std::make_shared<kpu_buffer_allocator>())
                .get());
    } else {
        neutral_target::register_allocators(type, allocators,
                                            allocator_holders);
    }
}

void k210_target::register_evaluator_ops() {
    neutral_target::register_evaluator_ops();
    ir::k210::register_k210_evaluators();
}

void k210_target::register_target_dependent_passes(
    [[maybe_unused]] const module_type_t &type,
    [[maybe_unused]] ir::transforms::pass_manager &pass_mgr,
    [[maybe_unused]] bool use_ptq) {
    {
        transform_pass p("sigmoid_lowering");
        p.emplace<split_sigmoid_transform>();
        pass_mgr.add_pass(std::move(p));
    }

    {
        transform_pass p("strided_slice_lowering");
        p.emplace<strided_slice_conv2d_pool>();
        pass_mgr.add_pass(std::move(p));
    }

    {
        transform_pass p("conv2d_transpose_lowering");
        p.emplace<conv2d_transpose_transform>();
        pass_mgr.add_pass(std::move(p));
    }
}

void k210_target::register_quantize_annotation_passes(
    const module_type_t &type, ir::transforms::pass_manager &pass_mgr) {
    {
        transform_pass p("annotate_kpu1");
        p.emplace<eliminate_dilated_conv2d_transform>();
        p.emplace<fake_kpu_conv2d_transform>();
        p.emplace<strided_slice_motion_transform>();
        p.emplace<fuse_fake_kpu_conv2d_strided_slice_transform>();
        add_default_transforms(p);
        pass_mgr.add_pass(std::move(p));
    }

    neutral_target::register_quantize_annotation_passes(type, pass_mgr);

    {
        transform_pass p("annotate_kpu2");
        p.emplace<add_to_conv2d_transform>();
        p.emplace<eliminate_dilated_conv2d_transform>();
        p.emplace<fake_kpu_conv2d_transform>();
        p.emplace<strided_slice_motion_transform>();
        p.emplace<fuse_fake_kpu_conv2d_strided_slice_transform>();
        add_default_transforms(p);
        pass_mgr.add_pass(std::move(p));
    }

    {
        transform_pass p("fused_unary_motion");
        p.emplace<slice_fused_unary_motion_transform>();
        p.emplace<pad_fused_unary_motion_transform>();
        pass_mgr.add_pass(std::move(p));
    }

    {
        transform_pass p("annotate_kpu_quantize");
        p.emplace<add_quant_checkpoints_transform>(
            std::in_place, ir::op_fused_unary,
            ir::k210::op_k210_fake_kpu_conv2d);
        pass_mgr.add_pass(std::move(p));
    }
}

void k210_target::register_quantize_passes(
    const module_type_t &type, ir::transforms::pass_manager &pass_mgr,
    [[maybe_unused]] datatype_t quant_type,
    [[maybe_unused]] std::string_view w_quant_type,
    [[maybe_unused]] bool use_mse_quant_w) {
    {
        transform_pass p("lowering_kpu_conv2d");
        p.emplace<kpu_conv2d_transform>(use_mse_quant_w);
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        transform_pass p("fuse_kpu_pool");
        p.emplace<fuse_kpu_conv2d_pool_transform>();
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        transform_pass p("fold_kpu_data_exchg");
        p.emplace<fold_kpu_upload_transform>();
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
    {
        neutral_target::register_quantize_passes(type, pass_mgr, quant_type,
                                                 w_quant_type, use_mse_quant_w);

        transform_pass p("fold_kpu_data_exchg2");
        // p.emplace<fuse_kpu_download_transform>();
        // p.emplace<fold_input_kpu_upload_transform>();
        add_default_transforms(p);
        p.emplace<fold_quantize_transform>();
        pass_mgr.add_pass(std::move(p));
    }
}
