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
#include "cpu_target.h"
#include <nncase/plugin_loader.h>
#include <nncase/transforms/neutral/add_quant_checkpoints.h>
#include <nncase/transforms/neutral/fold_constant.h>
#include <nncase/transforms/neutral/fuse_unary.h>
#include <nncase/transforms/neutral/fused_unary_to_lookup1d.h>
#include <nncase/transforms/neutral/lstm_transform.h>
#include <nncase/transforms/pass.h>

#if defined(_MSC_VER)
#define CPU_TARGET_API __declspec(dllexport)
#else
#define CPU_TARGET_API __attribute__((visibility("default")))
#endif

using namespace nncase;
using namespace nncase::targets;
using namespace nncase::runtime;
using namespace nncase::ir::transforms;

extern "C"
{
    CPU_TARGET_API target *create_target()
    {
        return new cpu_target();
    }
}

void cpu_target::register_target_dependent_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr, [[maybe_unused]] bool use_ptq, [[maybe_unused]] bool split_w_to_act)
{
    // lstm_transform
    {
        transform_pass p("lstm_transform");
        p.emplace<fold_constant_transform>();
        p.emplace<lstm_transform>();
        pass_mgr.add_pass(std::move(p));
    }
}

void cpu_target::register_quantize_annotation_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr)
{
    {
        transform_pass p("fuse_unary");
        p.emplace<fuse_one_unary_transform>();
        p.emplace<fuse_one_binary_transform>();
        p.emplace<fuse_two_fused_unary_transform>();
        p.emplace<fuse_one_fused_unary_with_binary_transform>();
        p.emplace<fuse_two_fused_unary_with_binary_transform>();
        pass_mgr.add_pass(std::move(p));
    }

    {
        transform_pass p("annotate_neutral_quantize");
        p.emplace<add_quant_checkpoints_transform>(std::in_place, ir::op_fused_unary, ir::op_bitcast, ir::op_dequantize, ir::op_binary, ir::op_output_node);
        pass_mgr.add_pass(std::move(p));
    }
}