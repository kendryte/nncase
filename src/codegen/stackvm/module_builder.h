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
#pragma once
#include <nncase/codegen/stackvm/module_builder.h>
#include <nncase/codegen/stackvm/op_writer.h>
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/broadcast.h>
#include <nncase/ir/ops/call.h>
#include <nncase/ir/ops/compare.h>
#include <nncase/ir/ops/compress.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/copy.h>
#include <nncase/ir/ops/cumsum.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/gather.h>
#include <nncase/ir/ops/gather_elements.h>
#include <nncase/ir/ops/gather_nd.h>
#include <nncase/ir/ops/gru.h>
#include <nncase/ir/ops/hardmax.h>
#include <nncase/ir/ops/instancenorm.h>
#include <nncase/ir/ops/layernorm.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/onehot.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/random_normal.h>
#include <nncase/ir/ops/random_uniform.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/reduce_arg.h>
#include <nncase/ir/ops/reduce_prod.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/roi_align.h>
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/softmax.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/table_lookup.h>
#include <nncase/ir/ops/ternary.h>
#include <nncase/ir/ops/tflite_detection_postprocess.h>
#include <nncase/ir/ops/topk.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/trilu.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/placeholders.h>
#include <nncase/schedule/scheduler.h>

namespace nncase::codegen::stackvm
{
class stackvm_op_builder : public op_builder
{
public:
    using op_builder::op_builder;

    void stshape(uint8_t rshape, const ir::shape_t &shape);
    void staxis(uint8_t rshape, const ir::axis_t &axis);
    void stpaddings(uint8_t rpaddings, std::span<padding const> paddings);
    void lea_buffer(const schedule::buffer_allocation &alloc);
    void ldpadding(const padding &pad);
    void ldscalar(const scalar &value);
};

class stackvm_module_builder : public module_builder
{
public:
    stackvm_module_builder(std::string_view module_name, const module_builder_params &params);

    module_type_t module_type() const noexcept override;
    uint32_t module_version() const noexcept override;

protected:
    section_writer &text_writer();

    void begin_emit_function(const schedule::function_schedule_result &function) override;
    void end_emit_function(const schedule::function_schedule_result &function) override;
    void emit(ir::node &node) override;

private:
#define DEFINE_OP(op_) void emit(ir::op_ &op, stackvm_op_builder &builder);
#include "ops.def"
#undef DEFINE_OP
};
}
