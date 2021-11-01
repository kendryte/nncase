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
#include "module_builder.h"
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/runtime/stackvm/runtime_module.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::stackvm;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

std::unique_ptr<module_builder> codegen::create_stackvm_module_builder(std::string_view module_name, const module_builder_params &params)
{
    return std::make_unique<stackvm_module_builder>(module_name, params);
}

stackvm_module_builder::stackvm_module_builder(std::string_view module_name, const module_builder_params &params)
    : module_builder(8, module_name, params)
{
}

module_type_t stackvm_module_builder::module_type() const noexcept
{
    return stackvm_module_type;
}

uint32_t stackvm_module_builder::module_version() const noexcept
{
    return stackvm_module_version;
}

section_writer &stackvm_module_builder::text_writer()
{
    return writer(".text");
}

void stackvm_module_builder::begin_emit_function([[maybe_unused]] const schedule::function_schedule_result &function)
{
    set_current_entry_point(text_writer().position());
}

void stackvm_module_builder::end_emit_function([[maybe_unused]] const schedule::function_schedule_result &function)
{
    set_current_function_text_end(text_writer().position());
}

void stackvm_module_builder::emit(ir::node &node)
{
    stackvm_op_builder builder(node, text_writer());
#define DEFINE_OP(op)                          \
    if (node.runtime_opcode() == op::opcode()) \
        return emit(static_cast<op &>(node), builder);
#include "ops.def"
#undef DEFINE_OP
    module_builder::emit(node);
}

void stackvm_op_builder::stshape(uint8_t rshape, const ir::shape_t &shape)
{
    assert(shape.size() <= std::numeric_limits<uint8_t>::max());
    for (auto dim : shape)
        ldc_i4_((int32_t)dim);
    stshape_(rshape, (uint8_t)shape.size());
}

void stackvm_op_builder::staxis(uint8_t rshape, const ir::axis_t &axis)
{
    assert(axis.size() <= std::numeric_limits<uint8_t>::max());
    for (auto dim : axis)
        ldc_i4_(dim);
    stshape_(rshape, (uint8_t)axis.size());
}

void stackvm_op_builder::stpaddings(uint8_t rpaddings, std::span<padding const> paddings)
{
    assert(paddings.size() <= std::numeric_limits<uint8_t>::max());
    for (auto &pad : paddings)
        ldpadding(pad);
    stpaddings_(rpaddings, (uint8_t)paddings.size());
}

void stackvm_op_builder::lea_buffer(const schedule::buffer_allocation &alloc)
{
    lea_buffer_(alloc.memory_location, 0, (uint32_t)alloc.start);
}

void stackvm_op_builder::ldpadding(const padding &pad)
{
    ldc_i4_((int32_t)pad.before);
    ldc_i4_((int32_t)pad.after);
    ldc_i4_((int32_t)pad.interior);
}

void stackvm_op_builder::ldscalar(const scalar &value)
{
    switch (value.type)
    {
    case dt_int8:
        ldc_i4_(value.as<int8_t>());
        break;
    case dt_int16:
        ldc_i4_(value.as<int16_t>());
        break;
    case dt_int32:
        ldc_i4_(value.as<int32_t>());
        break;
    case dt_uint8:
        ldc_i4_((int32_t)value.as<uint8_t>());
        break;
    case dt_uint16:
        ldc_i4_((int32_t)value.as<uint16_t>());
        break;
    case dt_uint32:
        ldc_i4_((int32_t)value.as<uint32_t>());
        break;
    case dt_bfloat16:
        ldc_r4_((float)value.as<bfloat16>());
        break;
    case dt_float32:
        ldc_r4_((float)value.as<float>());
        break;
    default:
        throw std::invalid_argument("Invalid scalar type");
    }
}
