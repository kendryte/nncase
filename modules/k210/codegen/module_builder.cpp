/* Copyright 2020 Canaan Inc.
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
//#include <nncase/runtime/stackvm/opcode.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::k210;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

std::unique_ptr<module_builder> codegen::create_k210_module_builder(std::string_view module_name, const schedule::module_schedule_result &sched)
{
    return std::make_unique<k210_module_builder>(module_name, sched);
}

k210_module_builder::k210_module_builder(std::string_view module_name, const schedule::module_schedule_result &sched)
    : module_builder(256, module_name, sched)
{
}

module_type_t k210_module_builder::module_type() const noexcept
{
    return k210_module_type;
}

section_writer &k210_module_builder::text_writer()
{
    return writer(".text");
}

void k210_module_builder::emit(ir::node &node)
{
    //    stackvm_op_builder builder(node, text_writer());
    //#define DEFINE_OP(op)                          \
//    if (node.runtime_opcode() == op::opcode()) \
//        return emit(static_cast<op &>(node), builder);
    //#include "ops.def"
    //#undef DEFINE_OP
    module_builder::emit(node);
}

//void stackvm_op_builder::stshape(uint8_t rshape, const ir::shape_t &shape)
//{
//    assert(shape.size() <= std::numeric_limits<uint8_t>::max());
//    for (auto dim : shape)
//        ldc_i4_((int32_t)dim);
//    stshape_(rshape, (uint8_t)shape.size());
//}
//
//void stackvm_op_builder::staxis(uint8_t rshape, const ir::axis_t &axis)
//{
//    assert(axis.size() <= std::numeric_limits<uint8_t>::max());
//    for (auto dim : axis)
//        ldc_i4_(dim);
//    stshape_(rshape, (uint8_t)axis.size());
//}
//
//void stackvm_op_builder::stpaddings(uint8_t rpaddings, std::span<padding const> paddings)
//{
//    assert(paddings.size() <= std::numeric_limits<uint8_t>::max());
//    for (auto &pad : paddings)
//        ldpadding(pad);
//    stpaddings_(rpaddings, (uint8_t)paddings.size());
//}
//
//void stackvm_op_builder::lea_buffer(const schedule::buffer_allocation &alloc)
//{
//    lea_buffer_(alloc.memory_location, 0, (uint32_t)alloc.start);
//}
//
//void stackvm_op_builder::ldpadding(const padding &pad)
//{
//    ldc_i4_((int32_t)pad.before);
//    ldc_i4_((int32_t)pad.after);
//}
