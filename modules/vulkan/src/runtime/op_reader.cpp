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
#include <nncase/runtime/vulkan/op_reader.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> op_visitor::next() noexcept {
    auto opcode = static_cast<opcode_t>(reader_.peek_unaligned<uint8_t>());
    switch (opcode) {
    case opcode_t::ldbuf:
        return visit(reader_.read_unaligned<ldbuf_op_t>());
    case opcode_t::ldbufbarrier:
        return visit(reader_.read_unaligned<ldbufbarrier_op_t>());
    case opcode_t::ldbufcopy:
        return visit(reader_.read_unaligned<ldbufcopy_op_t>());
    case opcode_t::copybuf:
        return visit(reader_.read_unaligned<copybuf_op_t>());
    case opcode_t::ldpipeline:
        return visit(reader_.read_unaligned<ldpipeline_op_t>());
    case opcode_t::dispatch:
        return visit(reader_.read_unaligned<dispatch_op_t>());
    case opcode_t::barrier:
        return visit(reader_.read_unaligned<barrier_op_t>());
    default:
        break;
    }

    return err(std::errc::operation_not_supported);
}

result<void> op_visitor::visit(std::span<const std::byte> text) noexcept {
    reader_ = span_reader(text);
    interrupted_ = false;

    while (!interrupted_ && !reader_.empty())
        try_(next());
    return ok();
}
