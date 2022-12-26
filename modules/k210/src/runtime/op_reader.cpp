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
#include <nncase/runtime/k210/error.h>
#include <nncase/runtime/k210/op_reader.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void> op_visitor::next() noexcept {
    auto opcode = static_cast<opcode_t>(reader_.peek_unaligned<uint8_t>());
    switch (opcode) {
    case opcode_t::kpu_conv2d:
        return visit(reader_.read_unaligned<kpu_conv2d_options>());
    case opcode_t::kpu_download:
        return visit(reader_.read_unaligned<kpu_download_options>());
    case opcode_t::kpu_upload:
        return visit(reader_.read_unaligned<kpu_upload_options>());
    case opcode_t::copy:
        return visit(reader_.read_unaligned<copy_options>());
    default:
        break;
    }

    return err(nncase_k210_errc::k210_illegal_instruction);
}

result<void> op_visitor::visit(gsl::span<const gsl::byte> text) noexcept {
    reader_ = span_reader(text);
    interrupted_ = false;

    while (!interrupted_ && !reader_.empty())
        try_(next());
    return ok();
}
