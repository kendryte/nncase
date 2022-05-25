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
#include "call_frame.h"
#include <nncase/runtime/dbg.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<stack_entry> call_frame::arg(size_t index) const noexcept {
    CHECK_WITH_ERR(index < args_.size(), std::errc::result_out_of_range);
    return ok(args_[index]);
}

result<void> call_frame::push_back_arg(stack_entry arg) noexcept {
    try {
        args_.emplace_back(std::move(arg));
        return ok();
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }
}

result<stack_entry> call_frame::field(size_t index) const noexcept {
    CHECK_WITH_ERR(index < fields_.size(), std::errc::result_out_of_range);
    return ok(fields_[index]);
}

result<void> call_frame::field(size_t index, stack_entry value) noexcept {
    if (fields_.size() <= index) {
        try {
            fields_.resize(index + 1);
        } catch (...) {
            return err(std::errc::not_enough_memory);
        }
    }

    fields_[index] = value;
    return ok();
}

bool call_frames::empty() const noexcept { return frames_.empty(); }

result<uintptr_t> call_frames::pop() noexcept {
    if (!empty()) {
        auto ret_addr = frames_.top().ret_addr();
        frames_.pop();
        return ok(ret_addr);
    }

    return err(nncase_errc::stackvm_stack_underflow);
}

result<call_frame *> call_frames::top() noexcept {
    if (!empty()) {
        return ok(&frames_.top());
    }

    return err(nncase_errc::stackvm_stack_underflow);
}

result<call_frame *> call_frames::push(uintptr_t ret_addr) noexcept {
    frames_.push({ret_addr});
    return ok(&frames_.top());
}
