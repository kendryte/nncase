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
#include "evaluate_stack.h"
#include <stack>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class call_frame {
  public:
    call_frame(uintptr_t ret_addr) noexcept : ret_addr_(ret_addr) {}

    uintptr_t ret_addr() const noexcept { return ret_addr_; }
    stack_entry arg(size_t index) const noexcept;

    stack_entry field(size_t index) const noexcept;
    void field(size_t index, stack_entry value) noexcept;

    result<void> push_back_arg(stack_entry arg) noexcept;

  private:
    uintptr_t ret_addr_;
    std::vector<stack_entry> args_;
    std::vector<stack_entry> fields_;
};

class call_frames {
  public:
    call_frames() = default;

    bool empty() const noexcept;
    result<uintptr_t> pop() noexcept;
    result<call_frame *> push(uintptr_t ret_addr) noexcept;
    result<call_frame *> top() noexcept;

  private:
    std::stack<call_frame> frames_;
};

END_NS_NNCASE_RT_MODULE
