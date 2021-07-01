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
#include "evaluate_stack.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

evaluate_stack::evaluate_stack() noexcept
    : top_(0)
{
    entries_.resize(64);
}

bool evaluate_stack::empty() const noexcept
{
    return top_ == 0;
}

bool evaluate_stack::full() const noexcept
{
    return top_ == entries_.size();
}

result<stack_entry> evaluate_stack::peek() noexcept
{
    if (!empty())
        return ok(entries_[top_ - 1]);
    return err(nncase_errc::stackvm_stack_underflow);
}

result<stack_entry> evaluate_stack::pop() noexcept
{
    if (!empty())
        return ok(entries_[--top_]);
    return err(nncase_errc::stackvm_stack_underflow);
}

result<void> evaluate_stack::push(stack_entry entry) noexcept
{
    if (full())
        entries_.resize(entries_.size() + 1);

    entries_[top_++] = entry;
    return ok();
}
