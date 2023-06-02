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
#include <cstdlib>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

namespace {
constexpr size_t INITIAL_STACK_ENTRIES = 64;
}

evaluate_stack::evaluate_stack() noexcept
    : entries_((stack_entry *)std::malloc(sizeof(stack_entry) *
                                          INITIAL_STACK_ENTRIES)),
      top_(entries_),
      end_(entries_ + INITIAL_STACK_ENTRIES) {
    dbg_check(entries_);
    std::uninitialized_default_construct(entries_, end_);
}

evaluate_stack::~evaluate_stack() {
    for (auto it = entries_; it != end_; ++it)
        it->~stack_entry();
    free(entries_);
}

void evaluate_stack::enlarge() noexcept {
    auto new_size = (end_ - entries_) * 3 / 2; // 1.5x
    auto top_offset = top_ - entries_;
#if defined(__GNUC__)
#pragma GCC diagnostic push
#if !defined(__clang__) && __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#endif
    auto new_entries =
        (stack_entry *)std::realloc(entries_, sizeof(stack_entry) * new_size);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (!new_entries)
        fail_fast("Out of memory");

    entries_ = new_entries;
    top_ = entries_ + top_offset;
    end_ = entries_ + new_size;
}

void evaluate_stack::push(stack_entry entry) noexcept {
    if (!full()) {
        new (top_++) stack_entry(std::move(entry));
    } else {
        enlarge();
        new (top_++) stack_entry(std::move(entry));
    }
}
