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
#include "runtime_tensor_impl.h"

BEGIN_NS_NNCASE_RUNTIME

namespace detail {
struct host_memory_block;

struct NNCASE_API physical_memory_block {
    uintptr_t physical_address;
    bool owned;

    physical_memory_block() noexcept;
    ~physical_memory_block();
    physical_memory_block(const physical_memory_block &) = delete;
    physical_memory_block(physical_memory_block &&other) noexcept;
    physical_memory_block &operator=(const physical_memory_block &) = delete;
    physical_memory_block &operator=(physical_memory_block &&other) noexcept;

    void free(NNCASE_UNUSED host_memory_block &block) noexcept;

    static result<void> acknowledge(host_memory_block &block) noexcept;
    static result<void> allocate(host_memory_block &block) noexcept;
    static result<void> sync(host_memory_block &block,
                             host_runtime_tensor::sync_op_t op) noexcept;
};
} // namespace detail

END_NS_NNCASE_RUNTIME
