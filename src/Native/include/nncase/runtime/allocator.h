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
#include "buffer.h"
#include <memory>
#include <nncase/compiler_defs.h>

BEGIN_NS_NNCASE_RUNTIME

struct buffer_allocate_options {
    size_t flags;
};

inline constexpr size_t HOST_BUFFER_ALLOCATE_CPU_ONLY = 1;
inline constexpr size_t HOST_BUFFER_ALLOCATE_SHARED = 2;

struct buffer_attach_options {
    size_t flags;
    uintptr_t physical_address;
    std::function<void(gsl::byte *)> deleter;
};

inline constexpr size_t HOST_BUFFER_ATTACH_SHARED = 1;

class NNCASE_API buffer_allocator {
  public:
    virtual result<buffer_t>
    allocate(size_t bytes, const buffer_allocate_options &options) = 0;

    virtual result<buffer_t> attach(gsl::span<gsl::byte> data,
                                    const buffer_attach_options &options) = 0;

    static buffer_allocator &host();
    virtual void shrink_memory_pool() = 0;
};

END_NS_NNCASE_RUNTIME
