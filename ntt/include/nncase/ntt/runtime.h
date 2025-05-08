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
#ifdef NNCASE_CPU_MODULE
#include "arch/cpu/topology.h"
#endif
#include "shape.h"
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define NTT_RUNTIME_API __declspec(dllexport)
#else
#define NTT_RUNTIME_API __attribute__((visibility("default")))
#endif

namespace nncase::ntt::runtime {
struct thread_inout_desc {
    std::byte *data;
    size_t size;
    size_t *shape;
    size_t *strides;
};

void *thread_alloc(size_t bytes, size_t alignment);
void thread_free(void *ptr);
} // namespace nncase::ntt::runtime

extern "C" void
thread_main(const nncase::ntt::runtime::thread_inout_desc *input_descs,
            nncase::ntt::runtime::thread_inout_desc *const output_descs,
            const std::byte *rdata, const std::byte *local_rdata,
            std::byte *output,
            nncase::ntt::ranked_shape<
                (size_t)nncase::ntt::distributed::topology::count__>
                program_ids);
