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
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define NTT_RUNTIME_API __declspec(dllexport)
#else
#define NTT_RUNTIME_API __attribute__((visibility("default")))
#endif

namespace nncase::ntt::runtime {
enum class module_main_reason {
    block_main,
    thread_main,
};
}

extern "C" {
extern void thread_main(std::byte *const *inouts, const std::byte *rdata);

extern NTT_RUNTIME_API void
module_entry(nncase::ntt::runtime::module_main_reason reason, void *params);
using module_entry_t = decltype(module_entry) *;
}
