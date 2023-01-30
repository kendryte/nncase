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
#include <nncase/runtime/k210/compiler_defs.h>
#include <nncase/schedule/buffer_allocator.h>

namespace nncase::schedule::k210 {
class NNCASE_MODULES_K210_API kpu_buffer_allocator
    : public first_fit_allocator {
  public:
    kpu_buffer_allocator();

    size_t get_size_in_bytes(const logical_buffer &buffer) override;

  protected:
    size_t alignment() const noexcept override;
};
} // namespace nncase::schedule::k210
