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
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nncase/compiler_defs.h>
#include <span>
#include <string_view>

BEGIN_NS_NNCASE_RUNTIME

class cuda_loader {
  public:
    cuda_loader() noexcept
        :
#if 0 
     ofi_(nullptr),
#endif
          mod_(nullptr),
          sym_(nullptr) {
    }
    ~cuda_loader();

    void load(std::span<const std::byte> fatbin);
    void load_from_file(std::string_view path);
    uintptr_t handle() const noexcept { return (uintptr_t)mod_; }
    cudaKernel_t entry() const noexcept { return sym_; }

  private:
    cudaLibrary_t mod_;
    cudaKernel_t sym_;
};

END_NS_NNCASE_RUNTIME
