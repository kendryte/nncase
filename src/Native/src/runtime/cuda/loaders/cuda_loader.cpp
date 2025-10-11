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
#include "cuda_loader.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase::runtime;

#define THROW_CUDA_IF_FAILED(x)                                                \
    if ((x) != cudaSuccess) {                                                  \
        throw std::runtime_error(cudaGetErrorString(x));                       \
    }

cuda_loader::~cuda_loader() {
    if (mod_) {
        cudaLibraryUnload(mod_);
    }
}

void cuda_loader::load(std::span<const std::byte> fatbin) {
    THROW_CUDA_IF_FAILED(cudaLibraryLoadData(&mod_, fatbin.data(), nullptr,
                                             nullptr, 0, nullptr, nullptr, 0));
    THROW_CUDA_IF_FAILED(cudaLibraryGetKernel(&sym_, mod_, "block_entry"));
}
void cuda_loader::load_from_file(std::string_view path) {
    THROW_CUDA_IF_FAILED(cudaLibraryLoadFromFile(
        &mod_, path.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    THROW_CUDA_IF_FAILED(cudaLibraryGetKernel(&sym_, mod_, "block_entry"));
}
