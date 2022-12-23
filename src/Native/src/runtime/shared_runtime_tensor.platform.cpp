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
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/shared_runtime_tensor.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

physical_memory_block::physical_memory_block() noexcept
    : physical_address(0), owned(false) {}

physical_memory_block::~physical_memory_block() {}

physical_memory_block::physical_memory_block(
    physical_memory_block &&other) noexcept
    : physical_address(other.physical_address), owned(other.owned) {
    other.physical_address = 0;
    other.owned = false;
}

physical_memory_block &
physical_memory_block::operator=(physical_memory_block &&other) noexcept {
    assert(owned == false);
    physical_address = other.physical_address;
    owned = other.owned;
    other.physical_address = 0;
    other.owned = false;
    return *this;
}

void physical_memory_block::free(
    NNCASE_UNUSED host_memory_block &block) noexcept {
    if (owned)
        delete[] reinterpret_cast<gsl::byte *>(physical_address);
    physical_address = 0;
    owned = false;
}

result<void>
physical_memory_block::acknowledge(host_memory_block &block) noexcept {
    if (block.virtual_address) {
        if (!block.physical_block.physical_address)
            block.physical_block.physical_address = block.virtual_address;
    } else {
        block.virtual_address = block.physical_block.physical_address;
    }

    block.physical_block.owned = false;
    return ok();
}

result<void>
physical_memory_block::allocate(host_memory_block &block) noexcept {
    auto buffer = new (std::nothrow) gsl::byte[block.size_bytes];
    CHECK_WITH_ERR(buffer, std::errc::not_enough_memory);
    block.physical_block.physical_address = reinterpret_cast<uintptr_t>(buffer);
    block.physical_block.owned = true;
    block.virtual_address = block.physical_block.physical_address;
    return ok();
}

result<void> physical_memory_block::sync(
    NNCASE_UNUSED host_memory_block &block,
    NNCASE_UNUSED host_runtime_tensor::sync_op_t op) noexcept {
    return ok();
}
