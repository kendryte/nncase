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
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

namespace {
#define SRAM_SIZE_PER_BLOCK (1024 * 1024 * 4)
#define SRAM_SIZE_PER_THREAD (SRAM_SIZE_PER_BLOCK)

static uint8_t _sram[1][SRAM_SIZE_PER_BLOCK];
static uint8_t *_block_sram_ptr[] = {_sram[0]};
static uint8_t *sram_address(int bid, int tid) {
    return _block_sram_ptr[bid] + (SRAM_SIZE_PER_BLOCK * tid);
}

static void failfast(const char *foramt, va_list args) {
    char buffer[1024];
    vsprintf(buffer, foramt, args);
    throw std::runtime_error(buffer);
}

nncase_runtime_cpu_mt_t nncase_cpu_mt_ = {
    .acosf = acosf,
    .acoshf = acoshf,
    .asinf = asinf,
    .asinhf = asinhf,
    .copysignf = copysignf,
    .cosf = cosf,
    .coshf = coshf,
    .expf = expf,
    .fmodf = fmodf,
    .logf = logf,
    .nearbyintf = nearbyintf,
    .powf = powf,
    .sinf = sinf,
    .sinhf = sinhf,
    .tanhf = tanhf,
    .sram_address = sram_address,
    .failfast = failfast,

#ifndef WIN32
    .memcpy = memcpy,
#endif
};
} // namespace

result<void> cpu_runtime_function::run(std::span<std::byte *> params) noexcept {
    size_t alignment = data_align_;
    size_t space = data_pool_size_ + alignment;
    auto alloced = new (std::nothrow) std::byte[space];
    if (alloced == nullptr) {
        return err(std::errc::not_enough_memory);
    }
    void *data = alloced;
    std::align(alignment, data_pool_size_, data, space);
    if (data == nullptr) {
        return err(std::errc::not_enough_memory);
    }
    kernel_entry_(&nncase_cpu_mt_, params.data(), module().rdata().data(),
                  reinterpret_cast<std::byte *>(data));
    delete[] alloced;
    return ok();
}
