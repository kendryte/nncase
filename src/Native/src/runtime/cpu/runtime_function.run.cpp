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
#include "nncase/ntt/runtime.h"
#include "nncase/ntt/runtime/cpu_runtime.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>
#include <stdexcept>
#include <thread>

#ifdef WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#else
#include <pthread.h>
#endif

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
    .erff = erff,
    .expf = expf,
    .fmodf = fmodf,
    .logf = logf,
    .nearbyintf = nearbyintf,
    .powf = powf,
    .roundf = roundf,
    .sinf = sinf,
    .sinhf = sinhf,
    .sqrtf = sqrtf,
    .tanhf = tanhf,
    .sram_address = sram_address,
    .failfast = failfast,

#ifndef WIN32
    .memcpy = memcpy,
    .memmove = memmove,
    .memset = memset,
#endif
};
} // namespace

result<void> cpu_runtime_function::run(std::span<std::byte *> params) noexcept {
    std::vector<std::thread> threads;
    for (size_t bid = 0; bid < bdim_; bid++) {
        nncase_runtime_cpu_block_params_t block_params{
            .cpu_mt = &nncase_cpu_mt_,
            .tdim = tdim_,
            .bdim = bdim_,
        };
        module_entry_(ntt::runtime::module_main_reason::block_main,
                      &block_params);
        for (size_t tid = 0; tid < tdim_; tid++) {
            threads.emplace_back([this, tid, bid, params] {
                size_t cpu_id = bid * tdim_ + tid;
#if WIN32
                SetThreadAffinityMask(GetCurrentThread(),
                                      (DWORD_PTR)1 << cpu_id);
#elif defined(__APPLE__)
#else
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu_id, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                       &cpuset);
#endif

                nncase_runtime_cpu_thread_params_t thread_params{
                    .tid = tid,
                    .bid = bid,
                    .inouts = params.data(),
                    .rdata = module().rdata().data(),
                };
                module_entry_(ntt::runtime::module_main_reason::thread_main,
                              &thread_params);
            });
        }
    }

    for (auto &t : threads)
        t.join();

    return ok();
}
