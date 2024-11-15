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
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <exception>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/runtime/cpu_runtime.h>
#include <thread>

#ifdef WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#endif

namespace nncase::ntt::runtime {
size_t tdim;
size_t bdim;
size_t cdim;

#ifdef __APPLE__
pthread_key_t cpu_thread_context_key;
#else
thread_local cpu_thread_context_t cpu_thread_context;
#endif

void *thread_alloc(size_t bytes, size_t alignment) {
#ifdef WIN32
    return _aligned_malloc(bytes, alignment);
#else
    size_t mask = alignment - 1;
    size_t aligned_bytes = bytes + (-bytes & mask);
    auto ptr = aligned_alloc(alignment, aligned_bytes);
    if (!ptr) {
        std::terminate();
    }
    return ptr;
#endif
}

void thread_free(void *ptr) {
#ifdef WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
} // namespace nncase::ntt::runtime

using namespace nncase::ntt::runtime;

cpu_thread_context_t &cpu_thread_context_t::current() noexcept {
#ifndef __APPLE__
    return cpu_thread_context;
#else
    return *reinterpret_cast<cpu_thread_context_t *>(
        pthread_getspecific(cpu_thread_context_key));
#endif
}

extern "C" void block_entry(const cpu_block_entry_params_t &params) {
    tdim = params.tdim;
    bdim = params.bdim;
    cdim = params.cdim;

#ifdef __APPLE__
    cpu_thread_context_key = params.cpu_thread_context_key;
#endif

    std::vector<std::thread> threads;
    for (size_t tid = 0; tid < tdim; tid++) {
        threads.emplace_back([tid, params] {
#ifdef __APPLE__
            pthread_setspecific(cpu_thread_context_key,
                                new cpu_thread_context_t
#else
            cpu_thread_context_t::current() =
#endif
                                {
                                    .tid = tid,
                                    .bid = params.bid,
                                    .cid = params.cid,
                                }
#ifdef __APPLE__
            );
#else
            ;
#endif

            size_t cpu_id = params.cpu_id_offset + tid;
#if WIN32
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu_id);
#elif defined(__APPLE__)
            thread_affinity_policy_data_t policy = {(int)cpu_id};
            thread_policy_set(pthread_mach_thread_np(pthread_self()),
                              THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                              THREAD_AFFINITY_POLICY_COUNT);
#else
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
            cpu_thread_context_t::current().tid = tid;
            thread_main(params.inouts, params.rdata);
        });
    }

    for (auto &t : threads)
        t.join();
}
