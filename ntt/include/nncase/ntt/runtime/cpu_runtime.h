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
#include "../distributed.h"
#include "../runtime.h"
#include <cstdarg>

extern "C" {
struct nncase_runtime_cpu_mt_t {
    float (*acosf)(float v);
    float (*acoshf)(float v);
    float (*asinf)(float v);
    float (*asinhf)(float v);
    float (*copysignf)(float mag, float sgn);
    float (*cosf)(float v);
    float (*coshf)(float v);
    float (*erff)(float v);
    float (*expf)(float v);
    float (*fmodf)(float x, float y);
    float (*logf)(float v);
    float (*nearbyintf)(float v);
    float (*powf)(float x, float y);
    float (*roundf)(float v);
    float (*sinf)(float v);
    float (*sinhf)(float v);
    float (*sqrtf)(float v);
    float (*tanhf)(float v);

    uint8_t *(*sram_address)(int bid, int tid);

    void (*failfast)(const char *format, va_list args);

#ifndef WIN32
    void *(*memcpy)(void *dst, const void *src, size_t len);
    void *(*memmove)(void *dst, const void *src, size_t len);
    void *(*memset)(void *b, int c, size_t len);
#endif
};

struct nncase_runtime_cpu_block_params_t {
    const nncase_runtime_cpu_mt_t *cpu_mt;
    size_t tdim;
    size_t bdim;
    size_t cdim;
};

struct nncase_runtime_cpu_thread_params_t {
    size_t tid;
    size_t bid;
    size_t cid;
    std::byte *const *inouts;
    const std::byte *rdata;
};
}

namespace nncase::ntt::runtime {
extern const nncase_runtime_cpu_mt_t *cpu_mt;
extern size_t tdim;
extern size_t bdim;
extern size_t cdim;

extern thread_local size_t tid;
extern thread_local size_t bid;
extern thread_local size_t cid;
} // namespace nncase::ntt::runtime

namespace nncase::ntt {
template <> struct program_id_getter<0> {
    static size_t id() noexcept { return runtime::tid; }
    static size_t dim() noexcept { return runtime::tdim; }
};

template <> struct program_id_getter<1> {
    static size_t id() noexcept { return runtime::bid; }
    static size_t dim() noexcept { return runtime::bdim; }
};

template <> struct program_id_getter<2> {
    static size_t id() noexcept { return runtime::cid; }
    static size_t dim() noexcept { return runtime::cdim; }
};
} // namespace nncase::ntt
