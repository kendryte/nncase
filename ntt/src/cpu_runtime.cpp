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
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/runtime/cpu_runtime.h>

namespace nncase::ntt::runtime {
const nncase_runtime_cpu_mt_t *cpu_mt;
size_t tdim;
size_t bdim;

thread_local size_t tid;
thread_local size_t bid;
} // namespace nncase::ntt::runtime

using namespace nncase::ntt::runtime;

extern "C" {
#ifndef NNCASE_STANDALONE
// compiler support
#if defined(_MSC_VER)
#pragma function(acosf)
#pragma function(asinf)
#pragma function(cosf)
#pragma function(coshf)
#pragma function(erff)
#pragma function(expf)
#pragma function(fmodf)
#pragma function(logf)
#pragma function(powf)
#pragma function(roundf)
#pragma function(sinf)
#pragma function(sinhf)
#pragma function(sqrtf)
#pragma function(tanhf)
#endif

float acosf(float v) { return cpu_mt->acosf(v); }
float acoshf(float v) { return cpu_mt->acoshf(v); }
float asinf(float v) { return cpu_mt->asinf(v); }
float asinhf(float v) { return cpu_mt->asinhf(v); }
float copysignf(float mag, float sgn) { return cpu_mt->copysignf(mag, sgn); }
float cosf(float v) { return cpu_mt->cosf(v); }
float coshf(float v) { return cpu_mt->coshf(v); }
float erff(float v) { return cpu_mt->erff(v); }
float expf(float v) { return cpu_mt->expf(v); }
float fmodf(float x, float y) { return cpu_mt->fmodf(x, y); }
float logf(float v) { return cpu_mt->logf(v); }
float nearbyintf(float v) { return cpu_mt->nearbyintf(v); }
float powf(float x, float y) { return cpu_mt->powf(x, y); }
float roundf(float v) { return cpu_mt->roundf(v); }
float sinf(float v) { return cpu_mt->sinf(v); }
float sinhf(float v) { return cpu_mt->sinhf(v); }
float sqrtf(float v) { return cpu_mt->sqrtf(v); }
float tanhf(float v) { return cpu_mt->tanhf(v); }

#ifdef WIN32
void _invalid_parameter(wchar_t const *const expression,
                        wchar_t const *const function_name,
                        wchar_t const *const file_name,
                        unsigned int const line_number,
                        uintptr_t const reserved) {
    cpu_mt->failfast("invalid_parameter", (va_list)0);
}

int _CrtDbgReport(int reportType, const char *filename, int linenumber,
                  const char *moduleName, const char *format, ...) {
    va_list args;
    va_start(args, format);
    cpu_mt->failfast(format, args);
    va_end(args);
    return 0;
}
#else
void *memcpy(void *dst, const void *src, size_t len) {
    return cpu_mt->memcpy(dst, src, len);
}

void *memmove(void *dst, const void *src, size_t len) {
    return cpu_mt->memmove(dst, src, len);
}

void *memset(void *b, int c, size_t len) { return cpu_mt->memset(b, c, len); }
#endif
#endif

void module_entry(nncase::ntt::runtime::module_main_reason reason,
                  void *params) {
    switch (reason) {
    case nncase::ntt::runtime::module_main_reason::block_main: {
        auto block_params =
            reinterpret_cast<nncase_runtime_cpu_block_params_t *>(params);
        cpu_mt = block_params->cpu_mt;
        tdim = block_params->tdim;
        bdim = block_params->bdim;
        break;
    }
    case nncase::ntt::runtime::module_main_reason::thread_main: {
        auto thread_params =
            reinterpret_cast<nncase_runtime_cpu_thread_params_t *>(params);
        tid = thread_params->tid;
        bid = thread_params->bid;
        thread_main(thread_params->inouts, thread_params->rdata);
        break;
    }
    default:
        break;
    }
}
} // extern "C"
