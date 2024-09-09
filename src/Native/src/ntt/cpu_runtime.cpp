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
#include <nncase/ntt/cpu_runtime.h>

extern "C" {
nncase_runtime_cpu_mt_t *g_cpu_mt;
size_t bid;
size_t tid;

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
#pragma function(sinf)
#pragma function(sinhf)
#pragma function(tanhf)
#endif

float acosf(float v) { return g_cpu_mt->acosf(v); }
float acoshf(float v) { return g_cpu_mt->acoshf(v); }
float asinf(float v) { return g_cpu_mt->asinf(v); }
float asinhf(float v) { return g_cpu_mt->asinhf(v); }
float copysignf(float mag, float sgn) { return g_cpu_mt->copysignf(mag, sgn); }
float cosf(float v) { return g_cpu_mt->cosf(v); }
float coshf(float v) { return g_cpu_mt->coshf(v); }
float erff(float v) { return g_cpu_mt->erff(v); }
float expf(float v) { return g_cpu_mt->expf(v); }
float fmodf(float x, float y) { return g_cpu_mt->fmodf(x, y); }
float logf(float v) { return g_cpu_mt->logf(v); }
float nearbyintf(float v) { return g_cpu_mt->nearbyintf(v); }
float powf(float x, float y) { return g_cpu_mt->powf(x, y); }
float sinf(float v) { return g_cpu_mt->sinf(v); }
float sinhf(float v) { return g_cpu_mt->sinhf(v); }
float tanhf(float v) { return g_cpu_mt->tanhf(v); }

#ifdef WIN32
void _invalid_parameter(wchar_t const *const expression,
                        wchar_t const *const function_name,
                        wchar_t const *const file_name,
                        unsigned int const line_number,
                        uintptr_t const reserved) {
    g_cpu_mt->failfast("invalid_parameter", (va_list)0);
}

int _CrtDbgReport(int reportType, const char *filename, int linenumber,
                  const char *moduleName, const char *format, ...) {
    va_list args;
    va_start(args, format);
    g_cpu_mt->failfast(format, args);
    va_end(args);
    return 0;
}
#else
void *memcpy(void *dst, const void *src, size_t len) {
    return g_cpu_mt->memcpy(dst, src, len);
}

void *memmove(void *dst, const void *src, size_t len) {
    return g_cpu_mt->memmove(dst, src, len);
}
#endif
#endif
}
