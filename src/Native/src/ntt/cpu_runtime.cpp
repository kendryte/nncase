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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nncase/ntt/cpu_runtime.h>

extern "C" {
nncase_runtime_cpu_mt_t *g_cpu_mt;

// compiler support
#if defined(_MSC_VER)
#define __ISA_AVAILABLE_X86 0
#define __ISA_AVAILABLE_SSE2 1
#define __ISA_AVAILABLE_SSE42 2
#define __ISA_AVAILABLE_AVX 3
#define __ISA_AVAILABLE_ENFSTRG 4
#define __ISA_AVAILABLE_AVX2 5
#define __ISA_AVAILABLE_AVX512 6

int _fltused = 0;

unsigned int __isa_available = __ISA_AVAILABLE_AVX;
unsigned int __favor = 0;

void __chkstk() {}

#pragma function(acosf)
#pragma function(asinf)
#pragma function(cosf)
#pragma function(coshf)
#pragma function(expf)
#pragma function(logf)
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
float expf(float v) { return g_cpu_mt->expf(v); }
float logf(float v) { return g_cpu_mt->logf(v); }
float nearbyintf(float v) { return g_cpu_mt->nearbyintf(v); }
float sinf(float v) { return g_cpu_mt->sinf(v); }
float sinhf(float v) { return g_cpu_mt->sinhf(v); }
float tanhf(float v) { return g_cpu_mt->tanhf(v); }

#if !defined(WIN32)
void *memcpy(void *dst, const void *src, size_t len) {
    return g_cpu_mt->memcpy(dst, src, len);
}
#endif
}