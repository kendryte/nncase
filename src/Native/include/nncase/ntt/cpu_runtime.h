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
#include <cstddef>
#include <cstdint>

extern "C" {
struct nncase_runtime_cpu_mt_t {
    float (*acosf)(float v);
    float (*acoshf)(float v);
    float (*asinf)(float v);
    float (*asinhf)(float v);
    float (*copysignf)(float mag, float sgn);
    float (*cosf)(float v);
    float (*coshf)(float v);
    float (*expf)(float v);
    float (*fmodf)(float x, float y);
    float (*logf)(float v);
    float (*nearbyintf)(float v);
    float (*powf)(float x, float y);
    float (*sinf)(float v);
    float (*sinhf)(float v);
    float (*tanhf)(float v);

#if !defined(WIN32)
    void *(*memcpy)(void *dst, const void *src, size_t len);
#endif
};

#ifdef NNCASE_CPU_MODULE
extern nncase_runtime_cpu_mt_t *g_cpu_mt;
extern size_t bid;
extern size_t tid;
#endif
}
