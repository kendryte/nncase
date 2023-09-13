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
#include <cstdint>

#if defined(_MSC_VER)
#ifdef NNCASE_MODULES_CPU_DLL
#define NNCASE_MODULES_CPU_API __declspec(dllexport)
#elif NNCASE_SHARED_LIBS
#define NNCASE_MODULES_CPU_API __declspec(dllimport)
#else
#define NNCASE_MODULES_CPU_API
#endif
#else
#define NNCASE_MODULES_CPU_API __attribute__((visibility("default")))
#endif

#define BEGIN_NS_NNCASE_RT_MODULE(MODULE)                                      \
    namespace nncase {                                                         \
    namespace runtime {                                                        \
    namespace MODULE {

#define END_NS_NNCASE_RT_MODULE                                                \
    }                                                                          \
    }                                                                          \
    }
