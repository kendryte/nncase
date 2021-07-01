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
#include <nncase/runtime/compiler_defs.h>

#if defined(_MSC_VER)
#ifdef NNCASE_MODULES_K210_DLL
#define NNCASE_MODULES_K210_API __declspec(dllexport)
#else
#define NNCASE_MODULES_K210_API __declspec(dllimport)
#endif
#else
#define NNCASE_MODULES_K210_API
#endif

#define BEGIN_NS_NNCASE_RT_K210 \
    namespace nncase            \
    {                           \
    namespace runtime           \
    {                           \
        namespace k210          \
        {
#define END_NS_NNCASE_RT_K210 \
    }                         \
    }                         \
    }

#define BEGIN_NS_NNCASE_KERNELS_K210 \
    namespace nncase                 \
    {                                \
    namespace kernels                \
    {                                \
        namespace k210               \
        {
#define END_NS_NNCASE_KERNELS_K210 \
    }                              \
    }                              \
    }
