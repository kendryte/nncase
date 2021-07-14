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
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

#define BEGIN_NS_NNCASE_KERNELS_CPU_OPT \
    namespace nncase                    \
    {                                   \
    namespace kernels                   \
    {                                   \
        namespace cpu                   \
        {                               \
            namespace optimized         \
            {

#define END_NS_NNCASE_KERNELS_CPU_OPT \
    }                                 \
    }                                 \
    }                                 \
    }

enum copy_impl_select
{
    all_contiguous,
    src_contiguous,
    dest_contiguous
};
