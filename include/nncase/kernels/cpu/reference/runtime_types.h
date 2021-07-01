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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

#define BEGIN_NS_NNCASE_KERNELS_CPU_REF \
    namespace nncase                    \
    {                                   \
    namespace kernels                   \
    {                                   \
        namespace cpu                   \
        {                               \
            namespace reference         \
            {

#define END_NS_NNCASE_KERNELS_CPU_REF \
    }                                 \
    }                                 \
    }                                 \
    }

BEGIN_NS_NNCASE_KERNELS_CPU_REF

namespace detail
{
template <class Callable>
result<void> apply_impl(Callable &&callable, runtime_shape_t index_prefix, runtime_shape_t::const_iterator index_begin, runtime_shape_t::const_iterator index_end) noexcept
{
    const auto head = *index_begin++;
    index_prefix.push_back(0);
    if (index_begin == index_end)
    {
        for (size_t i = 0; i < head; i++)
        {
            index_prefix.back() = i;
            try_(callable(index_prefix));
        }
    }
    else
    {
        for (size_t i = 0; i < head; i++)
        {
            index_prefix.back() = i;
            try_(apply_impl(std::forward<Callable>(callable), index_prefix, index_begin, index_end));
        }
    }

    return ok();
}
}

template <class Callable>
result<void> apply(const runtime_shape_t &shape, Callable &&callable) noexcept
{
    return detail::apply_impl(std::forward<Callable>(callable), runtime_shape_t(), shape.cbegin(), shape.cend());
}

END_NS_NNCASE_KERNELS_CPU_REF
