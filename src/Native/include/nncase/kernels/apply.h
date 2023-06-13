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
#include <malloc.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

#define BEGIN_NS_NNCASE_KERNELS_STACKVM                                        \
    namespace nncase {                                                         \
    namespace kernels {                                                        \
    namespace stackvm {
namespace reference {

#define END_NS_NNCASE_KERNELS_STACKVM                                          \
    }                                                                          \
    }                                                                          \
    }
} // namespace reference

BEGIN_NS_NNCASE_KERNELS_STACKVM

namespace detail {
template <class Callable>
result<void> apply_1(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    for (size_t i = 0; i < shape[0]; i++) {
        try_(callable(gsl::span(&i, 1)));
    }
    return ok();
}

template <class Callable>
result<void> apply_generic(gsl::span<const size_t> shape,
                           Callable &&callable) noexcept {
    auto index_buffer = (size_t *)
#ifdef _WIN32
        _alloca
#else
        alloca
#endif
        (sizeof(size_t) * shape.size());

    gsl::span<size_t> index(index_buffer, shape.size());
    std::fill(index.begin(), index.end(), 0);
    auto last_dim_idx = (int32_t)shape.size() - 1;
    while (true) {
        int dim = last_dim_idx;
        while (index[dim] == shape[dim]) {
            if (dim == 0) {
                return ok();
            }

            index[dim] = 0;
            index[--dim]++;
        }

        try_(callable(index));
        index[last_dim_idx]++;
    }
    return ok();
}
} // namespace detail

template <class Callable>
result<void> apply(gsl::span<const size_t> shape,
                   Callable &&callable) noexcept {
    switch (shape.size()) {
    case 0:
        return callable(shape);
    case 1:
        return detail::apply_1(shape, std::forward<Callable>(callable));
    default:
        break;
    }

    return detail::apply_generic(shape, std::forward<Callable>(callable));
}

END_NS_NNCASE_KERNELS_STACKVM
