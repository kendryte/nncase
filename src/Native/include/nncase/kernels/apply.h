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
#ifdef _WIN32
#include <malloc.h> // alloca
#else
#include <alloca.h> // alloca
#endif

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
#define APPLY_IMPL_FOR(i) for (index[i] = 0; index[i] < shape[i]; index[i]++)

template <class Callable>
result<void> apply_1(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    size_t index[1];
    APPLY_IMPL_FOR(0)
    try_(callable(gsl::span(index)));
    return ok();
}

template <class Callable>
result<void> apply_2(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    size_t index[2];
    APPLY_IMPL_FOR(0)
    APPLY_IMPL_FOR(1)
    try_(callable(gsl::span(index)));
    return ok();
}

template <class Callable>
result<void> apply_3(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    size_t index[3];
    APPLY_IMPL_FOR(0)
    APPLY_IMPL_FOR(1)
    APPLY_IMPL_FOR(2)
    try_(callable(gsl::span(index)));
    return ok();
}

template <class Callable>
result<void> apply_4(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    size_t index[4];
    APPLY_IMPL_FOR(0)
    APPLY_IMPL_FOR(1)
    APPLY_IMPL_FOR(2)
    APPLY_IMPL_FOR(3)
    try_(callable(gsl::span(index)));
    return ok();
}

template <class Callable>
result<void> apply_5(gsl::span<const size_t> shape,
                     Callable &&callable) noexcept {
    size_t index[5];
    APPLY_IMPL_FOR(0)
    APPLY_IMPL_FOR(1)
    APPLY_IMPL_FOR(2)
    APPLY_IMPL_FOR(3)
    APPLY_IMPL_FOR(4)
    try_(callable(gsl::span(index)));
    return ok();
}

template <class Callable>
result<void> apply_generic(gsl::span<const size_t> shape,
                           Callable &&callable) noexcept {
    auto index_buffer = (size_t *)
#ifdef _WIN32
        _alloca
#else
        __builtin_alloca
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
    case 2:
        return detail::apply_2(shape, std::forward<Callable>(callable));
    case 3:
        return detail::apply_3(shape, std::forward<Callable>(callable));
    case 4:
        return detail::apply_4(shape, std::forward<Callable>(callable));
    case 5:
        return detail::apply_5(shape, std::forward<Callable>(callable));
    default:
        break;
    }

    return detail::apply_generic(shape, std::forward<Callable>(callable));
}

END_NS_NNCASE_KERNELS_STACKVM
