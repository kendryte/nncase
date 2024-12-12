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
#include "tensor.h"

namespace nncase::ntt {
namespace detail {
template <size_t Axis, class Shape, class Callable> struct apply_impl {
    void operator()(ranked_shape<Shape::rank()> &index, const Shape &shape,
                    Callable &&callable) {
        for (index[Axis] = 0; index[Axis] < shape[Axis]; index[Axis]++) {
            if constexpr (Axis == Shape::rank() - 1) {
                callable(index);
            } else {
                apply_impl<Axis + 1, Shape, Callable>()(
                    index, shape, std::forward<Callable>(callable));
            }
        }
    }
};
} // namespace detail

template <class Shape, class Callable>
void apply(const Shape &shape, Callable &&callable) {
    ranked_shape<Shape::rank()> index;
    if constexpr (Shape::rank()) {
        detail::apply_impl<0, Shape, Callable>()(
            index, shape, std::forward<Callable>(callable));
    } else {
        callable(index);
    }
}
} // namespace nncase::ntt
