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
#include "../compiler_defs.h"
#include "nncase/ntt/shape.h"
#include <cstddef>
#include <type_traits>

namespace nncase::ntt::detail {
template <class Shape> class shape_storage {
  public:
    template <class TDummy = Shape,
              class = std::enable_if_t<FixedShape<TDummy>>>
    constexpr shape_storage() noexcept : shape_{} {}

    shape_storage(Shape shape) : shape_(shape) {}

    static constexpr auto rank() noexcept { return Shape::rank(); }
    constexpr const Shape &shape() const noexcept { return shape_; }

  private:
    Shape shape_;
};

template <class Strides> class strides_storage {
  public:
    template <class TDummy = Strides,
              class = std::enable_if_t<FixedStrides<TDummy>>>
    constexpr strides_storage() noexcept : strides_{} {}

    strides_storage(Strides strides) : strides_(strides) {}

    constexpr Strides &strides() noexcept { return strides_; }
    constexpr const Strides &strides() const noexcept { return strides_; }

  private:
    Strides strides_;
};

template <class Shape, class Strides>
struct NTT_EMPTY_BASES tensor_size_impl : public shape_storage<Shape>,
                                          public strides_storage<Strides> {

    template <
        class TDummy1 = Shape, class TDummy2 = Strides,
        class = std::enable_if_t<FixedShape<TDummy1> && FixedStrides<TDummy2>>>
    constexpr tensor_size_impl() noexcept {}

    tensor_size_impl(Shape shape, Strides strides)
        : shape_storage<Shape>(shape), strides_storage<Strides>(strides) {}

    constexpr size_t size() const noexcept {
        return shape_storage<Shape>::shape().length();
    }
};
} // namespace nncase::ntt::detail
