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
template <Shape TShape> class shape_storage {
  public:
    constexpr shape_storage(TShape shape) noexcept : shape_(shape) {}

    static constexpr auto rank() noexcept { return TShape::rank(); }
    constexpr const TShape &shape() const noexcept { return shape_; }
    constexpr size_t size() const noexcept { return shape().length(); }

  private:
    TShape shape_;
};

template <FixedShape TShape> class shape_storage<TShape> {
  public:
    constexpr shape_storage() noexcept = default;
    constexpr shape_storage(TShape) noexcept {}

    static constexpr auto rank() noexcept { return TShape::rank(); }
    static constexpr TShape shape() noexcept { return TShape{}; }
    static constexpr size_t size() noexcept { return shape().length(); }
};

template <Strides TStrides> class strides_storage {
  public:
    constexpr strides_storage(TStrides strides) noexcept : strides_(strides) {}

    constexpr const TStrides &strides() const noexcept { return strides_; }

  private:
    TStrides strides_;
};

template <FixedStrides TStrides> class strides_storage<TStrides> {
  public:
    constexpr strides_storage() noexcept = default;
    constexpr strides_storage(TStrides) noexcept {}

    static constexpr TStrides strides() noexcept { return TStrides{}; }
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
};
} // namespace nncase::ntt::detail
