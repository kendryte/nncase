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
#include "../shape.h"

namespace nncase::ntt::detail {
template <class Shape> class shape_storage {
  public:
    shape_storage(Shape shape) : shape_(shape) {}

    constexpr size_t rank() noexcept { return shape_.rank(); }
    constexpr const Shape &shape() const noexcept { return shape_; }

  private:
    Shape shape_;
};

template <size_t... Dims> class shape_storage<fixed_shape<Dims...>> {
  public:
    static constexpr size_t rank() noexcept { return sizeof...(Dims); }
    static constexpr auto shape() noexcept { return fixed_shape<Dims...>{}; }
};

template <class Strides> class strides_storage {
  public:
    strides_storage(Strides strides) : strides_(strides) {}

    constexpr Strides &strides() noexcept { return strides_; }
    constexpr const Strides &strides() const noexcept { return strides_; }

  private:
    Strides strides_;
};

template <size_t... Dims> class strides_storage<fixed_strides<Dims...>> {
  public:
    static constexpr auto strides() noexcept {
        return fixed_strides<Dims...>{};
    }
};

template <class Shape, class Strides>
struct NTT_EMPTY_BASES tensor_size_impl : public shape_storage<Shape>,
                                          public strides_storage<Strides> {
    tensor_size_impl(Shape shape, Strides strides)
        : shape_storage<Shape>(shape), strides_storage<Strides>(strides) {}

    constexpr size_t size() noexcept {
        return linear_size(this->shape(), this->strides());
    }
};

template <size_t... Shapes, size_t... Strides>
class NTT_EMPTY_BASES
    tensor_size_impl<fixed_shape<Shapes...>, fixed_strides<Strides...>>
    : public shape_storage<fixed_shape<Shapes...>>,
      public strides_storage<fixed_strides<Strides...>> {
  public:
    static constexpr size_t size() noexcept {
        return linear_size(fixed_shape<Shapes...>{},
                           fixed_strides<Strides...>{});
    }
};
} // namespace nncase::ntt::detail
