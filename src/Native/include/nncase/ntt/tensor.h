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
#include "shape.h"
#include <vector>

namespace nncase::ntt {
template <class T, class Shape, class Strides, bool IsView> class tensor_base;
template <class T, class Shape, class Strides = default_strides_t<Shape>>
using tensor = tensor_base<T, Shape, Strides, false>;
template <class T, class Shape, class Strides>
using tensor_view = tensor_base<T, Shape, Strides, true>;

namespace detail {
template <class T, class Shape, class Strides, bool IsView,
          bool FixedShape = is_fixed_dims_v<Shape> &&is_fixed_dims_v<Strides>>
class tensor_storage;

template <class T, class Shape, class Strides>
class tensor_storage<T, Shape, Strides, false, true> {
  public:
    using buffer_type = std::array<T, linear_size(Shape{}, Strides{})>;

    static constexpr auto shape() noexcept { return Shape{}; }
    static constexpr auto strides() noexcept { return Strides{}; }

    constexpr auto buffer() const noexcept { return std::span(buffer_); }
    constexpr auto buffer() noexcept { return std::span(buffer_); }

  private:
    buffer_type buffer_;
};

template <class T, class Shape, class Strides>
class tensor_storage<T, Shape, Strides, false, false> {
  public:
    using buffer_type = std::vector<T>;

    constexpr tensor_storage(Shape shape, Strides strides)
        : buffer_(linear_size(shape, strides)),
          shape_(std::move(shape)),
          strides_(std::move(strides)) {}

    constexpr const Shape &shape() noexcept { return shape_; }
    constexpr const Strides &strides() noexcept { return strides_; }

    constexpr const std::span<const T> buffer() const noexcept {
        return buffer_;
    }
    constexpr std::span<T> buffer() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
    Shape shape_;
    Strides strides_;
};

template <class T, class Shape, class Strides>
class tensor_storage<T, Shape, Strides, true, true> {
  public:
    using const_buffer_type = std::span<T, linear_size(Shape{}, Strides{})>;
    using buffer_type = std::span<T, linear_size(Shape{}, Strides{})>;

    constexpr tensor_storage(buffer_type buffer, Shape = {},
                             Strides = {}) noexcept
        : buffer_(std::move(buffer)) {}

    static constexpr auto shape() noexcept { return Shape{}; }
    static constexpr auto strides() noexcept { return Strides{}; }

    constexpr const_buffer_type buffer() const noexcept { return buffer_; }
    constexpr buffer_type buffer() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
};

template <class T, class Shape, class Strides>
class tensor_storage<T, Shape, Strides, true, false> {
  public:
    using const_buffer_type = std::span<const T>;
    using buffer_type = std::span<T>;

    constexpr tensor_storage(buffer_type buffer, Shape shape,
                             Strides strides) noexcept
        : buffer_(std::move(buffer)),
          shape_(std::move(shape)),
          strides_(std::move(strides)) {}

    constexpr const Shape &shape() const noexcept { return shape_; }
    constexpr const Strides &strides() const noexcept { return strides_; }

    constexpr const_buffer_type buffer() const noexcept { return buffer_; }
    constexpr buffer_type buffer() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
    Shape shape_;
    Strides strides_;
};
} // namespace detail

template <class T, class Shape, class Strides, bool IsView>
class tensor_base : public detail::tensor_storage<T, Shape, Strides, IsView> {
  public:
    using element_type = T;
    using storage_type = detail::tensor_storage<T, Shape, Strides, IsView>;

    using storage_type::buffer;
    using storage_type::shape;
    using storage_type::storage_type;
    using storage_type::strides;

    template <class Index, class UShape>
    constexpr tensor_view<T, UShape, Strides> view(Index index,
                                                   UShape shape) noexcept {
        if constexpr (is_fixed_dims_v<Strides>) {
            if constexpr (is_fixed_dims_v<Index>) {
                if constexpr (is_fixed_dims_v<UShape>) {
                    return {
                        buffer()
                            .template subspan<linear_offset(index, strides()),
                                              linear_size(shape, strides())>(),
                        shape, strides()};
                } else {
                    return {
                        buffer()
                            .template subspan<linear_offset(index, strides())>(
                                linear_size(shape, strides())),
                        shape, strides()};
                }
            }
        } else {
            return {buffer().subspan(linear_offset(index, strides()),
                                     linear_size(shape, strides())),
                    shape, strides()};
        }
    }

    constexpr tensor_view<T, Shape, Strides> view() noexcept {
        return view(zero_shape_t<Shape::rank()>{}, shape());
    }

    template <class... Indices>
    constexpr const T &operator()(Indices &&...index) const noexcept {
        if constexpr (sizeof...(index) == 1 &&
                      (!std::is_integral_v<Indices> && ...)) {
            return buffer()[linear_offset(index..., strides())];
        } else {
            return this->operator()(
                ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
        }
    }

    template <class... Indices>
    constexpr T &operator()(Indices &&...index) noexcept {
        if constexpr (sizeof...(index) == 1 &&
                      (!std::is_integral_v<Indices> && ...)) {
            return buffer()[linear_offset(index..., strides())];
        } else {
            return this->operator()(
                ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
        }
    }
};
} // namespace nncase::ntt
