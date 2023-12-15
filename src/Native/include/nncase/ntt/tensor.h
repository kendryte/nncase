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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

namespace nncase::ntt {
template <class T, class Shape, class Strides> class tensor_view;

// data types
namespace detail {
template <size_t... Dims> struct fixed_dims_base {
    static constexpr size_t rank() noexcept { return sizeof...(Dims); }

    static constexpr size_t at(size_t index) noexcept {
        return std::array<size_t, sizeof...(Dims)>{Dims...}[index];
    }
};
} // namespace detail

template <size_t... Dims>
struct fixed_shape : detail::fixed_dims_base<Dims...> {
    static constexpr size_t length() noexcept {
        return sizeof...(Dims) ? (Dims + ...) : 1;
    }
};

template <size_t... Strides>
struct fixed_strides : detail::fixed_dims_base<Strides...> {
    template <size_t I> struct prepend {
        using type = fixed_strides<I, Strides...>;
    };
};

namespace detail {
template <size_t I, size_t... Dims> struct default_strides_impl;

template <size_t I> struct default_strides_impl<I> {
    inline static constexpr size_t value = 1;
    using strides_t = fixed_strides<value>;
};

template <size_t I, size_t Dim, size_t... Dims>
struct default_strides_impl<I, Dim, Dims...> {
    using next_impl_t = default_strides_impl<I + 1, Dims...>;
    inline static constexpr size_t value = Dim * next_impl_t::value;
    using strides_t =
        typename next_impl_t::strides_t::template prepend<value>::type;
};

template <size_t Value, class Dims> struct repeat_shape_impl;

template <size_t Value, size_t... Dims>
struct repeat_shape_impl<Value, std::index_sequence<Dims...>> {
    using shape_t = fixed_shape<(Dims, Value)...>;
};

template <class Offset, class Strides, size_t... I>
constexpr size_t linear_offset(std::index_sequence<I...>) noexcept {
    return ((Offset::at(I) * Strides::at(I)) + ... + 0);
}
} // namespace detail

template <class Shape> struct default_strides;

template <> struct default_strides<fixed_shape<>> {
    using type = fixed_strides<>;
};

template <size_t Dim, size_t... Dims>
struct default_strides<fixed_shape<Dim, Dims...>> {
    using type = typename detail::default_strides_impl<0, Dims...>::strides_t;
};

template <class Shape>
using default_strides_t = typename default_strides<Shape>::type;

template <size_t Value, size_t Rank>
using repeat_shape_t =
    typename detail::repeat_shape_impl<Value,
                                       std::make_index_sequence<Rank>>::shape_t;

template <size_t Rank> using zero_shape_t = repeat_shape_t<0, Rank>;

template <class Offset, class Strides>
constexpr size_t linear_offset() noexcept {
    return detail::linear_offset<Offset, Strides>(
        std::make_index_sequence<Offset::rank()>());
}

template <class Shape, class Strides> constexpr size_t linear_size() noexcept {
    size_t max_stride = 1, max_shape = 1;
    for (size_t i = 0; i < Shape::rank(); i++) {
        if ((Shape::at(i) == 1 ? 0 : Strides::at(i)) >= max_stride) {
            max_stride = Strides::at(i);
            max_shape = Shape::at(i);
        }
    }

    size_t size = max_stride * max_shape;
    return size;
}

template <class T, class Shape, class Strides = default_strides_t<Shape>>
class tensor {
  public:
    static constexpr auto shape() noexcept { return Shape{}; }
    static constexpr auto strides() noexcept { return Strides{}; }

    auto buffer() noexcept { return std::span(buffer_); }

    template <class Offset = zero_shape_t<Shape::rank()>, class UShape = Shape>
    tensor_view<T, UShape, Strides> view() noexcept;

  private:
    T buffer_[linear_size<Shape, Strides>()];
};

template <class T, class Shape, class Strides> class tensor_view {
  public:
    static constexpr auto shape() noexcept { return Shape{}; }
    static constexpr auto strides() noexcept { return Strides{}; }

    tensor_view(std::span<T, linear_size<Shape, Strides>()> buffer) noexcept
        : buffer_(buffer) {}

    auto buffer() noexcept { return buffer_; }

    template <class Offset = zero_shape_t<Shape::rank()>, class UShape = Shape>
    tensor_view<T, UShape, Strides> view() noexcept {
        return buffer_.template subspan<linear_offset<Offset, Strides>()>();
    }

  private:
    std::span<T, linear_size<Shape, Strides>()> buffer_;
};

template <class T, class Shape, class Strides>
template <class Offset, class UShape>
tensor_view<T, UShape, Strides> tensor<T, Shape, Strides>::view() noexcept {
    return {buffer().template subspan<linear_offset<Offset, Strides>()>()};
}

// tensors
template <class T, class Shape, class StridesA, class StridesB>
void tensor_copy(tensor_view<T, Shape, StridesA> src,
                 tensor_view<T, Shape, StridesB> dest) noexcept {
    auto src_buffer = src.buffer();
    std::copy(src_buffer.begin(), src_buffer.end(), dest.buffer().begin());
}
} // namespace nncase::ntt
