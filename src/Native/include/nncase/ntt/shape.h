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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <utility>

namespace nncase::ntt {
namespace detail {
template <size_t... Dims> struct fixed_dims_base {
    static constexpr size_t rank() noexcept { return sizeof...(Dims); }

    static constexpr size_t at(size_t index) noexcept {
        return std::array<size_t, sizeof...(Dims)>{Dims...}[index];
    }

    constexpr size_t operator[](size_t index) const noexcept {
        return at(index);
    }
};

template <size_t Rank> struct ranked_dims_base {
    static constexpr size_t rank() noexcept { return Rank; }

    constexpr size_t operator[](size_t index) const noexcept {
        return at(index);
    }
    constexpr size_t &operator[](size_t index) noexcept { return at(index); }

    constexpr size_t at(size_t index) const noexcept { return dims_[index]; }
    constexpr size_t &at(size_t index) noexcept { return dims_[index]; }

    constexpr auto begin() const noexcept { return dims_.begin(); }
    constexpr auto end() const noexcept { return dims_.end(); }

    std::array<size_t, Rank> dims_;
};
} // namespace detail

template <size_t... Dims>
struct fixed_shape : detail::fixed_dims_base<Dims...> {
    template <size_t I> struct prepend {
        using type = fixed_shape<I, Dims...>;
    };

    template <size_t I> struct append {
        using type = fixed_shape<Dims..., I>;
    };

    static constexpr size_t length() noexcept {
        return sizeof...(Dims) ? (Dims * ...) : 1;
    }
};

template <size_t Rank> struct ranked_shape : detail::ranked_dims_base<Rank> {
    constexpr size_t length() noexcept {
        return std::accumulate(this->begin(), this->end(), 1,
                               std::multiplies<>());
    }
};

template <size_t... Strides>
struct fixed_strides : detail::fixed_dims_base<Strides...> {
    template <size_t I> struct prepend {
        using type = fixed_strides<I, Strides...>;
    };

    template <size_t I> struct append {
        using type = fixed_strides<Strides..., I>;
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
    using shape_t = fixed_shape<((void)Dims, Value)...>;
};
} // namespace detail

template <class Dims> struct is_fixed_dims : std::false_type {};

template <size_t... Dims>
struct is_fixed_dims<fixed_shape<Dims...>> : std::true_type {};

template <size_t... Dims>
struct is_fixed_dims<fixed_strides<Dims...>> : std::true_type {};

template <class Dims>
inline constexpr bool is_fixed_dims_v = is_fixed_dims<Dims>::value;

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

template <class Index, class Strides>
constexpr size_t linear_offset(const Index &index,
                               const Strides &strides) noexcept {
    size_t offset = 0;
    for (size_t i = 0; i < index.rank(); i++) {
        offset += index[i] * strides[i];
    }
    return offset;
}

template <class Shape, class Strides>
constexpr size_t linear_size(const Shape &shape,
                             const Strides &strides) noexcept {
    size_t max_stride = 1, max_shape = 1;
    for (size_t i = 0; i < shape.rank(); i++) {
        if ((shape[i] == 1 ? 0 : strides[i]) >= max_stride) {
            max_stride = strides[i];
            max_shape = shape[i];
        }
    }

    size_t size = max_stride * max_shape;
    return size;
}
} // namespace nncase::ntt