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
#include "dimension.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <sys/stat.h>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
enum dims_usage {
    normal,
    shape,
    strides,
};

namespace detail {
template <class T> struct cannonical_dim_type {
    using type = std::conditional_t<FixedDimension<std::decay_t<T>>,
                                    std::decay_t<T>, dim_t>;
};

template <class T>
using cannonical_dim_t = typename cannonical_dim_type<T>::type;

template <template <class... TDims> class Derived, class I>
struct dynamic_dims_type_impl;

template <template <class... TDims> class Derived, size_t... I>
struct dynamic_dims_type_impl<Derived, std::index_sequence<I...>> {
    template <std::size_t> using elem_type = dim_t;

    using type = Derived<elem_type<I>...>;
};

template <template <class... TDims> class Derived, size_t Rank>
using dynamic_dims_t =
    typename dynamic_dims_type_impl<Derived,
                                    std::make_index_sequence<Rank>>::type;

template <template <class... TDims> class Derived, Dimension... TDims>
constexpr auto make_dims_impl(const TDims &...dims) noexcept {
    return Derived<cannonical_dim_t<TDims>...>{dims...};
}

template <template <class... TDims> class Derived, dim_t... Dims>
inline constexpr auto fixed_dims_impl_v =
    make_dims_impl<Derived>(fixed_dim_v<Dims>...);

template <template <class... TDims> class Derived, size_t Rank, Dimension TDim>
constexpr auto make_repeat_dims_impl(const TDim &dim) noexcept {
    auto repeat_impl = [dim]<size_t... I>(std::index_sequence<I...>) {
        return make_dims_impl<Derived>(((void)I, dim)...);
    };
    return repeat_impl(std::make_index_sequence<Rank>());
}

template <template <class... TDims> class Derived, size_t Rank>
constexpr auto make_zeros_dims_impl() noexcept {
    return make_repeat_dims_impl<Derived, Rank>(dim_zero);
}

template <template <class... TDims> class Derived, size_t Rank>
constexpr auto make_ones_dims_impl() noexcept {
    return make_repeat_dims_impl<Derived, Rank>(dim_one);
}

template <template <class... TDims> class Derived, size_t Rank>
constexpr auto make_index_dims_impl() noexcept {
    auto index_impl = []<size_t... I>(std::index_sequence<I...>) {
        return make_dims_impl<Derived>(fixed_dim_v<I>...);
    };
    return index_impl(std::make_index_sequence<Rank>());
}

template <template <class... TDims> class Derived, size_t Rank,
          class TGenerator>
constexpr auto generate_dims_impl(TGenerator &&generator) noexcept {
    auto generate_impl =
        [&generator]<dim_t... I>(std::integer_sequence<dim_t, I...>) {
            return make_dims_impl<Derived>(generator(fixed_dim_v<I>)...);
        };
    return generate_impl(std::make_integer_sequence<dim_t, Rank>());
}

template <dims_usage Usage, template <class... TDims> class Derived,
          Dimension... TDims>
struct dims_base {
    struct iterator {
        using dims_type = dims_base<Usage, Derived, TDims...>;
        using difference_type = std::ptrdiff_t;
        using element_type = dim_t;

        constexpr iterator(const dims_type &dims, size_t index = 0) noexcept
            : dims_(dims), index_(index) {}

        element_type operator*() const { return dims_.at(index_); }

        iterator &operator++() {
            index_++;
            return *this;
        }
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        iterator &operator+=(int i) {
            index_ += i;
            return *this;
        }
        iterator operator+(const difference_type other) const {
            return index_ + other;
        }
        friend iterator operator+(const difference_type value,
                                  const iterator &other) {
            return other + value;
        }

        iterator &operator--() {
            index_--;
            return *this;
        }
        iterator operator--(int) {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }
        iterator &operator-=(int i) {
            index_ -= i;
            return *this;
        }
        difference_type operator-(const iterator &other) const {
            return index_ - other.index_;
        }
        iterator operator-(const difference_type other) const {
            return index_ - other;
        }
        friend iterator operator-(const difference_type value,
                                  const iterator &other) {
            return other - value;
        }

        element_type operator[](difference_type idx) const {
            return dims_.at(index_ + idx);
        }

        bool operator==(const iterator &other) const {
            return index_ == other.index_;
        }

      private:
        const dims_type &dims_;
        size_t index_;
    };

    constexpr dims_base(const TDims &...dims) noexcept
        : dims_(std::make_tuple(dims...)) {}

    static constexpr dims_usage usage() noexcept { return Usage; }

    static constexpr auto rank() noexcept {
        return fixed_dim_v<sizeof...(TDims)>;
    }

    static constexpr auto fixed_rank() noexcept {
        return fixed_dim_v<(0 + ... + (is_fixed_dim_v<TDims> ? 1 : 0))>;
    }

    static constexpr auto dynamic_rank() noexcept {
        return rank() - fixed_rank();
    }

    static constexpr bool is_fixed() noexcept { return fixed_rank() == rank(); }

    template <size_t Rank = rank(), class = std::enable_if_t<Rank != 0>>
    constexpr dims_base() noexcept : dims_(std::make_tuple(TDims{}...)) {}

    template <Dimension TIndex>
    constexpr auto operator[](const TIndex &index) const noexcept {
        return at(index);
    }

    template <FixedDimension TIndex>
    constexpr auto &operator[](const TIndex &) noexcept {
        return at(TIndex{});
    }

    template <Dimension TIndex>
    constexpr auto at(const TIndex &) const noexcept {
        if constexpr (FixedDimension<TIndex>) {
            return std::get<TIndex::value>(dims_);
        } else {
            return to_array()[TIndex{}];
        }
    }

    template <FixedDimension TIndex>
    constexpr auto &at(const TIndex &) noexcept {
        return std::get<TIndex::value>(dims_);
    }

    template <size_t TIndex> constexpr auto at() const noexcept {
        return std::get<TIndex>(dims_);
    }

    template <size_t TIndex> constexpr auto &at() noexcept {
        return std::get<TIndex>(dims_);
    }

    constexpr auto begin() const noexcept { return iterator(); }
    constexpr auto end() const noexcept { return iterator(rank()); }

    constexpr auto last() const noexcept { return at<rank() - 1>(); }
    constexpr auto &last() noexcept { return at<rank() - 1>(); }

    template <Dimension TDim>
    constexpr bool contains([[maybe_unused]] const TDim &value) noexcept {
        auto contains_impl = [this,
                              value]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return (false || ... || (at(fixed_dim_v<I>) == value));
        };
        return contains_impl(std::make_index_sequence<rank()>());
    }

    template <Dimension... UDims>
    constexpr Derived<TDims..., UDims...>
    append(const UDims &...values) const noexcept {
        auto append_impl = [this]<size_t... I>(const UDims &...values,
                                               std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at(fixed_dim_v<I>)..., values...);
        };
        return append_impl(
            values..., std::make_index_sequence<rank() + sizeof...(UDims)>());
    }

    template <Dimension... UDims>
    constexpr Derived<UDims..., TDims...>
    prepend(const UDims &...values) const noexcept {
        auto prepend_impl = [this]<size_t... I>(const UDims &...values,
                                                std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(values..., at(fixed_dim_v<I>)...);
        };
        return prepend_impl(values..., std::make_index_sequence<rank()>());
    }

    template <size_t Index> constexpr auto remove_at() {
        auto remove_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<(I < Index ? I : I + 1)>()...);
        };
        return remove_impl(std::make_index_sequence<rank() - 1>());
    }

    template <size_t Index, Dimension TDim>
    constexpr auto replace_at(const TDim &dim) {
        auto replace_impl = [&dim,
                             this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>((I == Index ? dim : at<I>())...);
        };
        return replace_impl(std::make_index_sequence<rank() - 1>());
    }

    constexpr auto reverse() const noexcept {
        auto reverse_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<rank() - 1 - I>()...);
        };
        return reverse_impl(std::make_index_sequence<rank()>());
    }

    template <size_t Start, size_t Rank = rank() - Start>
    constexpr auto slice() const noexcept {
        auto slice_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<I + Start>()...);
        };
        return slice_impl(std::make_index_sequence<Rank>());
    }

    template <Dimension... UDims>
    constexpr Derived<TDims..., UDims...>
    concat(const Derived<UDims...> &other) const noexcept {
        auto concat_impl = [this, &other]<size_t... I, size_t... U>(
                               std::index_sequence<I...>,
                               std::index_sequence<U...>) {
            (void)this;
            return make_dims_impl<Derived>(at(fixed_dim_v<I>)...,
                                           other.at(fixed_dim_v<U>)...);
        };
        return concat_impl(std::make_index_sequence<rank()>(),
                           std::make_index_sequence<other.rank()>());
    }

    constexpr std::array<dim_t, rank()> to_array() const noexcept {
        auto at_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return std::array<dim_t, rank()>{at(fixed_dim_v<I>)...};
        };
        return at_impl(std::make_index_sequence<rank()>());
    }

  private:
    [[no_unique_address]] std::tuple<TDims...> dims_;
};
} // namespace detail

template <Dimension... TDims>
struct dims_t : detail::dims_base<dims_usage::normal, dims_t, TDims...> {
    using base_t = detail::dims_base<dims_usage::normal, dims_t, TDims...>;
    using base_t::base_t;
};

template <Dimension... TDims>
struct shape_t : detail::dims_base<dims_usage::shape, shape_t, TDims...> {
    using base_t = detail::dims_base<dims_usage::shape, shape_t, TDims...>;
    using base_t::base_t;

    constexpr size_t length() const noexcept {
        auto length_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            return (dim_one * ... * base_t::at(fixed_dim_v<I>));
        };
        return length_impl(std::make_index_sequence<base_t::rank()>());
    }
};

template <Dimension... TDims>
struct strides_t : detail::dims_base<dims_usage::strides, strides_t, TDims...> {
    using base_t = detail::dims_base<dims_usage::strides, strides_t, TDims...>;
    using base_t::base_t;
};

template <class T>
concept Dimensions = requires {
    T::rank();
    T::fixed_rank();
    T::dynamic_rank();
    T::usage();
};

template <class T>
concept FixedDimensions = Dimensions<T> && T::is_fixed();

template <class T>
concept Shape = Dimensions<T> && T::usage() == dims_usage::shape;

template <class T>
concept FixedShape = Shape<T> && T::is_fixed();

template <class T>
concept Strides = Dimensions<T> && T::usage() == dims_usage::strides;

template <class T>
concept FixedStrides = Strides<T> && T::is_fixed();

template <Dimensions TDims> struct empty_dims_alike_type;

template <Dimensions TDims>
using empty_dims_alike_t = typename empty_dims_alike_type<TDims>::type;

#define DEFINE_NTT_MAKE_DIMS(type)                                             \
    template <size_t Rank>                                                     \
    using dynamic_##type##_t = detail::dynamic_dims_t<type##_t, Rank>;         \
                                                                               \
    template <Dimension... TDims>                                              \
    struct empty_dims_alike_type<type##_t<TDims...>> {                         \
        using type = type##_t<>;                                               \
    };                                                                         \
                                                                               \
    template <Dimension... TDims>                                              \
    constexpr auto make_##type(const TDims &...dims) noexcept {                \
        return detail::make_dims_impl<type##_t>(dims...);                      \
    }                                                                          \
                                                                               \
    template <dim_t... Dims>                                                   \
    inline constexpr auto fixed_##type##_v =                                   \
        detail::fixed_dims_impl_v<type##_t, Dims...>;                          \
                                                                               \
    template <size_t Rank> constexpr auto make_zeros_##type() noexcept {       \
        return detail::make_zeros_dims_impl<type##_t, Rank>();                 \
    }                                                                          \
                                                                               \
    template <size_t Rank> constexpr auto make_ones_##type() noexcept {        \
        return detail::make_ones_dims_impl<type##_t, Rank>();                  \
    }                                                                          \
                                                                               \
    template <size_t Rank> constexpr auto make_index_##type() noexcept {       \
        return detail::make_index_dims_impl<type##_t, Rank>();                 \
    }                                                                          \
                                                                               \
    template <size_t Rank, Dimension TDim>                                     \
    constexpr auto make_repeat_##type(const TDim &dim) noexcept {              \
        return detail::make_repeat_dims_impl<type##_t, Rank>(dim);             \
    }                                                                          \
                                                                               \
    template <size_t Rank, class TGenerator>                                   \
    constexpr auto generate_##type(TGenerator &&generator) noexcept {          \
        return detail::generate_dims_impl<type##_t, Rank>(generator);          \
    }

DEFINE_NTT_MAKE_DIMS(dims)
DEFINE_NTT_MAKE_DIMS(shape)
DEFINE_NTT_MAKE_DIMS(strides)

#undef DEFINE_NTT_MAKE_DIMS

namespace detail {
template <size_t Axis, Shape TShape> struct default_strides_impl {
    static_assert(Axis > 0 && Axis <= TShape::rank(), "Axis out of bounds");

    template <Strides TStrides>
    constexpr auto
    operator()([[maybe_unused]] const TShape &shape,
               [[maybe_unused]] const TStrides &strides) noexcept {
        auto new_stride = [&shape, &strides]() {
            if constexpr (Axis == TShape::rank()) {
                (void)shape;
                (void)strides;
                return dim_one;
            } else {
                auto dim = shape[fixed_dim_v<Axis>];
                auto last_stride = strides[dim_zero];
                return last_stride * dim;
            }
        }();
        auto new_strides = strides.prepend(new_stride);
        if constexpr (Axis == 1) {
            return new_strides;
        } else {
            return default_strides_impl<Axis - 1, TShape>{}(shape, new_strides);
        }
    }
};
} // namespace detail

template <Shape TShape>
constexpr auto default_strides([[maybe_unused]] const TShape shape) noexcept {
    constexpr auto rank = TShape::rank();
    if constexpr (rank == 0) {
        return strides_t<>();
    } else {
        return detail::default_strides_impl<rank, TShape>{}(shape,
                                                            strides_t<>());
    }
}

template <Dimensions TIndex, Strides TStrides>
constexpr auto linear_offset(const TIndex &index,
                             const TStrides &strides) noexcept {
    static_assert(TIndex::rank() == TStrides::rank(),
                  "index and strides must have the same rank");

    constexpr auto rank = TIndex::rank();
    if constexpr (rank == 0) {
        return dim_zero;
    } else {
        auto impl = [index, strides]<size_t... I>(std::index_sequence<I...>) {
            return ((index.at(fixed_dim_v<I>) * strides.at(fixed_dim_v<I>)) +
                    ...);
        };
        return impl(std::make_index_sequence<rank>());
    }
}

namespace detail {
template <size_t Axis, Strides TStrides> struct unravel_index_impl;

template <size_t Axis, Dimension... TStrides>
struct unravel_index_impl<Axis, strides_t<TStrides...>> {
    static_assert(Axis < sizeof...(TStrides), "Axis out of bounds");

    template <Dimension TOffset, Dimensions TIndex>
    constexpr auto operator()(const TIndex &index, const TOffset &offset,
                              const strides_t<TStrides...> &strides) noexcept {
        const auto stride = strides[Axis];
        auto cnt_index = offset / stride;
        auto remain = offset % stride;
        auto result = index.append(cnt_index);
        if constexpr (Axis + 1 < sizeof...(TStrides)) {
            return unravel_index_impl<Axis + 1, strides_t<TStrides...>>{}(
                result, remain, strides);
        } else {
            return result;
        }
    }
};
} // namespace detail

template <template <class... TDims> class TDimensions = dims_t,
          Dimension TOffset, Strides TStrides>
constexpr auto unravel_index(const TOffset &offset,
                             const TStrides &strides) noexcept {
    if constexpr (TStrides::rank() == 0) {
        return detail::make_dims_impl<TDimensions>(offset);
    } else {
        return detail::unravel_index_impl<0, TStrides>{}(TDimensions<>{},
                                                         offset, strides);
    }
}

namespace detail {
template <size_t Axis, Shape TShape, Strides TStrides> struct linear_size_impl {
    static_assert(Axis < TShape::rank() && Axis < TStrides::rank(),
                  "Axis out of bounds");

    template <Dimension TMaxStride, Dimension TMaxShape>
    constexpr auto operator()(TMaxStride max_stride, TMaxShape max_shape,
                              const TShape &shape,
                              const TStrides &strides) noexcept {
        const auto cnt_dim = shape.at(fixed_dim_v<Axis>);
        const auto cnt_stride = strides.at(fixed_dim_v<Axis>);
        auto [new_max_stride, new_max_shape] = [cnt_dim, cnt_stride, max_stride,
                                                max_shape]() {
            (void)cnt_stride;
            (void)max_shape;
            if constexpr (FixedDimension<std::decay_t<decltype(cnt_dim)>> &&
                          FixedDimension<std::decay_t<decltype(cnt_stride)>>) {
                if constexpr (cnt_dim == 1) {
                    return std::make_pair(max_stride, max_shape);
                } else if constexpr (cnt_stride >= max_stride) {
                    return std::make_pair(cnt_stride, cnt_dim);
                } else {
                    return std::make_pair(max_stride, max_shape);
                }
            } else {
                if (cnt_dim == 1) {
                    return std::make_pair(dim_value(max_stride),
                                          dim_value(max_shape));
                } else if (cnt_stride >= max_stride) {
                    return std::make_pair(dim_value(cnt_stride),
                                          dim_value(cnt_dim));
                } else {
                    return std::make_pair(dim_value(max_stride),
                                          dim_value(max_shape));
                }
            }
        }();
        if constexpr (Axis + 1 < TStrides::rank()) {
            return linear_size_impl<Axis + 1, TShape, TStrides>{}(
                new_max_stride, new_max_shape, shape, strides);
        } else {
            return new_max_stride * new_max_shape;
        }
    }
};
} // namespace detail

template <Shape TShape, Strides TStrides>
constexpr auto linear_size(const TShape &shape,
                           const TStrides &strides) noexcept {
    static_assert(TShape::rank() == TStrides::rank(),
                  "shape and strides must have the same rank");
    if constexpr (TShape::rank() == 0) {
        return dim_one;
    } else {
        return detail::linear_size_impl<0, TShape, TStrides>{}(dim_one, dim_one,
                                                               shape, strides);
    }
}

namespace detail {
template <Shape TShape, Strides TStrides>
constexpr size_t contiguous_dims_impl(const TShape &shape,
                                      const TStrides &strides) {
    auto def_strides = default_strides(shape);
    for (ptrdiff_t i = (ptrdiff_t)strides.rank() - 1; i >= 0; --i) {
        if (strides[i] != def_strides[i]) {
            return shape.rank() - i - 1;
        }
    }
    return shape.rank();
}
} // namespace detail

/**
 * @brief calculate the number of contigous dimensions.
 *
 * @param shape fixed/ranked
 * @param strides fixed/ranked
 * @return constexpr size_t contigous dimension numbers.
 */
template <Shape TShape, Strides TStrides>
constexpr auto contiguous_dims([[maybe_unused]] const TShape &shape,
                               [[maybe_unused]] const TStrides &strides) {
    if constexpr (TShape::is_fixed() && TStrides::is_fixed()) {
        return fixed_dim_v<detail::contiguous_dims_impl(TShape{},
                                                        TStrides{})>();
    } else {
        return detail::contiguous_dims_impl(shape, strides);
    }
}

namespace detail {
template <Shape TShape, Strides TStrides> constexpr size_t max_size_impl() {
    if constexpr (TShape::is_fixed() && TStrides::is_fixed()) {
        return linear_size(TShape{}, TStrides{});
    } else {
        return std::dynamic_extent;
    }
}
} // namespace detail

template <Shape TShape, Strides TStrides>
inline constexpr size_t max_size_v = detail::max_size_impl<TShape, TStrides>();

template <class Index, class Shape>
constexpr bool in_bound(const Index &index, const Shape &shape) {
    if (index.rank() == shape.rank()) {
        for (size_t i = 0; i < index.rank(); i++) {
            if (index[i] >= shape[i]) {
                return false;
            }
        }

        return true;
    }

    return false;
}

template <size_t Rank, class Index, class Shape, size_t DimsExt>
constexpr dynamic_shape_t<Rank> get_reduced_offset(Index in_offset,
                                                   Shape reduced_shape) {
    dynamic_shape_t<Rank> off;
    for (size_t i = 0; i < reduced_shape.rank(); i++) {
        off.at(i) = (in_offset.at(i + DimsExt) >= reduced_shape.at(i))
                        ? 0
                        : in_offset.at(i + DimsExt);
    }
    return off;
}

template <size_t Rank, class Index, class Shape>
dynamic_shape_t<Rank> get_reduced_offset(Index in_offset, Shape reduced_shape) {
    dynamic_shape_t<Rank> off;
    const auto dims_ext = in_offset.rank() - reduced_shape.rank();
    for (size_t i = 0; i < reduced_shape.rank(); i++) {
        if (in_offset.at(i + dims_ext) >= reduced_shape.at(i))
            off.at(i) = 0;
        else
            off.at(i) = in_offset.at(i + dims_ext);
    }

    return off;
}

template <size_t Axes, size_t Rank, class Index>
dynamic_shape_t<Rank> get_reduced_offset(Index in_offset) {
    dynamic_shape_t<Rank> off;
    for (size_t i = 0; i < Axes; i++) {
        off.at(i) = in_offset.at(i);
    }

    if constexpr (in_offset.rank() == Rank) {
        off.at(Axes) = 0;
        for (size_t i = Axes + 1; i < in_offset.rank(); i++) {
            off.at(i) = in_offset.at(i);
        }
    } else {
        for (size_t i = Axes + 1; i < in_offset.rank(); i++) {
            off.at(i - 1) = in_offset.at(i);
        }
    }

    return off;
}

namespace detail {
template <FixedShape Axes, size_t CntAxis> struct squeeze_dims_impl {
    static_assert(CntAxis < Axes::rank(), "CntAxis out of bounds");

    template <Dimensions TSrcDims, Dimensions TResultDims>
    constexpr auto operator()(const TSrcDims &src_dims,
                              const TResultDims &result_dims) noexcept {
        auto new_result_dims = [&src_dims, &result_dims] {
            if constexpr (Axes::contains(CntAxis)) {
                return result_dims.append(src_dims.at(CntAxis));
            } else {
                return result_dims;
            }
        }();
        if constexpr (CntAxis + 1 < Axes::rank()) {
            return squeeze_dims_impl<Axes, CntAxis + 1>{}(src_dims,
                                                          new_result_dims);
        } else {
            return new_result_dims;
        }
    }
};
} // namespace detail

template <Dimensions TDims, FixedShape TAxes>
constexpr auto squeeze_dims(const TDims &dims, const TAxes &) noexcept {
    return detail::squeeze_dims_impl<TAxes, 0>{}(dims,
                                                 empty_dims_alike_t<TDims>{});
}

template <Dimensions TDimsA, Dimensions TDimsB>
bool operator==([[maybe_unused]] const TDimsA &lhs,
                [[maybe_unused]] const TDimsB &rhs) noexcept {
    if constexpr (TDimsA::rank() != TDimsB::rank()) {
        return false;
    } else {
        for (size_t i = 0; i < TDimsA::rank(); i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
}

template <Dimension TIndex, Dimension TExtent>
constexpr auto positive_index(const TIndex &index,
                              [[maybe_unused]] const TExtent &dim) {
    if constexpr (FixedDimension<TIndex>) {
        if constexpr (TIndex::value < 0) {
            return index + dim;
        } else {
            return index;
        }
    } else {
        return index < 0 ? index + dim : index;
    }
}

template <Dimensions TIndex, Shape TShape>
constexpr auto positive_index(const TIndex &index, const TShape &shape) {
    static_assert(TIndex::rank() == TShape::rank(),
                  "index and shape must have the same rank");
    return generate_shape<TIndex::rank()>([&index, &shape](auto axis) {
        return positive_index(index[axis], shape[axis]);
    });
}

template <Dimensions TAxes, Dimension TExtent>
constexpr auto positive_axes(const TAxes &axes, const TExtent &dim) {
    return generate_shape<TAxes::rank()>(
        [&axes, &dim](auto axis) { return positive_index(axes[axis], dim); });
}
} // namespace nncase::ntt
