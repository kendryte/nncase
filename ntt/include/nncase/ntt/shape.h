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
#include "compiler_defs.h"
#include "dimension.h"
#include "loop.h"
#include "nncase/ntt/tensor_traits.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
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
    return Derived<cannonical_dim_t<TDims>...>{
        static_cast<cannonical_dim_t<TDims>>(dims)...};
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

template <template <class... TDims> class Derived, size_t Rank, dim_t Start = 0>
constexpr auto make_index_dims_impl() noexcept {
    auto index_impl = []<size_t... I>(std::index_sequence<I...>) {
        return make_dims_impl<Derived>(fixed_dim_v<Start + I>...);
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

struct default_aggregate_selector {
    template <class T>
    constexpr decltype(auto) operator()(T &&value) const noexcept {
        return std::forward<T>(value);
    }
};

template <dim_t Axis, class TDims, class TAccumulate, class TFunc,
          class TSelector, dim_t... I>
constexpr auto aggregate_impl(const TDims &dims, TAccumulate &&seed,
                              TFunc &&func, TSelector &&selector,
                              std::integer_sequence<dim_t, I...>) noexcept {
    auto new_seed = func(std::forward<TAccumulate>(seed),
                         dims[fixed_dim_v<Axis>], fixed_dim_v<Axis>);
    if constexpr (Axis + 1 < TDims::rank()) {
        return aggregate_impl<Axis + 1>(dims, std::move(new_seed),
                                        std::forward<TFunc>(func),
                                        std::forward<TSelector>(selector),
                                        std::integer_sequence<dim_t, I...>{});
    } else {
        return selector(std::move(new_seed));
    }
}

template <dim_t ReplaceAxis, dim_t CntIndex, class TDims, Dimension TDim>
constexpr auto replace_dim_impl(const TDims &dims, const TDim &dim) noexcept {
    if constexpr (ReplaceAxis == CntIndex) {
        return dim;
    } else {
        return dims.template at<CntIndex>();
    }
}

template <dims_usage Usage, template <class... TDims> class Derived,
          Dimension... TDims>
struct dims_base {
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
    constexpr decltype(auto) operator[](const TIndex &) noexcept {
        return at<TIndex::value>();
    }

    template <class TAccumulate, class TFunc,
              class TSelector = default_aggregate_selector>
    constexpr decltype(auto) aggregate(
        TAccumulate &&seed, [[maybe_unused]] TFunc &&func,
        [[maybe_unused]] TSelector &&selector = TSelector{}) const noexcept {
        if constexpr (rank() == 0) {
            return selector(std::forward<TAccumulate>(seed));
        } else {
            return aggregate_impl<0>(
                *this, std::forward<TAccumulate>(seed),
                std::forward<TFunc>(func), std::forward<TSelector>(selector),
                std::make_integer_sequence<dim_t, rank()>{});
        }
    }

    template <Dimension TIndex>
    constexpr auto at([[maybe_unused]] const TIndex &index) const noexcept {
        if constexpr (FixedDimension<TIndex>) {
            return at<TIndex::value>();
        } else {
            const auto pos_index = positive_index(index, rank());
            return to_array()[pos_index];
        }
    }

    template <FixedDimension TIndex>
    constexpr decltype(auto) at(const TIndex &) noexcept {
        return at<TIndex::value>();
    }

    template <dim_t Index> constexpr auto at() const noexcept {
        constexpr auto PositiveIndex = positive_index(Index, rank());
        if constexpr (is_fixed()) {
            return std::tuple_element_t<PositiveIndex, decltype(dims_)>{};
        } else {
            return std::get<PositiveIndex>(dims_);
        }
    }

    template <dim_t Index> constexpr decltype(auto) at() noexcept {
        constexpr auto PositiveIndex = positive_index(Index, rank());
        if constexpr (is_fixed()) {
            return std::tuple_element_t<PositiveIndex, decltype(dims_)>{};
        } else {
            return std::get<PositiveIndex>(dims_);
        }
    }

    constexpr auto front() const noexcept { return at<0>(); }
    constexpr decltype(auto) front() noexcept { return at<0>(); }

    constexpr auto back() const noexcept { return at<rank() - 1>(); }
    constexpr decltype(auto) back() noexcept { return at<rank() - 1>(); }

    template <Dimension TDim>
    constexpr auto contains([[maybe_unused]] const TDim &value) const noexcept {
        if constexpr (rank() == 0) {
            return std::false_type{};
        } else if constexpr (FixedDimension<TDim>) {
            constexpr auto fixed_contains =
                (false || ... ||
                 std::conditional_t<
                     FixedDimension<TDims>,
                     std::integral_constant<bool, TDims{} == TDim::value>,
                     std::false_type>{});
            if constexpr (fixed_contains) {
                return std::true_type{};
            } else if constexpr (is_fixed()) {
                return std::false_type{};
            } else {
                auto contains_impl =
                    [this, value]<size_t... I>(std::index_sequence<I...>) {
                        (void)this;
                        return (false || ... || (at(fixed_dim_v<I>) == value));
                    };
                return contains_impl(std::make_index_sequence<rank()>());
            }
        } else {
            auto contains_impl =
                [this, value]<size_t... I>(std::index_sequence<I...>) {
                    (void)this;
                    return (false || ... || (at(fixed_dim_v<I>) == value));
                };
            return contains_impl(std::make_index_sequence<rank()>());
        }
    }

    template <Dimension... UDims>
    constexpr auto append(const UDims &...values) const noexcept {
        auto append_impl = [&, this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at(fixed_dim_v<I>)..., values...);
        };
        return append_impl(std::make_index_sequence<rank()>());
    }

    template <Dimension... UDims>
    constexpr auto concat(const Derived<UDims...> &other) const noexcept {
        auto concat_impl = [this, &other]<size_t... I, size_t... U>(
                               std::index_sequence<I...>,
                               std::index_sequence<U...>) {
            (void)this;
            return make_dims_impl<Derived>(at(fixed_dim_v<I>)...,
                                           other.at(fixed_dim_v<U>)...);
        };
        return concat_impl(std::make_index_sequence<rank()>(),
                           std::make_index_sequence<sizeof...(UDims)>());
    }

    template <Dimension TDim>
    constexpr auto index_of(const TDim &dim) const noexcept {
        return aggregate(-1_dim, [&](auto acc, auto value, auto index) {
            return ntt::where(acc == -1_dim,
                              ntt::where(value == dim, index, -1_dim), acc);
        });
    }

    template <Dimension... UDims>
    constexpr auto prepend(const UDims &...values) const noexcept {
        auto prepend_impl = [&, this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(values..., at(fixed_dim_v<I>)...);
        };
        return prepend_impl(std::make_index_sequence<rank()>());
    }

    template <size_t Index> constexpr auto remove_at() const noexcept {
        auto remove_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<(I < Index ? I : I + 1)>()...);
        };
        return remove_impl(std::make_index_sequence<rank() - 1>());
    }

    template <dim_t Index, Dimension TDim>
    constexpr auto replace_at(const TDim &dim) const noexcept {
        constexpr auto PositiveIndex = positive_index(Index, rank());
        auto replace_impl = [&dim,
                             this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(
                replace_dim_impl<PositiveIndex, I>(*this, dim)...);
        };
        return replace_impl(std::make_index_sequence<rank()>());
    }

    constexpr auto reverse() const noexcept {
        auto reverse_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<rank() - 1 - I>()...);
        };
        return reverse_impl(std::make_index_sequence<rank()>());
    }

    template <FixedDimensions TIndicies>
    constexpr auto select(const TIndicies &indicies) const noexcept {
        auto select_impl = [&, this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at(indicies.at(fixed_dim_v<I>))...);
        };
        return select_impl(std::make_index_sequence<TIndicies::rank()>());
    }

    template <size_t Start, size_t Rank = rank() - Start>
    constexpr auto slice() const noexcept {
        auto slice_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return make_dims_impl<Derived>(at<I + Start>()...);
        };
        return slice_impl(std::make_index_sequence<Rank>());
    }

    constexpr std::array<dim_t, rank()> to_array() const noexcept {
        auto at_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return std::array<dim_t, rank()>{at(fixed_dim_v<I>)...};
        };
        return at_impl(std::make_index_sequence<rank()>());
    }

    template <class UDims> constexpr void copy_to(UDims &other) const noexcept {
        loop<rank()>([&](auto axis) { other[axis] = at(axis); });
    }

  private:
    NTT_NO_UNIQUE_ADDRESS std::tuple<TDims...> dims_;
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

    constexpr auto length() const noexcept {
        auto length_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
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

template <Dimensions TDims> struct empty_dims_alike_type;

template <Dimensions TDims>
using empty_dims_alike_t = typename empty_dims_alike_type<TDims>::type;

template <Dimensions TDims, size_t Rank> struct zeros_dims_alike_type;

template <Dimensions TDims, size_t Rank>
inline constexpr auto zeros_dims_alike_v =
    zeros_dims_alike_type<TDims, Rank>::value;

#define DEFINE_NTT_MAKE_DIMS(dims_type)                                        \
    template <size_t Rank>                                                     \
    using dynamic_##dims_type##_t =                                            \
        detail::dynamic_dims_t<dims_type##_t, Rank>;                           \
                                                                               \
    template <dim_t... Dims>                                                   \
    using fixed_##dims_type##_t = dims_type##_t<fixed_dim<Dims>...>;           \
                                                                               \
    template <Dimension... TDims>                                              \
    struct empty_dims_alike_type<dims_type##_t<TDims...>> {                    \
        using type = dims_type##_t<>;                                          \
    };                                                                         \
                                                                               \
    template <Dimension... TDims>                                              \
    constexpr auto make_##dims_type(const TDims &...dims) noexcept {           \
        return detail::make_dims_impl<dims_type##_t>(dims...);                 \
    }                                                                          \
                                                                               \
    template <dim_t... Dims>                                                   \
    inline constexpr auto fixed_##dims_type##_v =                              \
        detail::fixed_dims_impl_v<dims_type##_t, Dims...>;                     \
                                                                               \
    template <size_t Rank> constexpr auto make_zeros_##dims_type() noexcept {  \
        return detail::make_zeros_dims_impl<dims_type##_t, Rank>();            \
    }                                                                          \
                                                                               \
    template <size_t Rank> constexpr auto make_ones_##dims_type() noexcept {   \
        return detail::make_ones_dims_impl<dims_type##_t, Rank>();             \
    }                                                                          \
                                                                               \
    template <size_t Rank, dim_t Start = 0>                                    \
    constexpr auto make_index_##dims_type() noexcept {                         \
        return detail::make_index_dims_impl<dims_type##_t, Rank, Start>();     \
    }                                                                          \
                                                                               \
    template <size_t Rank, Dimension TDim>                                     \
    constexpr auto make_repeat_##dims_type(const TDim &dim) noexcept {         \
        return detail::make_repeat_dims_impl<dims_type##_t, Rank>(dim);        \
    }                                                                          \
                                                                               \
    template <size_t Rank, class TGenerator>                                   \
    constexpr auto generate_##dims_type(TGenerator &&generator) noexcept {     \
        return detail::generate_dims_impl<dims_type##_t, Rank>(generator);     \
    }                                                                          \
                                                                               \
    template <Dimension... TDims, size_t Rank>                                 \
    struct zeros_dims_alike_type<dims_type##_t<TDims...>, Rank> {              \
        static constexpr auto value = make_zeros_##dims_type<Rank>();          \
    };

DEFINE_NTT_MAKE_DIMS(dims)
DEFINE_NTT_MAKE_DIMS(shape)
DEFINE_NTT_MAKE_DIMS(strides)

#undef DEFINE_NTT_MAKE_DIMS

template <Shape TShape, Strides TStrides>
constexpr auto canonicalize_strides(const TShape &shape,
                                    const TStrides &strides) noexcept {
    static_assert(TShape::rank() == TStrides::rank(),
                  "Shape and strides must have the same rank");
    // Replace the stride with 0 if the shape is 1
    return generate_strides<TShape::rank()>([&](auto axis) {
        const auto dim = shape[axis];
        if constexpr (FixedDimension<decltype(dim)>) {
            return ntt::where(dim == dim_one, dim_zero, strides[axis]);
        } else {
            return dim == 1 ? 0 : dim_value(strides[axis]);
        }
    });
}

namespace detail {
template <size_t Axis, Shape TShape, bool Canonical = true> struct default_strides_impl {
    static_assert(Axis > 0 && Axis <= TShape::rank(), "Axis out of bounds");

    template <Strides TStrides>
    constexpr auto
    operator()([[maybe_unused]] const TShape &shape,
               [[maybe_unused]] const TStrides &cnt_strides) noexcept {
        auto new_stride = [&shape, &cnt_strides]() {
            if constexpr (Axis == TShape::rank()) {
                (void)shape;
                (void)cnt_strides;
                return dim_one;
            } else {
                auto dim = shape[fixed_dim_v<Axis>];
                auto last_stride = cnt_strides[dim_zero];
                return last_stride * dim;
            }
        }();
        auto new_strides = cnt_strides.prepend(new_stride);
        if constexpr (Axis == 1) {
          if constexpr(Canonical) {
            return canonicalize_strides(shape, new_strides);
          } else {
            return new_strides;   
          }
        } else {
            return default_strides_impl<Axis - 1, TShape, Canonical>{}(shape, new_strides);
        }
    }
};
} // namespace detail

template <dim_t ExtendRank, Strides TStrides>
constexpr auto broadcast_strides(const TStrides &strides) noexcept {
    static_assert(ExtendRank >= 0,
                  "Extend rank must be greater than or equal to 0");
    return make_zeros_strides<ExtendRank>().concat(strides);
}

template <Shape TShape, bool Canonical = true>
constexpr auto default_strides([[maybe_unused]] const TShape shape) noexcept {
    constexpr auto rank = TShape::rank();
    if constexpr (rank == 0) {
        return strides_t<>();
    } else {
        return detail::default_strides_impl<rank, TShape, Canonical>{}(shape,
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

template <Dimensions TIndex, Shape TShape>
constexpr auto linear_offset(const TIndex &index,
                             const TShape &shape) noexcept {
    static_assert(TIndex::rank() == TShape::rank(),
                  "index and shape must have the same rank");
    return linear_offset(index, default_strides(shape));
}

template <template <class... TDims> class TDimensions = shape_t,
          Dimension TOffset, Shape TShape>
constexpr auto unravel_index(const TOffset &offset,
                             const TShape &shape) noexcept {
    return shape.reverse().aggregate(
        std::make_tuple(offset, TDimensions<>{}),
        [&](auto acc, auto dim, [[maybe_unused]] auto axis) {
            auto [last_remain, index] = acc;
            auto cnt_index = last_remain % dim;
            auto remain = last_remain / dim;
            return std::make_tuple(remain, index.prepend(cnt_index));
        },
        [](auto acc) { return std::get<1>(acc); });
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
        const auto new_max_stride = ntt::where(
            cnt_dim == dim_one, max_stride,
            ntt::where(cnt_stride >= max_stride, cnt_stride, max_stride));
        const auto new_max_shape = ntt::where(
            cnt_dim == dim_one, max_shape,
            ntt::where(cnt_stride >= max_stride, cnt_dim, max_shape));
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
constexpr dim_t contiguous_dims_impl(const TShape &shape,
                                     const TStrides &strides) {
    auto def_strides = default_strides(shape);
    auto def_strides_2 = default_strides<TShape, false>(shape);
    for (ptrdiff_t i = (ptrdiff_t)strides.rank() - 1; i >= 0; --i) {
        if (strides[i] != def_strides[i] && !(shape[i] == dim_one && strides[i] == def_strides_2[i])) {
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

namespace detail {
template <FixedShape Axes, size_t CntAxis> struct squeeze_dims_impl {
    template <Dimensions TSrcDims, Dimensions TResultDims>
    constexpr auto operator()(const TSrcDims &src_dims,
                              const TResultDims &result_dims) {
        static_assert(CntAxis < TSrcDims::rank(), "CntAxis out of bounds");
        auto new_result_dims =
            ntt::where(Axes{}.contains(fixed_dim_v<CntAxis>), result_dims,
                       result_dims.append(src_dims[fixed_dim_v<CntAxis>]));
        if constexpr (CntAxis + 1 < TSrcDims::rank()) {
            return squeeze_dims_impl<Axes, CntAxis + 1>()(src_dims,
                                                          new_result_dims);
        } else {
            return new_result_dims;
        }
    }
};
} // namespace detail

template <Dimensions TDims, FixedDimensions TAxes>
constexpr auto squeeze_dims(const TDims &dims, const TAxes &) noexcept {
    constexpr auto positive_axes_v = positive_axes(TAxes{}, TDims::rank());
    return dims.aggregate(empty_dims_alike_t<TDims>{},
                          [&](auto result, auto dim, auto axis) {
                              if constexpr (positive_axes_v.contains(axis)) {
                                  return result;
                              } else {
                                  return result.append(dim);
                              }
                          });
}

template <Dimensions TDims, FixedDimensions TAxes, Dimension TDim>
constexpr auto unsqueeze_dims(const TDims &dims, const TAxes &,
                              const TDim &insert_dim) noexcept {
    constexpr auto positive_axes_v = positive_axes(TAxes{}, TDims::rank());
    return make_zeros_shape<TDims::rank() + TAxes::rank()>().aggregate(
        std::make_tuple(empty_dims_alike_t<TDims>{}, dim_zero),
        [&](auto acc, auto, auto axis) {
            auto [last_result, offset] = acc;
            if constexpr (positive_axes_v.contains(axis)) {
                return std::make_tuple(last_result.append(insert_dim), offset);
            } else {
                return std::make_tuple(last_result.append(dims[offset]),
                                       offset + dim_one);
            }
        },
        [](auto acc) { return std::get<0>(acc); });
}

template <Dimensions TDimsA, Dimensions TDimsB>
constexpr bool operator==([[maybe_unused]] const TDimsA &lhs,
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

namespace std {
// structured bindings support
template <nncase::ntt::Dimensions TDims>
struct tuple_size<TDims> : std::integral_constant<size_t, TDims::rank()> {};

template <size_t I, template <class... TDims> class Derived,
          nncase::ntt::Dimension... TDims>
    requires(nncase::ntt::Dimensions<Derived<TDims...>>)
struct tuple_element<I, Derived<TDims...>> {
    using type = std::tuple_element_t<I, std::tuple<TDims...>>;
};

template <size_t I, nncase::ntt::Dimensions TDims>
constexpr auto get(const TDims &dims) noexcept {
    return dims.template at<I>();
}
} // namespace std
