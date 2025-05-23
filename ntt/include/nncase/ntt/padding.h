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
#include "shape.h"
#include <array>
#include <tuple>

namespace nncase::ntt {
template <Dimension TBefore, Dimension TAfter> struct padding_t {
    using before_type = TBefore;
    using after_type = TAfter;

    constexpr padding_t() = default;

    constexpr padding_t(const TBefore &before, const TAfter &after) noexcept
        : before(before), after(after) {}

    constexpr auto sum() const noexcept { return before() + after(); }

    constexpr std::array<dim_t, 2> to_array() const noexcept {
        return {dim_value(before), dim_value(after)};
    }

    [[no_unique_address]] TBefore before;
    [[no_unique_address]] TAfter after;
};

inline constexpr padding_t<fixed_dim<0>, fixed_dim<0>> padding_zero{};

template <class T>
concept Padding = requires {
    typename T::before_type;
    typename T::after_type;
};

template <class T>
concept FixedPadding = Padding<T> && FixedDimension<typename T::before_type> &&
                       FixedDimension<typename T::after_type>;

template <Padding TLhs, Padding TRhs>
constexpr bool operator==(const TLhs &lhs, const TRhs &rhs) noexcept {
    return lhs.before() == rhs.before() && lhs.after() == rhs.after();
}

template <Dimension TBefore, Dimension TAfter>
constexpr auto make_padding(const TBefore &before,
                            const TAfter &after) noexcept {
    return padding_t<detail::cannonical_dim_t<TBefore>,
                     detail::cannonical_dim_t<TAfter>>(before, after);
}

template <dim_t Before, dim_t After>
inline constexpr auto fixed_padding_v =
    make_padding(fixed_dim_v<Before>, fixed_dim_v<After>);

template <Padding... TPaddings> class paddings_t {
  public:
    using paddings_type = std::tuple<TPaddings...>;

    constexpr paddings_t() = default;

    constexpr paddings_t(const TPaddings &...paddings) noexcept
        : paddings_(std::make_tuple(paddings...)) {}

    static constexpr auto rank() noexcept {
        return fixed_dim_v<sizeof...(TPaddings)>;
    }

    static constexpr auto fixed_rank() noexcept {
        return fixed_dim_v<(0 + ... + (FixedPadding<TPaddings> ? 1 : 0))>;
    }

    static constexpr auto dynamic_rank() noexcept {
        return rank() - fixed_rank();
    }

    static constexpr bool is_fixed() noexcept { return fixed_rank() == rank(); }

    template <size_t Rank = rank(), class = std::enable_if_t<Rank != 0>>
    constexpr paddings_t() noexcept
        : paddings_(std::make_tuple(TPaddings{}...)) {}

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
            return std::get<TIndex::value>(paddings_);
        } else {
            return to_array()[TIndex{}];
        }
    }

    template <dim_t TIndex> constexpr auto at() const noexcept {
        return std::get<TIndex>(paddings_);
    }

    template <dim_t TIndex> constexpr auto &at() noexcept {
        return std::get<TIndex>(paddings_);
    }

    template <FixedDimension TIndex>
    constexpr auto &at(const TIndex &) noexcept {
        return std::get<TIndex::value>(paddings_);
    }

    constexpr std::array<std::array<dim_t, 2>, rank()>
    to_array() const noexcept {
        auto at_impl = [this]<size_t... I>(std::index_sequence<I...>) {
            (void)this;
            return std::array<std::array<dim_t, 2>, rank()>{
                at(fixed_dim_v<I>).to_array()...};
        };
        return at_impl(std::make_index_sequence<rank()>());
    }

  private:
    [[no_unique_address]] std::tuple<TPaddings...> paddings_;
};

template <class T>
concept Paddings = requires {
    typename T::paddings_type;
    T::rank();
    T::fixed_rank();
    T::dynamic_rank();
};

template <class T>
concept FixedPaddings = Dimensions<T> && T::is_fixed();

namespace detail {
template <class I> struct dynamic_paddings_type_impl;

template <size_t... I>
struct dynamic_paddings_type_impl<std::index_sequence<I...>> {
    template <std::size_t> using elem_type = dim_t;

    using type = paddings_t<elem_type<I>...>;
};

template <size_t Rank>
using dynamic_paddings_t =
    typename dynamic_paddings_type_impl<std::make_index_sequence<Rank>>::type;
} // namespace detail

template <Padding... TPaddings>
constexpr auto make_padding_impl(const TPaddings &...paddings) noexcept {
    return paddings_t<TPaddings...>{paddings...};
}

template <size_t Rank, Padding TPadding>
constexpr auto make_repeat_paddings_impl(const TPadding &padding) noexcept {
    auto repeat_impl = [padding]<size_t... I>(std::index_sequence<I...>) {
        return make_padding_impl(((void)I, padding)...);
    };
    return repeat_impl(std::make_index_sequence<Rank>());
}

template <size_t Rank> constexpr auto make_zeros_paddings_impl() noexcept {
    return make_repeat_paddings_impl<Rank>(padding_zero);
}
} // namespace nncase::ntt
