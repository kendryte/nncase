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
#include "detail/shape_storage.h"
#include "detail/vector_storage.h"
#include "nncase/ntt/dimension.h"
#include "tensor_traits.h"
#include <type_traits>

namespace nncase::ntt {
template <Scalar T, FixedShape Lanes>
class basic_vector
    : public detail::tensor_size_impl<Lanes,
                                      decltype(default_strides(Lanes{}))> {
    using size_impl_type =
        detail::tensor_size_impl<Lanes, decltype(default_strides(Lanes{}))>;

  public:
    static constexpr bool IsVector = true;

    using element_type = T;
    using traits_type = vector_storage_traits<T, Lanes>;
    using buffer_type = traits_type::buffer_type;
    using shape_type = Lanes;
    using strides_type = decltype(default_strides(Lanes{}));

    using size_impl_type::rank;
    using size_impl_type::shape;
    using size_impl_type::size;
    using size_impl_type::strides;

    template <size_t Index> static constexpr auto lane() noexcept {
        static_assert(Index < Lanes::rank(), "Dimension index out of bounds");
        return Lanes{}.template at<Index>();
    }

    template <ScalarOrVector U>
    static basic_vector<T, Lanes> from_scalar(U value) noexcept;

    template <ScalarOrVector U>
    static basic_vector<T, Lanes> unaligned_load_from(const U *ptr) noexcept;

    constexpr basic_vector() noexcept = default;
    constexpr basic_vector(const buffer_type &buffer) noexcept
        : buffer_(std::move(buffer)) {}
    constexpr explicit basic_vector(element_type value) noexcept
        : basic_vector(from_scalar(value)) {}

    operator const buffer_type &() const noexcept { return buffer(); }
    operator buffer_type &() noexcept { return buffer(); }

    constexpr const buffer_type &buffer() const noexcept { return buffer_; }
    constexpr buffer_type &buffer() noexcept { return buffer_; }

    template <Dimension... Indices>
    constexpr decltype(auto)
    operator()(const Indices &...index) const noexcept {
        return this->operator()(make_shape(index...));
    }

    template <Dimension... Indices>
    constexpr decltype(auto) operator()(const Indices &...index) noexcept {
        return this->operator()(make_shape(index...));
    }

    template <Dimensions TIndex>
    constexpr decltype(auto) operator()(const TIndex &index) noexcept {
        if constexpr (requires { traits_type::element_at(buffer_, index); }) {
            return traits_type::element_at(buffer_, index);
        } else {
            return detail::vector_storage_element_proxy<traits_type, TIndex>(
                buffer_, index);
        }
    }

    template <Dimensions TIndex>
    constexpr decltype(auto) operator()(const TIndex &index) const noexcept {
        if constexpr (requires { traits_type::element_at(buffer_, index); }) {
            return traits_type::element_at(buffer_, index);
        } else {
            return traits_type::get_element(buffer_, index);
        }
    }

  private:
    buffer_type buffer_;
};

template <Scalar T, size_t... Lanes>
using vector = basic_vector<T, shape_t<fixed_dim<Lanes>...>>;

template <class T, Scalar U> struct replace_element_type;

template <Scalar T, Scalar U> struct replace_element_type<T, U> {
    using type = U;
};

template <Scalar T, FixedShape Lanes, Scalar U>
struct replace_element_type<basic_vector<T, Lanes>, U> {
    using type = basic_vector<U, Lanes>;
};

template <class T, Scalar U>
using replace_element_t =
    typename replace_element_type<std::decay_t<T>, U>::type;

template <Vector T, size_t... Lanes> struct replace_lanes_type {
    using type = vector<typename T::element_type, Lanes...>;
};

template <Vector T, size_t... Lanes>
using replace_lanes_t = typename replace_lanes_type<T, Lanes...>::type;

template <class T> struct vector_rank {
    static constexpr auto value = dim_zero;
};

template <Vector T> struct vector_rank<T> {
    static constexpr auto value = fixed_dim_v<T::rank()>;
};

template <class T> constexpr inline auto vector_rank_v = vector_rank<T>::value;
} // namespace nncase::ntt
