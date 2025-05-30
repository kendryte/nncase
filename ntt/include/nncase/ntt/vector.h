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
#include "nncase/ntt/shape.h"
#include "tensor_traits.h"

namespace nncase::ntt {
template <Scalar T, size_t... Lanes>
class basic_vector
    : public detail::tensor_size_impl<shape_t<fixed_dim<Lanes>...>,
                                      decltype(default_strides(
                                          make_shape(fixed_dim_v<Lanes>...)))> {
    using size_impl_type =
        detail::tensor_size_impl<shape_t<fixed_dim<Lanes>...>,
                                 decltype(default_strides(
                                     make_shape(fixed_dim_v<Lanes>...)))>;

  public:
    static constexpr bool IsVector = true;

    using element_type = T;
    using traits_type = vector_storage_traits<T, Lanes...>;
    using buffer_type = traits_type::buffer_type;
    using shape_type = shape_t<fixed_dim<Lanes>...>;
    using strides_type =
        decltype(default_strides(make_shape(fixed_dim_v<Lanes>...)));

    using size_impl_type::rank;
    using size_impl_type::shape;
    using size_impl_type::size;
    using size_impl_type::strides;

    static basic_vector<T, Lanes...> from_scalar(T value) noexcept;

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

    template <size_t Index> static constexpr size_t lane() noexcept {
        static_assert(Index < sizeof...(Lanes),
                      "Dimension index out of bounds");
        return std::get<Index>(std::make_tuple(Lanes...));
    }

  private:
    buffer_type buffer_;
    static constexpr std::array<size_t, sizeof...(Lanes)> lanes = {Lanes...};
};

template <Scalar T, size_t... Lanes> using vector = basic_vector<T, Lanes...>;
} // namespace nncase::ntt
