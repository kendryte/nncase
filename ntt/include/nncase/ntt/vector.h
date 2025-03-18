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
#include "tensor_traits.h"

namespace nncase::ntt {
template <class T, size_t... Lanes>
class basic_vector
    : public detail::tensor_size_impl<
          fixed_shape<Lanes...>, default_strides_t<fixed_shape<Lanes...>>> {
    using size_impl_type =
        detail::tensor_size_impl<fixed_shape<Lanes...>,
                                 default_strides_t<fixed_shape<Lanes...>>>;

  public:
    static constexpr bool IsVector = true;

    using element_type = T;
    using traits_type = vector_storage_traits<T, Lanes...>;
    using buffer_type = traits_type::buffer_type;
    using shape_type = fixed_shape<Lanes...>;
    using strides_type = default_strides_t<shape_type>;

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

    template <class... Indices>
    constexpr decltype(auto) operator()(Indices &&...index) const noexcept {
        return this->operator()(
            ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
    }

    template <class... Indices>
    constexpr decltype(auto) operator()(Indices &&...index) noexcept {
        return this->operator()(
            ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
    }

    template <size_t IndexRank>
    constexpr decltype(auto)
    operator()(ranked_shape<IndexRank> index) noexcept {
        if constexpr (requires { traits_type::element_at(buffer_, index); }) {
            return traits_type::element_at(buffer_, index);
        } else {
            return detail::vector_storage_element_proxy<
                traits_type, ranked_shape<IndexRank>>(buffer_, index);
        }
    }

    template <size_t IndexRank>
    constexpr decltype(auto)
    operator()(ranked_shape<IndexRank> index) const noexcept {
        if constexpr (requires { traits_type::element_at(buffer_, index); }) {
            return traits_type::element_at(buffer_, index);
        } else {
            return traits_type::get_element(buffer_, index);
        }
    }

  private:
    buffer_type buffer_;
};

template <class T, size_t... Lanes> using vector = basic_vector<T, Lanes...>;

template <class T, size_t... OldLanes, size_t... NewLanes>
struct fixed_tensor_alike_type<basic_vector<T, OldLanes...>, NewLanes...> {
    using type = vector<T, NewLanes...>;
};
} // namespace nncase::ntt
