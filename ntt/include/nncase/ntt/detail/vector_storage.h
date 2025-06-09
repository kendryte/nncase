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
#include "../dimension.h"
#include "../shape.h"
#include "../tensor_traits.h"

namespace nncase::ntt {
template <Scalar T, FixedShape Lanes> class basic_vector;
template <Scalar T, FixedShape Lanes> struct vector_storage_traits;

namespace detail {
template <class TTraits, class TIndex> class vector_storage_element_proxy {
  public:
    using buffer_type = typename TTraits::buffer_type;
    using element_type = typename TTraits::element_type;

    constexpr vector_storage_element_proxy(buffer_type &buffer,
                                           TIndex index) noexcept
        : buffer_(buffer), index_(index) {}

    constexpr operator element_type() const noexcept {
        return TTraits::get_element(buffer_, index_);
    }

    constexpr vector_storage_element_proxy &
    operator=(element_type value) noexcept {
        TTraits::set_element(buffer_, index_, value);
        return *this;
    }

    constexpr vector_storage_element_proxy &
    operator=(const vector_storage_element_proxy &value) noexcept {
        return operator=((element_type)value);
    }

  private:
    buffer_type &buffer_;
    TIndex index_;
};
} // namespace detail

template <class T, FixedDimension Lane>
struct vector_storage_traits<T, shape_t<Lane>> {
    using buffer_type = std::array<T, Lane::value>;
    using element_type = T;

    template <Dimensions TIndex>
    static constexpr T &element_at(buffer_type &array,
                                   const TIndex &index) noexcept {
        static_assert(TIndex::rank() == 1, "Index rank must be 1");
        return array[index[0]];
    }

    template <Dimensions TIndex>
    static constexpr const T &element_at(const buffer_type &array,
                                         const TIndex &index) noexcept {
        static_assert(TIndex::rank() == 1, "Index rank must be 1");
        return array[index[0]];
    }
};

template <class T, FixedDimension OuterLane, FixedDimension... InnerLanes>
struct vector_storage_traits<T, shape_t<OuterLane, InnerLanes...>> {
    using buffer_type =
        std::array<basic_vector<T, shape_t<InnerLanes...>>, OuterLane::value>;
    using element_type = T;

    template <Dimensions TIndex>
    static constexpr decltype(auto) element_at(buffer_type &vec,
                                               const TIndex &index) noexcept {
        auto &inner_vector = vec[index[0]];
        const auto remaining_index = index.template slice<1>();
        if constexpr (TIndex::rank() > 1) {
            return inner_vector(remaining_index);
        } else {
            return inner_vector;
        }
    }

    template <Dimensions TIndex>
    static constexpr decltype(auto) element_at(const buffer_type &vec,
                                               const TIndex &index) noexcept {
        auto &inner_vector = vec[index[0]];
        const auto remaining_index = index.template slice<1>();
        if constexpr (TIndex::rank() > 1) {
            return inner_vector(remaining_index);
        } else {
            return inner_vector;
        }
    }
};
} // namespace nncase::ntt
