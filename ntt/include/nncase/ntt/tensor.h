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
#include "detail/tensor_storage.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"
#include "tensor_traits.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
template <class T, Shape TShape, Strides TStrides, bool IsView>
class basic_tensor;

template <class T, Shape TShape, Strides TStrides, bool IsView, class U>
struct replace_element_type<basic_tensor<T, TShape, TStrides, IsView>, U> {
    using type = basic_tensor<U, TShape, TStrides, IsView>;
};

template <class T, Shape TShape, Strides TStrides>
using tensor = basic_tensor<T, TShape, TStrides, false>;

template <class T, Shape TShape, Strides TStrides>
using tensor_view = basic_tensor<T, TShape, TStrides, true>;

namespace detail {
template <class T, class TBuffer> struct tensor_element_type_from_buffer {
    using type = T;
};

template <class TBuffer> struct tensor_element_type_from_buffer<void, TBuffer> {
    using type = typename TBuffer::element_type;
};

template <class T, size_t N>
struct tensor_element_type_from_buffer<void, T[N]> {
    using type = T;
};

template <class T, class TBuffer>
using tensor_element_type_from_buffer_t =
    typename tensor_element_type_from_buffer<
        T, std::remove_reference_t<TBuffer>>::type;
} // namespace detail

template <class T, Shape TShape, Strides TStrides>
constexpr auto make_span(T *data, const TShape &shape,
                         const TStrides &strides) noexcept {
    if constexpr (FixedShape<TShape> && FixedStrides<TStrides>) {
        constexpr size_t size = linear_size(TShape{}, TStrides{});
        return std::span<T, size>(data, size);
    } else {
        return std::span<T>(data, linear_size(shape, strides));
    }
}

template <class T, Shape TShape, Strides TStrides>
constexpr auto make_tensor(const TShape &shape, const TStrides &strides) {
    return tensor<T, TShape, TStrides>(shape, strides);
}

template <class T, Shape TShape>
constexpr auto make_tensor(const TShape &shape) {
    return make_tensor<T>(shape, default_strides(shape));
}

template <class T, size_t... Lanes> constexpr auto make_fixed_tensor() {
    return make_tensor<T>(make_shape(fixed_dim_v<Lanes>...));
}

template <class T, Shape TShape, Strides TStrides>
constexpr auto make_unique_tensor(const TShape &shape,
                                  const TStrides &strides) {
    return std::make_unique<tensor<T, TShape, TStrides>>(shape, strides);
}

template <class T, Shape TShape>
constexpr auto make_unique_tensor(const TShape &shape) {
    return make_unique_tensor<T>(shape, default_strides(shape));
}

template <class T, size_t... Lanes> constexpr auto make_unique_fixed_tensor() {
    return make_unique_tensor<T>(make_shape(fixed_dim_v<Lanes>...));
}

template <class T = void, class TBuffer, Shape TShape, Strides TStrides>
constexpr auto make_tensor_view(TBuffer &&buffer, const TShape &shape,
                                const TStrides &strides) {
    using element_type = detail::tensor_element_type_from_buffer_t<T, TBuffer>;
    return tensor_view<element_type, TShape, TStrides>(
        std::forward<TBuffer>(buffer), shape, strides);
}

template <class T = void, class TBuffer, Shape TShape>
constexpr auto make_tensor_view(TBuffer &&buffer, const TShape &shape) {
    return make_tensor_view<T>(std::forward<TBuffer>(buffer), shape,
                               default_strides(shape));
}

template <class T, Shape TShape, Strides TStrides>
constexpr auto make_tensor_view_from_address(T *address, const TShape &shape,
                                             const TStrides &strides) {
    return make_tensor_view<T>(make_span(address, shape, strides), shape,
                               strides);
}

template <class T, Shape TShape>
constexpr auto make_tensor_view_from_address(T *address, const TShape &shape) {
    return make_tensor_view_from_address<T>(address, shape,
                                            default_strides(shape));
}

template <class T, Shape TShape, Strides TStrides, bool IsView>
class basic_tensor
    : public detail::tensor_size_impl<TShape, TStrides>,
      public detail::tensor_storage<T, max_size_v<TShape, TStrides>, IsView> {
    using size_impl_type = detail::tensor_size_impl<TShape, TStrides>;

  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    using storage_type =
        detail::tensor_storage<T, max_size_v<TShape, TStrides>, IsView>;
    using buffer_type = typename storage_type::buffer_type;
    using shape_type = TShape;
    using strides_type = TStrides;

    using size_impl_type::rank;
    using size_impl_type::shape;
    using size_impl_type::size;
    using size_impl_type::strides;
    using storage_type::buffer;
    using storage_type::elements;

    template <bool IsViewV = IsView, class = std::enable_if_t<!IsViewV>>
    constexpr basic_tensor(TShape shape, TStrides strides) noexcept
        : size_impl_type(std::move(shape), std::move(strides)),
          storage_type(shape.length()) {}

    constexpr basic_tensor(buffer_type buffer, TShape shape,
                           TStrides strides) noexcept
        : size_impl_type(std::move(shape), std::move(strides)),
          storage_type(std::in_place, std::move(buffer)) {}

    class const_iterator {
      public:
        const_iterator(const basic_tensor &tensor,
                       dynamic_shape_t<shape_type::rank()> index) noexcept
            : tensor_(tensor), index_(index) {}

        const_iterator &operator++(int) noexcept {
            index_.last() += 1;
            for (size_t i = index_.rank() - 1; i > 0; i--) {
                if (index_[i] >= tensor_.shape()[i]) {
                    index_[i - 1]++;
                    index_[i] = 0;
                }
            }
            return *this;
        }

        const_iterator operator++() noexcept {
            auto old = *this;
            operator++(0);
            return old;
        }

        T &operator*() noexcept { return tensor_(index_); }

        bool operator==(const const_iterator &other) const noexcept {
            return &tensor_ == &other.tensor_ && index_ == other.index_;
        }

      private:
        const basic_tensor &tensor_;
        dynamic_shape_t<shape_type::rank()> index_;
    };

    static basic_tensor<T, TShape, TStrides, IsView>
    from_scalar(T value) noexcept;

    operator const buffer_type &() const noexcept { return buffer(); }
    operator buffer_type &() noexcept { return buffer(); }

    const_iterator begin() const noexcept {
        return const_iterator(*this, dynamic_shape_t<shape_type::rank()>{});
    }

    const_iterator end() const noexcept {
        return const_iterator(*this,
                              dynamic_shape_t<shape_type::rank()>{shape()[0]});
    }

    template <Shape UShape>
    constexpr auto broadcast_to(const UShape &new_shape) {
        static_assert(UShape::rank() >= TShape::rank(),
                      "Broadcast shape must have rank greater than or equal to "
                      "the tensor shape");
        auto new_strides =
            broadcast_strides<UShape::rank() - rank()>(strides());
        return make_tensor_view<T>(elements(), new_shape, new_strides);
    }

    template <Shape UShape>
    constexpr auto broadcast_to(const UShape &new_shape) const {
        static_assert(UShape::rank() >= TShape::rank(),
                      "Broadcast shape must have rank greater than or equal to "
                      "the tensor shape");
        auto new_strides =
            broadcast_strides<UShape::rank() - rank()>(strides());
        return make_tensor_view<const T>(elements(), new_shape, new_strides);
    }

    template <Dimensions Index, Shape UShape>
    constexpr auto view(const Index &index, const UShape &shape) noexcept {
        auto offset = linear_offset(index, strides());
        auto begin = elements().data() + offset;
        return make_tensor_view_from_address<T>(
            begin, shape, canonicalize_strides(shape, strides()));
    }

    template <Dimensions Index, Shape UShape>
    constexpr auto view(const Index &index,
                        const UShape &shape) const noexcept {
        auto offset = linear_offset(index, strides());
        auto begin = elements().data() + offset;
        return make_tensor_view_from_address<const T>(
            begin, shape, canonicalize_strides(shape, strides()));
    }

    template <Dimensions Index>
    constexpr auto view(const Index &index) noexcept {
        auto left_shape = make_ones_shape<Index::rank()>();
        auto right_shape = shape().template slice<Index::rank()>();
        auto new_shape = left_shape.concat(right_shape);
        auto t_view = view(index, new_shape);
        return t_view.squeeze(make_index_shape<Index::rank()>());
    }

    template <Dimensions Index>
    constexpr auto view(const Index &index) const noexcept {
        auto left_shape = make_ones_shape<Index::rank()>();
        auto right_shape = shape().template slice<Index::rank()>();
        auto new_shape = left_shape.concat(right_shape);
        auto t_view = view(index, new_shape);
        return t_view.squeeze(make_index_shape<Index::rank()>());
    }

    constexpr tensor_view<T, TShape, TStrides> view() noexcept {
        return view(make_zeros_shape<TShape::rank()>(), shape());
    }

    template <Shape TNewShape> constexpr auto reshape(const TNewShape &shape) {
        return make_tensor_view<T>(buffer(), shape, default_strides(shape));
    }

    template <FixedShape TAxes> constexpr auto squeeze(const TAxes &axes) {
        auto new_shape = squeeze_dims(shape(), axes);
        auto new_strides = squeeze_dims(strides(), axes);
        return make_tensor_view<T>(buffer(), new_shape, new_strides);
    }

    template <Dimensions TIndex>
    constexpr const T &operator()(const TIndex &index) const noexcept {
        return elements()[linear_offset(index, strides())];
    }

    template <Dimensions TIndex>
    constexpr T &operator()(const TIndex &index) noexcept {
        return elements()[linear_offset(index, strides())];
    }

    template <Dimension... Indices>
    constexpr const T &operator()(const Indices &...index) const noexcept {
        return this->operator()(make_shape(index...));
    }

    template <Dimension... Indices>
    constexpr T &operator()(const Indices &...index) noexcept {
        return this->operator()(make_shape(index...));
    }
};

static_assert(sizeof(tensor_view<float, shape_t<fixed_dim<1>>,
                                 strides_t<fixed_dim<1>>>) ==
                  sizeof(std::span<float, 1>),
              "fixed tensor_view size should be same as std::span");
} // namespace nncase::ntt
