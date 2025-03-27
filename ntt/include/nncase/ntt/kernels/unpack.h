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
#include "../apply.h"
#include "../loop.h"
#include "../shape_infer/unpack.h"
#include "../ukernels.h"

namespace nncase::ntt {
namespace detail {

template <class InShape, class InElemShape, class OutShape, class InStrides,
          class OutStrides, size_t... Axes>
class unpack_impl;

// fixed shape(1D)
template <size_t... InDims, size_t... InElemDims, class OutShape,
          size_t... InStrides, class OutStrides, size_t PackAxis>
class unpack_impl<fixed_shape<InDims...>, fixed_shape<InElemDims...>, OutShape,
                  fixed_strides<InStrides...>, OutStrides, PackAxis> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        using TVec = typename TIn::element_type;
        constexpr auto rank = TIn::shape_type::rank();
        constexpr auto in_conti_dims = contiguous_dims(
            fixed_shape<InDims...>{}, fixed_strides<InStrides...>{});
        if constexpr (in_conti_dims == rank) {
            auto pin = input.buffer().data();
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            if constexpr (PackAxis == (rank - 1))
                ntt::u_unary<ntt::ops::copy<TVec>, TVec>(
                    pin, 1, reinterpret_cast<TVec *>(pout), 1, count);
            else {
                constexpr auto in_strides =
                    std::array<size_t, sizeof...(InStrides)>{InStrides...};
                constexpr auto v_shape =
                    std::array<size_t, sizeof...(InElemDims)>{InElemDims...};
                ntt::u_unpack_1d_fixed<in_strides[PackAxis], v_shape[0], TIn,
                                       typename TOut::element_type, PackAxis>(
                    input, 1, pout, count);
            }
        } else {
            constexpr auto elem_rank = TVec::shape_type::rank();
            constexpr fixed_shape<InDims..., InElemDims...> domain{};
            apply(domain, [&](auto index) {
                auto in_index = slice_index<rank>(index);
                auto elem_index = slice_index<elem_rank>(index, rank);
                auto out_index = slice_index<rank>(index);
                out_index[PackAxis] =
                    out_index[PackAxis] * TVec::shape()[0] + index[rank];
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};

// fixed shape(2D)
template <size_t... InDims, size_t... InElemDims, class OutShape,
          size_t... InStrides, class OutStrides, size_t Axis1, size_t Axis2>
class unpack_impl<fixed_shape<InDims...>, fixed_shape<InElemDims...>, OutShape,
                  fixed_strides<InStrides...>, OutStrides, Axis1, Axis2> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        using TVec = typename TIn::element_type;
        constexpr auto rank = TIn::shape_type::rank();
        constexpr auto in_conti_dims = contiguous_dims(
            fixed_shape<InDims...>{}, fixed_strides<InStrides...>{});
        if constexpr ((in_conti_dims == rank) && (Axis2 == Axis1 + 1)) {
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            constexpr auto in_strides =
                std::array<size_t, sizeof...(InStrides)>{InStrides...};
            constexpr auto v_shape =
                std::array<size_t, sizeof...(InElemDims)>{InElemDims...};
            ntt::u_unpack_2d_fixed<in_strides[Axis1], v_shape[0],
                                   in_strides[Axis2], v_shape[1], TIn,
                                   typename TOut::element_type, Axis1, Axis2>(
                input, 1, pout, count);
        } else {
            constexpr auto elem_rank = TVec::shape_type::rank();
            constexpr fixed_shape<InDims..., InElemDims...> domain{};
            constexpr auto axes = std::array<size_t, 2>{Axis1, Axis2};
            apply(domain, [&](auto index) {
                auto in_index = slice_index<rank>(index);
                auto elem_index = slice_index<elem_rank>(index, rank);
                auto out_index = slice_index<rank>(index);
                loop<axes.size()>([&](auto i) {
                    out_index[axes[i]] =
                        out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
                });
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};

// fixed shape
template <size_t... InDims, size_t... InElemDims, class OutShape,
          size_t... InStrides, class OutStrides, size_t... Axes>
class unpack_impl<fixed_shape<InDims...>, fixed_shape<InElemDims...>, OutShape,
                  fixed_strides<InStrides...>, OutStrides, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        using TVec = typename TIn::element_type;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto rank = TIn::shape_type::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr fixed_shape<InDims..., InElemDims...> domain{};
        apply(domain, [&](auto index) {
            auto in_index = slice_index<rank>(index);
            auto elem_index = slice_index<elem_rank>(index, rank);
            auto out_index = slice_index<rank>(index);
            loop<axes.size()>([&](auto i) {
                out_index[axes[i]] =
                    out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
            });
            output(out_index) = input(in_index)(elem_index);
        });
    }
};

// ranked shape(1D)
template <size_t in_rank, size_t... InElemDims, class OutShape, class InStrides,
          class OutStrides, size_t PackAxis>
class unpack_impl<ranked_shape<in_rank>, fixed_shape<InElemDims...>, OutShape,
                  InStrides, OutStrides, PackAxis> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        using TVec = typename TIn::element_type;
        constexpr auto rank = in_rank;
        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto in_conti_dims = contiguous_dims(input_shape, input_strides);
        if (in_conti_dims == rank) {
            auto pin = input.buffer().data();
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            if constexpr (PackAxis == (rank - 1))
                ntt::u_unary<ntt::ops::copy<TVec>, TVec>(
                    pin, 1, reinterpret_cast<TVec *>(pout), 1, count);
            else
                ntt::u_unpack_1d_ranked<TVec::shape()[0], TVec,
                                        typename TOut::element_type>(
                    pin, 1, input.strides()[PackAxis], pout, count);
        } else {
            constexpr auto elem_rank = TVec::shape_type::rank();
            fixed_shape<InElemDims...> elem_shape{};
            constexpr auto domain_rank = in_rank + elem_rank;
            ranked_shape<domain_rank> domain{};
            for (size_t i = 0, j = 0; i < domain_rank; i++) {
                if (i < in_rank)
                    domain[i] = input_shape[i];
                else
                    domain[i] = elem_shape[j++];
            }

            apply(domain, [&](auto index) {
                auto in_index = slice_index<rank>(index);
                auto elem_index = slice_index<elem_rank>(index, rank);
                auto out_index = slice_index<rank>(index);
                out_index[PackAxis] =
                    out_index[PackAxis] * TVec::shape()[0] + index[rank];
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};

// ranked shape(2D)
template <size_t in_rank, size_t... InElemDims, class OutShape, class InStrides,
          class OutStrides, size_t Axis1, size_t Axis2>
class unpack_impl<ranked_shape<in_rank>, fixed_shape<InElemDims...>, OutShape,
                  InStrides, OutStrides, Axis1, Axis2> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        using TVec = typename TIn::element_type;
        constexpr auto rank = in_rank;
        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto in_conti_dims = contiguous_dims(input_shape, input_strides);
        if ((in_conti_dims == rank) && (Axis2 == Axis1 + 1) &&
            (Axis2 != (rank - 1))) {
            auto pout = output.buffer().data();
            auto count = input.shape().length();
            ntt::u_unpack_2d_ranked<TVec::shape()[0], TVec::shape()[1], TIn,
                                    typename TOut::element_type, Axis1, Axis2>(
                input, 1, input_strides[Axis1], input_strides[Axis2], pout,
                count);
        } else {
            constexpr auto axes = std::array<size_t, 2>{Axis1, Axis2};
            constexpr auto elem_rank = TVec::shape_type::rank();
            fixed_shape<InElemDims...> elem_shape{};
            constexpr auto domain_rank = in_rank + elem_rank;
            ranked_shape<domain_rank> domain{};
            for (size_t i = 0, j = 0; i < domain_rank; i++) {
                if (i < in_rank)
                    domain[i] = input_shape[i];
                else
                    domain[i] = elem_shape[j++];
            }

            apply(domain, [&](auto index) {
                auto in_index = slice_index<rank>(index);
                auto elem_index = slice_index<elem_rank>(index, rank);
                auto out_index = slice_index<rank>(index);
                loop<axes.size()>([&](auto i) {
                    out_index[axes[i]] =
                        out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
                });
                output(out_index) = input(in_index)(elem_index);
            });
        }
    }
};

// ranked shape
template <size_t in_rank, size_t... InElemDims, class OutShape, class InStrides,
          class OutStrides, size_t... Axes>
class unpack_impl<ranked_shape<in_rank>, fixed_shape<InElemDims...>, OutShape,
                  InStrides, OutStrides, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename TIn::element_type;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto rank = in_rank;
        constexpr auto elem_rank = TVec::shape_type::rank();

        auto input_shape = input.shape();
        fixed_shape<InElemDims...> elem_shape{};
        constexpr auto domain_rank = in_rank + elem_rank;
        ranked_shape<domain_rank> domain{};
        for (size_t i = 0, j = 0; i < domain_rank; i++) {
            if (i < in_rank)
                domain[i] = input_shape[i];
            else
                domain[i] = elem_shape[j++];
        }

        apply(domain, [&](auto index) {
            auto in_index = slice_index<rank>(index);
            auto elem_index = slice_index<elem_rank>(index, rank);
            auto out_index = slice_index<rank>(index);
            loop<axes.size()>([&](auto i) {
                out_index[axes[i]] =
                    out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
            });
            output(out_index) = input(in_index)(elem_index);
        });
    }
};
} // namespace detail

template <size_t... Axes, class TIn, class TOut>
void unpack(const TIn &input, TOut &&output) noexcept {
    detail::unpack_impl<
        typename TIn::shape_type, typename TIn::element_type::shape_type,
        typename std::decay_t<TOut>::shape_type, typename TIn::strides_type,
        typename std::decay_t<TOut>::strides_type, Axes...>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
