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
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../ukernels.h"
#include "../utility.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TCond, class TX, class TY, class TOut> class where_impl {
  public:
    constexpr void operator()(const TCond &cond, const TX &x, const TY &y,
                              TOut &output) {
        constexpr auto out_shape = TOut::shape();

        apply(out_shape, [&](auto index) {
            const auto cond_index =
                shape_infer::reduced_index_by_shape(index, cond.shape());
            const auto x_index =
                shape_infer::reduced_index_by_shape(index, x.shape());
            const auto y_index =
                shape_infer::reduced_index_by_shape(index, y.shape());

            output(index) = cond(cond_index) ? x(x_index) : y(y_index);
        });
    }
};

template <IsFixedTensor TCond, IsFixedTensor TX, IsFixedTensor TY,
          IsFixedTensor TOut>
class where_impl<TCond, TX, TY, TOut> {
  public:
    constexpr void operator()(const TCond &cond, const TX &x, const TY &y,
                              TOut &output) {
        constexpr auto conti_dims =
            std::min({contiguous_dims(TCond::shape(), TCond::strides()),
                      contiguous_dims(TX::shape(), TX::strides()),
                      contiguous_dims(TY::shape(), TY::strides()),
                      contiguous_dims(TOut::shape(), TOut::strides())});
        auto cond_p = cond.elements().data();
        auto x_p = x.elements().data();
        auto y_p = y.elements().data();
        auto out_p = output.elements().data();
        apply<0, conti_dims>(cond, x, y, output, cond_p, x_p, y_p, out_p);
    }

  private:
    template <size_t Axis, size_t ContiguousDims, class TCondP, class TXP,
              class TYP, class TOutP>
    constexpr void apply(const TCond &cond, const TX &x, const TY &y,
                         TOut &output, TCondP cond_p, TXP x_p, TYP y_p,
                         TOutP out_p) {
        if constexpr (Axis + ContiguousDims >= TOut::rank()) {
            constexpr auto rest_rank = TOut::rank() - Axis;
            constexpr auto cond_rest_dims =
                slice_dims<rest_rank, TCond::rank() - rest_rank>(
                    TCond::shape());
            constexpr auto x_rest_dims =
                slice_dims<rest_rank, TX::rank() - rest_rank>(TX::shape());
            constexpr auto y_rest_dims =
                slice_dims<rest_rank, TY::rank() - rest_rank>(TY::shape());

            if constexpr (is_same_seq(x_rest_dims, y_rest_dims) &&
                          is_same_seq(cond_rest_dims, y_rest_dims)) {
                return where_non_broadcast(cond_p, x_p, y_p, out_p,
                                           cond_rest_dims.length());
            } else if constexpr (x_rest_dims.length() == 1 &&
                                 y_rest_dims.length() == 1) {
                return where_x_y_broadcast(cond_p, x_p, y_p, out_p,
                                           cond_rest_dims.length());
            } else if constexpr (x_rest_dims.length() == 1 &&
                                 cond_rest_dims.length() == 1 &&
                                 y_rest_dims.length() != 1) {
                return where_c_x_broadcast(cond_p, x_p, y_p, out_p,
                                           y_rest_dims.length());
            } else if constexpr (y_rest_dims.length() == 1 &&
                                 cond_rest_dims.length() == 1 &&
                                 x_rest_dims.length() != 1) {
                return where_c_y_broadcast(cond_p, x_p, y_p, out_p,
                                           x_rest_dims.length());
            } else if constexpr (x_rest_dims.length() == 1) {
                return where_x_broadcast(cond_p, x_p, y_p, out_p,
                                         y_rest_dims.length());
            } else if constexpr (y_rest_dims.length() == 1) {
                return where_y_broadcast(cond_p, x_p, y_p, out_p,
                                         x_rest_dims.length());
            } else if constexpr (cond_rest_dims.length() == 1) {
                return where_cond_broadcast(cond_p, x_p, y_p, out_p,
                                            x_rest_dims.length());
            }
        }

        if constexpr (Axis < TOut::shape().rank()) {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply<Axis + 1, ContiguousDims>(cond, x, y, output, cond_p, x_p,
                                                y_p, out_p);
                cond_p +=
                    utility_detail::get_safe_stride(cond, Axis, TOut::shape());
                x_p += utility_detail::get_safe_stride(x, Axis, TOut::shape());
                y_p += utility_detail::get_safe_stride(y, Axis, TOut::shape());
                out_p += output.strides()[Axis];
            }
        }
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_non_broadcast(const TCondElem *cond, const TXElem *x,
                             const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 1, x, 1, y, 1,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_x_broadcast(const TCondElem *cond, const TXElem *x,
                           const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 1, x, 0, y, 1,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_y_broadcast(const TCondElem *cond, const TXElem *x,
                           const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 1, x, 1, y, 0,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_cond_broadcast(const TCondElem *cond, const TXElem *x,
                              const TYElem *y, TOutElem *output,
                              size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 0, x, 1, y, 1,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_x_y_broadcast(const TCondElem *cond, const TXElem *x,
                             const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 1, x, 0, y, 0,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_c_x_broadcast(const TCondElem *cond, const TXElem *x,
                             const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 0, x, 0, y, 1,
                                                          output, 1, extent);
    }

    template <class TCondElem, class TXElem, class TYElem, class TOutElem>
    void where_c_y_broadcast(const TCondElem *cond, const TXElem *x,
                             const TYElem *y, TOutElem *output, size_t extent) {
        ntt::u_where<TCondElem, TXElem, TYElem, TOutElem>(cond, 0, x, 1, y, 0,
                                                          output, 1, extent);
    }
};
} // namespace detail

template <class TCond, class TX, class TY, class TOut>
void where(const TCond &cond, const TX &x, const TY &y, TOut &&output) {
    detail::where_impl<std::decay_t<TCond>, std::decay_t<TX>, std::decay_t<TY>,
                       std::decay_t<TOut>>()(cond, x, y, output);
}
} // namespace nncase::ntt