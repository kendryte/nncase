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
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut> class cast_impl {
    inline static constexpr size_t rank = TIn::rank();

    // FIXME: vector<bool> of x86 may fail.
    using InElemType = element_or_scalar_t<TIn>;
    using OutElemType = element_or_scalar_t<TOut>;
    static_assert((Vector<InElemType> && Vector<OutElemType>) ||
                      (Scalar<InElemType> && Scalar<OutElemType>),
                  "input & output must have the same type.");
    inline static constexpr auto in_ele_size =
        sizeof(std::conditional_t<Vector<InElemType>,
                                  element_or_scalar_t<InElemType>, size_t>);
    inline static constexpr auto out_ele_size =
        sizeof(std::conditional_t<Vector<OutElemType>,
                                  element_or_scalar_t<OutElemType>, size_t>);
    inline static constexpr float scale = (float)in_ele_size / out_ele_size;

    inline static constexpr auto in_offset_scale = scale > 1.0f ? (size_t)scale
                                                                : (size_t)1;
    inline static constexpr auto
        out_offset_scale = scale > 1.0f ? (size_t)1 : (size_t)(1.0f / scale);

  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        if constexpr (scale != 1.0f) {
            static_assert(TIn::rank() == 1,
                          "Only support 1D tensor repack for now!");
        }

        dynamic_shape_t<rank> index{};
        const auto conti_dims =
            ntt::min(contiguous_dims(input.shape(), input.strides()),
                     contiguous_dims(output.shape(), output.strides()));

        if constexpr (scale >= 1.0f) {
            apply<0>(conti_dims, output.shape(), index, input, output);
        } else {
            apply<0>(conti_dims, input.shape(), index, input, output);
        }
    }

  private:
    template <size_t Axis, Dimension TContiguousDims, Shape TRestDims>
    constexpr void
    apply(const TContiguousDims &conti_dims, const TRestDims &rest_dims,
          dynamic_shape_t<rank> &index, const TIn &input, TOut &output) {
        if (conti_dims == rest_dims.rank()) {
            const auto inner_size = rest_dims.length();
            auto in_offset =
                linear_offset(index, input.strides()) * in_offset_scale;
            auto out_offset =
                linear_offset(index, output.strides()) * out_offset_scale;
            auto input_p = input.elements().data() + in_offset;
            auto output_p = output.elements().data() + out_offset;
            cast_contiguous(input_p, output_p, inner_size);
        } else if constexpr (Axis + 1 < rank) {
            for (index[fixed_dim_v<Axis>] = 0;
                 index[fixed_dim_v<Axis>] < rest_dims[dim_zero];
                 index[fixed_dim_v<Axis>]++) {
                apply<Axis + 1>(conti_dims, rest_dims.template slice<1>(),
                                index, input, output);
            }
        }
    }

    template <class T1, class T2>
    constexpr void cast_contiguous(const T1 *input, T2 *output, size_t extent) {
        ntt::u_cast<T1, T2, in_offset_scale, out_offset_scale>(input, 1, output,
                                                               1, extent);
    }
};
} // namespace detail

template <Tensor TIn, class TOut>
void cast(const TIn &input, TOut &&output) noexcept {
    detail::cast_impl<TIn, std::decay_t<TOut>> impl;
    impl(input, output);
}
} // namespace nncase::ntt
