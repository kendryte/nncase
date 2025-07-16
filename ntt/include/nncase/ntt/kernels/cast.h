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
template <Tensor TIn, Tensor TOut, FixedDimensions PackedAxes> class cast_impl {
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
    constexpr void operator()(const TIn &input, TOut &output,
                              const PackedAxes &) noexcept {
#if 0        
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
#endif
        constexpr PackedAxes packedAxes;
        if constexpr (scale >= 1.f) {
            ntt::apply(output.shape(), [&](auto index) {
                auto in_index = index;
                if constexpr (packedAxes.rank() == 1)
                    in_index[fixed_dim_v<packedAxes.at(0)>] *= in_offset_scale;
                ntt::u_cast<in_offset_scale, out_offset_scale>(
                    &input(in_index), packedAxes.rank() == 1 ? input.strides()[packedAxes.at(0)] : 1, &output(index), 1, 1);
            });
        } else {
            ntt::apply(input.shape(), [&](auto index) {
                auto out_index = index;
                if constexpr (packedAxes.rank() == 1)
                    out_index[fixed_dim_v<packedAxes.at(0)>] *=
                        out_offset_scale;
                ntt::u_cast<in_offset_scale, out_offset_scale>(
                    &input(index), 1, &output(out_index), packedAxes.rank() == 1 ? output.strides()[packedAxes.at(0)] : 1, 1);
            });
        }
    }

  private:
#if 0    
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
#endif

    template <size_t in_offset_scale, size_t out_offset_scale, class T1,
              class T2>
    constexpr void cast_contiguous(const T1 *input, T2 *output, size_t extent) {
        ntt::u_cast<in_offset_scale, out_offset_scale>(input, 1, output, 1,
                                                       extent);
    }
};
} // namespace detail

template <Tensor TIn, class TOut, FixedDimensions PackedAxes = shape_t<>>
void cast(const TIn &input, TOut &&output,
          const PackedAxes &packedAxes = {}) noexcept {
    detail::cast_impl<TIn, std::decay_t<TOut>, PackedAxes> impl;
    impl(input, output, packedAxes);
}
} // namespace nncase::ntt
