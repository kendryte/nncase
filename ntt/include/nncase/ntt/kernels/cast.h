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
#include "../post_ops.h"
#include "../tensor_ops.h"
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"
#include <cassert>
#include <stdio.h>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut, template <class> class TPostOp>
class cast_impl {
    inline static constexpr size_t rank = TIn::rank();
    // !! For vector<bool>, the element counts must be same as the other cast
    // oprand.
    using InElemType = element_or_scalar_t<TIn>;
    using OutElemType = element_or_scalar_t<TOut>;
    static_assert((Vector<InElemType> && Vector<OutElemType>) ||
                      (Scalar<InElemType> && Scalar<OutElemType>),
                  "input & output must have the same type.");
    inline static constexpr auto in_ele_size =
        sizeof(std::conditional_t<Vector<InElemType>, // if vector
                                  element_or_scalar_t<InElemType>, size_t>);
    inline static constexpr auto out_ele_size =
        sizeof(std::conditional_t<Vector<OutElemType>,
                                  element_or_scalar_t<OutElemType>, size_t>);

    inline static constexpr bool is_bool_vector =
        Vector<InElemType> &&
        (std::is_same_v<element_or_scalar_t<InElemType>, bool> ||
         std::is_same_v<element_or_scalar_t<OutElemType>, bool>);

    inline static constexpr float scale =
        is_bool_vector ? 1.0f : (float)in_ele_size / out_ele_size;

    inline static constexpr auto in_offset_scale = scale > 1.0f ? (size_t)scale
                                                                : (size_t)1;
    inline static constexpr auto
        out_offset_scale = scale > 1.0f ? (size_t)1 : (size_t)(1.0f / scale);

  public:
    constexpr void operator()(const TIn &input, TOut &output) noexcept {
        constexpr auto rank = TIn::rank();
        auto input_conti_dims = contiguous_dims(input.shape(), input.strides());
        auto output_conti_dims =
            contiguous_dims(output.shape(), output.strides());
        auto conti_dims = std::min(input_conti_dims, output_conti_dims);
        auto outer_shape = generate_shape<rank>([&](auto i) {
            if (i > rank - conti_dims - 1)
                return (dim_t)1;
            else
                return (dim_t)output.shape()[i];
        });

        auto inner_shape = generate_shape<rank>([&](auto i) {
            if (i > rank - conti_dims - 1)
                return (dim_t)output.shape()[i];
            else
                return (dim_t)1_dim;
        });

        auto len = inner_shape.length();
        ntt::apply(outer_shape, [&](auto index) {
            auto in_ptr = &input(index);
            auto out_ptr = &output(index);
            ntt::u_cast<TPostOp>(in_ptr, 1, out_ptr, 1, len);
        });
    }
};
} // namespace detail

template <template <class> class TPostOp = DefaultPostOp, Tensor TIn, Tensor TOut>
void cast(const TIn &input, TOut &&output) noexcept {
    detail::cast_impl<TIn, std::decay_t<TOut>, TPostOp> impl;
    impl(input, output);
}
} // namespace nncase::ntt
