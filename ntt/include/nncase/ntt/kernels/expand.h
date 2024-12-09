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
#include "../utility.h"

namespace nncase::ntt {

namespace expand_detail {

template <IsFixedTensor TIn, IsFixedTensor TOut>
void expand_impl(const TIn &input, TOut &&output) noexcept {
    constexpr auto in_rank = TIn::shape_type::rank();
    constexpr auto in_shape = typename TIn::shape_type{};
    constexpr auto out_shape = std::decay_t<TOut>::shape();

    using TIElem = typename TIn::element_type;
    using TOElem = typename std::decay_t<TOut>::element_type;

    auto conti_dims_input = contiguous_dims(input.shape(), input.strides());
    auto conti_dims_output = contiguous_dims(output.shape(), output.strides());

    auto expand_dims_cnt = 0;
    auto before_expand_dims = 1;
    auto after_expand_dims = 1;
    auto expand_dim = 0;
    for (size_t i = 0; i < in_rank; i++) {
        if (in_shape[i] != out_shape[i]) {
            expand_dims_cnt++;
            expand_dim = out_shape[i];
        }
        if (expand_dims_cnt == 0) {
            before_expand_dims *= in_shape[i];
        } else {
            after_expand_dims *= in_shape[i];
        }
    }

    auto in_ptr = reinterpret_cast<const TIElem *>(input.elements().data());
    auto out_ptr = reinterpret_cast<TOElem *>(output.elements().data());
    auto pattern_size = after_expand_dims * sizeof(TIElem);
    if (conti_dims_input == in_rank && conti_dims_output == in_rank &&
        expand_dims_cnt == 1 && after_expand_dims > 256) {
        for (size_t i = 0; i < before_expand_dims; i++) {
            for (size_t j = 0; j < expand_dim; j++) {
                std::memcpy(out_ptr, in_ptr, pattern_size);
                out_ptr += after_expand_dims;
            }
            in_ptr += after_expand_dims;
        }
    } else {
        apply(out_shape, [&](auto index) {
            const auto in_index = get_reduced_offset<in_rank>(index, in_shape);
            output(index) = input(in_index);
        });
    }
}
} // namespace expand_detail

template <typename TIn, typename TOut>
void expand(const TIn &input, TOut &&output) noexcept {
    expand_detail::expand_impl(input, output);
}
} // namespace nncase::ntt
