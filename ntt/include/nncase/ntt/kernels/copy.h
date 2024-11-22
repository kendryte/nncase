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

namespace nncase::ntt {
namespace detail {
template <typename TIn, typename TOut> struct copy_impl;

template <IsFixedTensor TIn, IsFixedTensor TOut> struct copy_impl<TIn, TOut> {
    constexpr void operator()(const TIn &input, TOut &output) {
        constexpr auto rank = TIn::rank();
        constexpr auto input_shape = TIn::shape();
        constexpr auto input_strides = TIn::strides();

        constexpr auto output_shape = std::decay_t<TOut>::shape();
        constexpr auto output_strides = std::decay_t<TOut>::strides();

        constexpr auto cdims_input =
            contiguous_dims(input_shape, input_strides);
        constexpr auto cdims_output =
            contiguous_dims(output_shape, output_strides);
        constexpr auto cdims = std::min(cdims_input, cdims_output);
        constexpr auto caxis = input_shape.rank() - cdims;

        ranked_shape<rank> index{};
        apply<0, rank, caxis>(index, input, output);

        // if constexpr (cdim_input == cdim_output &&
        //               cdim_input == input_shape.rank() &&
        //               cdim_output == output_shape.rank()) {
        //     auto out_buffer = output.buffer();
        //     memcpy(out_buffer.data(), input.buffer().data(),
        //            out_buffer.size_bytes());
        // } else {
        //     apply(input_shape,
        //           [&](auto index) { output(index) = input(index); });
        // }
    }

  private:
    template <size_t Axis, size_t Rank, size_t ContiguousAxis>
    constexpr void apply(ranked_shape<Rank> &index, const TIn &input,
                         TOut &output) {
        if constexpr (Axis == ContiguousAxis) {
            constexpr auto rest_dims =
                slice_fixed_dims<Rank - Axis, Axis>(TIn::shape());
            constexpr auto inner_size =
                rest_dims.length() * sizeof(typename TIn::element_type);
            auto input_p =
                input.elements().data() + linear_offset(index, input.strides());
            auto output_p = output.elements().data() +
                            linear_offset(index, output.strides());
            memcpy(output_p, input_p, inner_size);
        } else {
            apply_next<Axis, Rank, ContiguousAxis>(index, input, output);
        }
    }

    template <size_t Axis, size_t Rank, size_t ContiguousAxis>
    constexpr void apply_next(ranked_shape<Rank> &index, const TIn &input,
                              TOut &output) {
        for (index[Axis] = 0; index[Axis] < input.shape()[Axis];
             index[Axis]++) {
            apply<Axis + 1, Rank, ContiguousAxis>(index, input, output);
        }
    }
};

template <typename TIn, typename TOut> struct copy_impl;
template <IsRankedTensor TIn, IsRankedTensor TOut> struct copy_impl<TIn, TOut> {
    constexpr void operator()(const TIn &input, TOut &output) {
        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto output_shape = output.shape();
        auto output_strides = output.strides();
        auto cdims_input = contiguous_dims(input_shape, input_strides);
        auto cdims_output = contiguous_dims(output_shape, output_strides);
        auto cdims = std::min(cdims_input, cdims_output);

        auto domain = ntt::ranked_shape<TIn::rank()>();
        auto caxis = input_shape.rank() - cdims;
        for (size_t i = 0; i < caxis; i++) {
            domain[i] = input_shape[i];
        }
        for (size_t i = caxis; i < TIn::rank(); i++) {
            domain[i] = 1;
        }

        apply(domain, [&](auto index) {
            memcpy(output.buffer().data() +
                       linear_offset(index, output_strides),
                   input.buffer().data() + linear_offset(index, input_strides),
                   input_shape[caxis] * input_strides[caxis] *
                       sizeof(typename TIn::element_type));
        });
    }
};
} // namespace detail

template <class TIn, class TOut>
void tensor_copy(const TIn &input, TOut &&output) noexcept {
    detail::copy_impl<TIn, TOut> impl;
    impl(input, output);
}
} // namespace nncase::ntt
