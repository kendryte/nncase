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
#include "../../loop.h"
#include "../../tensor_traits.h"
#include "../../utility.h"
#include <tuple>

namespace nncase::ntt::detail {
template <bool Broadcastable, class TDerived, Tensor TOutput, Tensor... TInputs>
class elementwise_impl {
  public:
    template <class... TArgs>
    constexpr void operator()(const TInputs &...inputs, TOutput &output,
                              TArgs &&...args) {
        const auto conti_dims =
            ntt::min(contiguous_dims(inputs.shape(), inputs.strides())...,
                     contiguous_dims(output.shape(), output.strides()));
        apply_wrapper(conti_dims, inputs...,
                      std::make_tuple(inputs.elements().data()...), output,
                      output.elements().data(), std::forward<TArgs>(args)...);
    }

  private:
    template <Dimension TContiguousDims, class TInputsP, class TOutP,
              class... TArgs>
    constexpr void apply_wrapper(const TContiguousDims &conti_dims,
                                 const TInputs &...inputs, TInputsP inputs_p,
                                 TOutput &output, TOutP out_p,
                                 TArgs &&...args) {
        apply<0>(conti_dims, inputs..., inputs_p, output, out_p,
                 std::forward<TArgs>(args)...);
    }

    template <size_t Axis, Dimension TContiguousDims, class TInputsP,
              class TOutP, class... TArgs>
    constexpr void apply(const TContiguousDims &conti_dims,
                         const TInputs &...inputs, TInputsP &inputs_p,
                         TOutput &output, TOutP &out_p, TArgs &&...args) {
        if (Axis + conti_dims >= TOutput::rank()) {
            // 1. In contiguous axes
            constexpr auto rest_rank = TOutput::rank() - fixed_dim_v<Axis>;
            constexpr auto min_rank = ntt::min(inputs.rank()..., output.rank());
            if constexpr (min_rank >= rest_rank) {
                // If all inputs and output have enough rank, we can apply
                // contiguous directly.
                auto apply_contiguous =
                    [&]<size_t... I>(std::index_sequence<I...>) {
                        return derived().template apply_contiguous<Axis>(
                            std::get<I>(inputs_p)...,
                            inputs.shape()
                                .template slice<rest_rank,
                                                inputs.rank() - rest_rank>()...,
                            out_p, std::forward<TArgs>(args)...);
                    };
                apply_contiguous(
                    std::make_index_sequence<sizeof...(TInputs)>());
            }
        } else if constexpr (Axis + 1 < TOutput::rank()) {
            // 2. Out of contiguous axes
            for (size_t i = 0; i < output.shape()[fixed_dim_v<Axis>]; i++) {
                apply<Axis + 1>(conti_dims, inputs..., inputs_p, output, out_p,
                                std::forward<TArgs>(args)...);
                if constexpr (Broadcastable) {
                    loop<sizeof...(TInputs)>([&](auto i) {
                        const auto &input =
                            std::get<i>(std::forward_as_tuple(inputs...));
                        std::get<i>(inputs_p) +=
                            utility_detail::get_safe_stride<Axis>(
                                input, output.shape());
                    });
                } else {
                    loop<sizeof...(TInputs)>([&](auto i) {
                        const auto &input_strides =
                            std::get<i>(std::forward_as_tuple(inputs...))
                                .strides();
                        std::get<i>(inputs_p) +=
                            input_strides[fixed_dim_v<Axis>];
                    });
                }
                out_p += output.strides()[fixed_dim_v<Axis>];
            }
        }
    }

  protected:
    TDerived &derived() noexcept { return static_cast<TDerived &>(*this); }
};
} // namespace nncase::ntt::detail
