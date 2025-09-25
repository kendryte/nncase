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
#include "../primitive_ops.h"
#include "nncase/ntt/tensor_traits.h"

namespace nncase::ntt {
namespace ukernels {

template <class TIn, bool Arch> struct u_layer_norm_policy {
    static constexpr size_t unroll = 2;
};

template <bool UseMean, class TEIn, class TEScale, class TEBias, class TEOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis, bool Arch>
struct u_layer_norm {
  public:
    constexpr void operator()(const TEIn *NTT_RESTRICT input,
                              const TEScale *NTT_RESTRICT scale,
                              const TEBias *NTT_RESTRICT bias,
                              TEOut *NTT_RESTRICT output, const float &epsilon,
                              const VectorizedAxes &, const PadedNums &,
                              const TAxis &, const size_t &inner_size,
                              const float &norm_factor, size_t stride_in,
                              size_t stride_out) noexcept {

        using policy_t = u_layer_norm_policy<TEIn, Arch>;
        constexpr auto unroll = policy_t::unroll;

        using TElemScalar = element_or_scalar_t<TEIn>;
        using TAccElem = decltype(ntt::cast_elem<float>(std::declval<TEIn>()));
        auto mean = (TElemScalar)0;
        auto extended_sum = (TAccElem)0;

        size_t count = inner_size;
        auto inner_ptr_in = input;
        auto inner_ptr_out = output;
        if constexpr (UseMean) {
            auto extended_mean = (TAccElem)0;
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    extended_mean += *(inner_ptr_in);
                    inner_ptr_in += stride_in;
                    count--;
                }
            }
            for (size_t i = 0; i < count; i++) {
                extended_mean += *(inner_ptr_in);
                inner_ptr_in += stride_in;
            }
            extended_mean *= norm_factor;
            auto extended_mean_s = reduce_sum(extended_mean);

            count = inner_size;
            inner_ptr_in = input;
            while (count / unroll) {
                for (auto i = 0; i < unroll; i++) {
                    const auto val =
                        ntt::square(ntt::cast_elem<float>(*(inner_ptr_in)) -
                                    extended_mean_s);
                    inner_ptr_in += stride_in;
                    extended_sum += val;
                    count--;
                }
            }

            for (auto i = 0; i < count; i++) {
                const auto val = ntt::square(
                    ntt::cast_elem<float>(*(inner_ptr_in)) - extended_mean_s);
                inner_ptr_in += stride_in;
                extended_sum += val;
            }
            mean = (TElemScalar)extended_mean_s;
        } else {
            count = inner_size;
            inner_ptr_in = input;

            while (count / unroll) {
                for (auto i = 0; i < unroll; i++) {
                    const auto input_val = *(inner_ptr_in);
                    inner_ptr_in += stride_in;
                    extended_sum =
                        ntt::mul_add(input_val, input_val, extended_sum);
                    count--;
                }
            }

            for (auto i = 0; i < count; i++) {
                const auto input_val = *(inner_ptr_in);
                inner_ptr_in += stride_in;
                extended_sum = ntt::mul_add(input_val, input_val, extended_sum);
            }
        }

        const auto extended_sum_s = reduce_sum(extended_sum) * norm_factor;
        auto extended_add = extended_sum_s + epsilon;
        auto rsqrt = ntt::cast_elem<TElemScalar>(ntt::rsqrt(extended_add));

        if constexpr (UseMean) {
            count = inner_size;
            inner_ptr_in = input;
            inner_ptr_out = output;

            while (count / unroll) {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = (*(inner_ptr_in)-mean) * rsqrt;
                    *(inner_ptr_out) = ntt::mul_add(val, scale[i], bias[i]);
                    inner_ptr_in += stride_in;
                    inner_ptr_out += stride_out;
                    count--;
                }
            }

            for (auto i = 0; i < count; i++) {
                auto val = (*(inner_ptr_in)-mean) * rsqrt;
                *(inner_ptr_out) = ntt::mul_add(val, scale[i], bias[i]);
                inner_ptr_in += stride_in;
                inner_ptr_out += stride_out;
            }
        } else {
            while (count / unroll) {
                for (auto i = 0; i < inner_size; i++) {
                    auto val = *(inner_ptr_in)*rsqrt;
                    *(inner_ptr_out) =
                        val * scale[i]; // RMSNorm doesn't need bias
                    inner_ptr_in += stride_in;
                    inner_ptr_out += stride_out;
                    count--;
                }
            }

            for (auto i = 0; i < count; i++) {
                auto val = *(inner_ptr_in)*rsqrt;
                *(inner_ptr_out) = val * scale[i]; // RMSNorm doesn't need bias
                inner_ptr_in += stride_in;
                inner_ptr_out += stride_out;
            }
        }
    }
};
} // namespace ukernels

template <bool UseMean, class TEIn, class TEScale, class TEBias, class TEOut,
          FixedDimensions VectorizedAxes, Dimensions PadedNums,
          FixedDimension TAxis>
constexpr void u_layer_norm(const TEIn *input, const TEScale *scale,
                            const TEBias *bias, TEOut *output,
                            const float &epsilon,
                            const VectorizedAxes &vectorizedAxes,
                            const PadedNums &padedNums, const TAxis &axis,
                            const size_t &inner_size, const float &norm_factor,
                            size_t stride_in, size_t stride_out) noexcept {
    ukernels::u_layer_norm<UseMean, TEIn, TEScale, TEBias, TEOut,
                           VectorizedAxes, PadedNums, TAxis, true>
        impl;
    impl(input, scale, bias, output, epsilon, vectorizedAxes, padedNums, axis,
         inner_size, norm_factor, stride_in, stride_out);
}
} // namespace nncase::ntt
