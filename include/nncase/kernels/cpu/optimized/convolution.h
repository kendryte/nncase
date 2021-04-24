/* Copyright 2020 Canaan Inc.
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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

#define BEGIN_NS_NNCASE_KERNELS_CPU_OPT \
    namespace nncase                    \
    {                                   \
    namespace kernels                   \
    {                                   \
        namespace cpu                   \
        {                               \
            namespace optimized         \
            {

#define END_NS_NNCASE_KERNELS_CPU_OPT \
    }                                 \
    }                                 \
    }                                 \
    }

BEGIN_NS_NNCASE_KERNELS_CPU_OPT

NNCASE_API result<void> conv2d(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation) noexcept;

NNCASE_API result<void> conv3x3s1_sse(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept;

END_NS_NNCASE_KERNELS_CPU_OPT