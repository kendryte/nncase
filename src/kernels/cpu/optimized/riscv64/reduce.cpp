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

#include <iostream>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#if __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

result<void> reduce_mean_rvv(NNCASE_UNUSED float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &out_shape, NNCASE_UNUSED const runtime_shape_t &out_strides) noexcept;

template <>
result<void> optimized::reduce<float>(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    if (reduce_mean == op)
    {
        auto out_shape = kernels::detail::get_reduced_shape(in_shape, axis, keep_dims);
        return reduce_mean_rvv(init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides);
    }
    else
    {
        return cpu::reference::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims, context);
    }
}

template result<void> optimized::reduce<int32_t>(reduce_op_t op, int32_t init_value, const int32_t *input, int32_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template result<void> optimized::reduce<int64_t>(reduce_op_t op, int64_t init_value, const int64_t *input, int64_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template <typename T>
result<void> optimized::reduce(reduce_op_t op, T init_value, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    return cpu::reference::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims, context);
}

result<void> reduce_mean_rvv(NNCASE_UNUSED float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &out_shape, NNCASE_UNUSED const runtime_shape_t &out_strides) noexcept
{
    if (axis[0] == 0)
    {
        size_t out_n = 1;
        size_t inner_n = 1;
        for (size_t i = 0; i < axis.size(); ++i)
        {
            inner_n *= in_shape[i];
        }
        for (size_t i = 0; i < in_shape.size() - axis.size(); ++i)
        {
            out_n *= in_shape[in_shape.size() - i - 1];
        }
#if (!__riscv_vector)
        for (size_t i = 0; i < out_n; ++i)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < inner_n; ++j)
            {
                sum += input[j * out_n + i];
            }
            output[i] = sum / inner_n;
        }
#else
        size_t vl;
        float lr = 1.0f / inner_n;
        size_t i = 0;
        size_t n2 = out_n;
        while (n2)
        {
            vl = vsetvl_e32m8(n2);
            vfloat32m8_t _p = vle32_v_f32m8(input + 0 * out_n + i, vl);

            for (size_t j = 1; j < inner_n; ++j)
            {
                vfloat32m8_t _p1 = vle32_v_f32m8(input + j * out_n + i, vl);
                _p = vfadd_vv_f32m8(_p, _p1, vl);
            }
            _p = vfmul_vf_f32m8(_p, lr, vl);
            vse32_v_f32m8(output + i, _p, vl);
            i += vl;
            n2 -= vl;
        }
#endif
    }
    else
    {
        size_t out_n = 1;
        size_t inner_n = 1;
        for (size_t i = 0; i < axis.size(); ++i)
        {
            out_n *= in_shape[i];
        }
        for (size_t i = 0; i < in_shape.size() - axis.size(); ++i)
        {
            inner_n *= in_shape[in_shape.size() - i - 1];
        }
        for (size_t i = 0; i < out_n; ++i)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < inner_n; ++j)
            {
                sum += input[i * inner_n + j];
            }
            output[i] = sum / inner_n;
        }
    }
    return ok();
}