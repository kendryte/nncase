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
#include "utils.h"
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
#if __riscv_vector

result<void> optimized_sigmoid_impl(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{

    if (get_default_strides(in_shape) != in_strides)
    {
        size_t ndim = in_shape.size();
        if (ndim > 1)
        {
            const float *ptr_input = input;
            float *ptr_output = output;
            runtime_shape_t new_in_shape(in_shape.begin() + 1, in_shape.end());
            runtime_shape_t new_in_strides(in_strides.begin() + 1, in_strides.end());
            runtime_shape_t new_out_strides(out_strides.begin() + 1, out_strides.end());
            for (size_t i = 0; i < in_shape[0]; i++)
            {
                auto ret = optimized_sigmoid_impl(ptr_input, ptr_output, new_in_shape, new_in_strides, new_out_strides);
                (void)ret;
                ptr_input += in_strides[0];
                ptr_output += out_strides[0];
            }
        }
        else
        {
            // TODO: ndim == 1 with stride
        }
    }
    else
    {
        size_t n = compute_size(in_shape);
        const float *ptr_input = input;
        float *ptr_output = output;
        float one = 1.f;
        while (n)
        {
            auto vl = vsetvl_e32m8(n);
            auto v_out = vle32_v_f32m8(ptr_input, vl);
            v_out = vfneg_v_f32m8(v_out, vl);
            v_out = exp_ps(v_out, vl);
            v_out = vfadd_vf_f32m8(v_out, one, vl);
            v_out = vfrdiv_vf_f32m8(v_out, one, vl);
            vse32_v_f32m8(ptr_output, v_out, vl);

            ptr_input += vl;
            ptr_output += vl;
            n -= vl;
        }
    }

    return ok();
}
#endif
}

template result<void> optimized::sigmoid<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> optimized::sigmoid(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides) noexcept
{
#if __riscv_vector
    return optimized_sigmoid_impl(input, output, in_shape, in_strides, out_strides);
#endif

    return cpu::reference::sigmoid(input, output, in_shape, in_strides, out_strides);
}