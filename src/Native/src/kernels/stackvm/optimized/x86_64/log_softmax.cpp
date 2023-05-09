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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

#include <math.h>
#include "nncase_log.h"

void log_softmax_golden_step1(int32_t len, const float* x, float* dx)
{
    float max_value = x[0];
    for (int32_t i = 1; i < len; i++)
    {
        max_value = fmaxf(max_value, x[i]);
    }
    float sum_value = 0;
    for (int32_t i = 0; i < len; i++)
    {
		sum_value += expf(x[i] - max_value);
    }
	sum_value = log(sum_value);
    for (int32_t i = 0; i < len; i++)
    {
        dx[i] = x[i] - max_value - sum_value;
    }
}

void log_softmax_golden_step_not1(int32_t len, const float* x, float* dx, int step)
{
    float max_value = x[0];
    for (int32_t i = 1; i < len; i++)
    {
        max_value = fmaxf(max_value, x[i * step]);
    }
    float sum_value = 0;
    for (int32_t i = 0; i < len; i++)
    {
		sum_value += exp(x[i * step] - max_value);
    }
	sum_value = log(sum_value);
    for (int32_t i = 0; i < len; i++)
    {
        dx[i] = x[i * step] - max_value - sum_value;
    }
}

void log_softmax_golden_step_not1_2(int32_t len, const float* x, float* dx, int step)
{
	printf("----------!!!!!!!!!!!!!!!!!!!!! %d,%d\n", len , step);
	
	// print_vector_by_type(x, len * step, 16, "... data ... ", 1);
	for(int j = 0; j < step; ++j)
	{
		/// data_type: 1 float, 2 int32, 3 int64
// static void print_vector_by_type(const void *a, int len, int align_number, const char *title, int data_type)
		float max_value = x[0];
		for (int32_t i = 1; i < len; i++)
		{
			max_value = fmaxf(max_value, x[i * step]);
		}
		float sum_value = 0;
		for (int32_t i = 0; i < len; i++)
		{
			sum_value += exp(x[i * step] - max_value);
		}
		sum_value = log(sum_value);
		for (int32_t i = 0; i < len; i++)
		{
			dx[i] = x[i * step] - max_value - sum_value;
		}
		++ x;
		++ dx;
	}
}


static void log_softmax_impl(const float *input, float *output, const dims_t &in_shape, int axis)
{
	size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++)
	{
        in_side *= in_shape[i];
	}
	 printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ndim: %d, out_side: %d, in_side: %d, axis: %d, axis_dim:%d\n", (int)ndim, (int)out_side, 
	 (int)in_side, (int)positive_axis, (int)axis_dim);
	if (positive_axis == (ndim - 1)) {
		const float *ptr_input = input;
        float *ptr_output = output;
        for (size_t i = 0; i < out_side; i++) {
            int n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;
			log_softmax_golden_step1(n, ptr_input_vl, ptr_output_vl);
			
			ptr_input += axis_dim;
            ptr_output += axis_dim;
		}
	}
	else
	{
		const float *ptr_input = input;
        float *ptr_output = output;
        for (size_t i = 0; i < out_side; i++) {
            int n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;
			#if(0)
			for(int i = 0; i < in_side; ++i)
			{
				log_softmax_golden_step_not1(n, ptr_input_vl, ptr_output_vl, in_side);
				ptr_input_vl += 1;
				ptr_output_vl += 1;
			}
			#else
				log_softmax_golden_step_not1_2(n, ptr_input_vl, ptr_output_vl, in_side);
			#endif

			ptr_input += axis_dim * in_side;
            ptr_output += axis_dim * in_side;
		}
	}
}

template result<void>
optimized::log_softmax<float>(const float *input, float *output,
                          const dims_t &in_shape, [[maybe_unused]] const dims_t &in_strides,
                          [[maybe_unused]] const dims_t &out_strides, int32_t axis,
                          [[maybe_unused]]float beta) noexcept;

template <typename T>
result<void>
optimized::log_softmax(const T *input, T *output, const dims_t &in_shape,
                   [[maybe_unused]] const dims_t &in_strides, [[maybe_unused]]const dims_t &out_strides,
                   int32_t axis, [[maybe_unused]]float beta) noexcept {
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! x86 ....   \n");
	log_softmax_impl(input, output, in_shape, axis);
	return ok();
}

