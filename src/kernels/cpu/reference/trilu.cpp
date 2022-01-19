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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::trilu<float>(const float *input, float *output, const runtime_shape_t &in_shape, const bool upper, const int64_t k) noexcept;

template <typename T>
result<void> reference::trilu(const T *input, T *output, const runtime_shape_t &in_shape, const bool upper, const int64_t k) noexcept
{
    // copy input to output
    memcpy(static_cast<void *>(output), static_cast<const void *>(input), compute_size(in_shape) * sizeof(T));

    size_t ndim = in_shape.size();
    assert(ndim >= 2);
    int64_t batch_size = 1;
    for (size_t i = 0; i < ndim - 2; ++i)
    {
        batch_size *= in_shape[i];
    }

    int64_t matrix_h = static_cast<int64_t>(in_shape[ndim - 2]);
    int64_t matrix_w = static_cast<int64_t>(in_shape[ndim - 1]);
    int64_t num_matrix_elems = matrix_h * matrix_w;

    // zero the specific diagonal
    for (int64_t b = 0; b < batch_size; b++)
    {
        auto p = output + (b * num_matrix_elems);
        if (upper)
        {
            int64_t start_i = k > 0 ? 0 : 1 - k;
            for (int64_t i = start_i; i < matrix_h; i++)
            {
                for (int64_t j = 0; j < i + k && j < matrix_w; j++)
                {
                    p[i * matrix_w + j] = static_cast<T>(0);
                }
            }
        }
        else
        {
            int64_t end_i = std::min(matrix_h, matrix_w - k);
            for (int64_t i = 0; i < end_i; i++)
            {
                for (int64_t j = std::max(static_cast<int64_t>(0), i + k + 1); j < matrix_w; j++)
                {
                    p[i * matrix_w + j] = static_cast<T>(0);
                }
            }
        }
    }

    return ok();
}
