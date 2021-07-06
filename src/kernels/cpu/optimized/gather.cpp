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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
template <class T>
result<void> gather_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
                         const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis,
                         kernel_context &context) noexcept
{
    auto outer_count = std::accumulate(in_shape.begin(), in_shape.begin() + axis, 1, std::multiplies<size_t>{});
    auto indices_count = compute_size(indices_shape);
    auto block_size = std::accumulate(in_shape.begin() + axis + 1, in_shape.end(), 1, std::multiplies<size_t>{});

    auto *in_ptr = input;
    auto *out_ptr = output;
    for(size_t o = 0; o < outer_count; ++o)
    {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for(size_t i = 0; i < indices_count; ++i)
        {
            auto *o_ptr = out_ptr + i * block_size;
            auto indices_ptr = indices[i];
            memcpy(o_ptr, in_ptr + (indices_ptr * block_size), block_size * sizeof(T));
        }
        in_ptr += in_shape[axis] * block_size;
        out_ptr += indices_count * block_size;
    }
    return ok();
}
}

#define GATHER_IMPL(size, type) \
    case size:                  \
        return gather_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, out_shape, in_strides, out_strides, indices, indices_shape, axis, context);

result<void> optimized::gather(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis, kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, GATHER_IMPL);
}
