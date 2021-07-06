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
result<void> gather_nd_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
                            const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t batch_dims,
                            kernel_context &context) noexcept
{
    auto last_indices_index = indices_shape.size() - 1;
    auto indices_list_size = indices_shape[last_indices_index];
    auto indices_block_count = std::accumulate(indices_shape.begin() + batch_dims, indices_shape.end() - 1, 1, std::multiplies<size_t>{});

    auto block_size = std::accumulate(in_shape.begin() + indices_shape[last_indices_index] + batch_dims, in_shape.end(), 1, std::multiplies<size_t>{});


    auto *out_ptr = output;
    auto *in_ptr = input;
    auto *indices_ptr = indices;
    auto batch_size = std::accumulate(in_shape.begin(), in_shape.begin() + batch_dims, 1, std::multiplies<size_t>{});

    for(size_t i = 0; i < batch_size; ++i)
    {
        for(size_t j = 0; j < indices_block_count; ++j)
        {
            auto *batch_begin_input = in_ptr;
            // set batch_dims value used for select input

            // get offset
            for(size_t k = 0; k < indices_list_size; ++k)
            {
                batch_begin_input += indices[k] * in_strides[k + batch_dims];
            }
            memcpy(out_ptr, batch_begin_input, block_size * sizeof(T));
            out_ptr += block_size;
            indices += indices_list_size;
        }
        in_ptr += std::accumulate(in_shape.begin() + batch_dims, in_shape.end(), 1, std::multiplies<size_t>{});
        // output += i * std::accumulate(out_shape.begin() + batch_dims, out_shape.end(), 1, std::multiplies<size_t>{});
        // indices += i * std::accumulate(indices_shape.begin() + batch_dims, indices_shape.end(), 1, std::multiplies<size_t>{});
    }
    return ok();
}
}

#define GATHER_ND_IMPL(size, type) \
    case size:                     \
        return gather_nd_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, out_shape, in_strides, out_strides, indices, indices_shape, batch_dims, context);

result<void> optimized::gather_nd(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
                                  const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t batch_dims, kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, GATHER_ND_IMPL);
}
