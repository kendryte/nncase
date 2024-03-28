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
#include "opt_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
template <class T, class IndicesT>
result<void>
gather_nd_impl(const T *input, T *output, std::span<const size_t> in_shape,
               NNCASE_UNUSED std::span<const size_t> out_shape,
               std::span<const size_t> in_strides,
               NNCASE_UNUSED std::span<const size_t> out_strides,
               const IndicesT *indices, std::span<const size_t> indices_shape,
               size_t batch_dims,
               NNCASE_UNUSED kernel_context &context) noexcept {
    auto last_indices_index = indices_shape.size() - 1;
    auto indices_list_size = indices_shape[last_indices_index];
    size_t indices_block_count =
        std::accumulate(indices_shape.begin() + batch_dims,
                        indices_shape.end() - 1, 1, std::multiplies<size_t>{});

    size_t block_size = std::accumulate(
        in_shape.begin() + indices_shape[last_indices_index] + batch_dims,
        in_shape.end(), 1, std::multiplies<size_t>{});

    size_t batch_size =
        std::accumulate(in_shape.begin(), in_shape.begin() + batch_dims, 1,
                        std::multiplies<size_t>{});

    size_t input_batch_block_size =
        std::accumulate(in_shape.begin() + batch_dims, in_shape.end(), 1,
                        std::multiplies<size_t>{});
    size_t output_batch_block_size =
        std::accumulate(out_shape.begin() + batch_dims, out_shape.end(), 1,
                        std::multiplies<size_t>{});
    size_t indices_batch_block_size =
        std::accumulate(indices_shape.begin() + batch_dims, indices_shape.end(),
                        1, std::multiplies<size_t>{});
    for (size_t i = 0; i < batch_size; ++i) {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        for (int j = 0; j < indices_block_count; ++j) {
            const auto *indices_ptr = indices + j * indices_list_size;
            auto *out_ptr = output + j * block_size;
            auto *batch_begin_input = input;
            // set batch_dims value used for select input

            // get offset
            for (size_t k = 0; k < indices_list_size; ++k) {
                batch_begin_input +=
                    indices_ptr[k] * in_strides[k + batch_dims];
            }
            memcpy(out_ptr, batch_begin_input, block_size * sizeof(T));
        }
        input += input_batch_block_size;
        output += output_batch_block_size;
        indices += indices_batch_block_size;
    }
    return ok();
}
} // namespace

#define GATHER_ND_IMPL(size, type)                                             \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return gather_nd_impl(reinterpret_cast<const type *>(input),       \
                                  reinterpret_cast<type *>(output), in_shape,  \
                                  out_shape, in_strides, out_strides,          \
                                  indices_value, indices_shape, batch_dims,    \
                                  context);                                    \
        });

result<void> optimized::gather_nd(
    datatype_t type, const std::byte *input, std::byte *output,
    std::span<const size_t> in_shape, std::span<const size_t> out_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    datatype_t indices_type, const std::byte *indices,
    std::span<const size_t> indices_shape, size_t batch_dims,
    kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, GATHER_ND_IMPL);
}
