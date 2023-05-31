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

#include "opt_common.h"
#include "opt_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

#define UNFOLD_OUTPUT_ASSIGN_4(_output_ptr)                                    \
    auto *o0 = _output_ptr + 0;                                                \
    auto *o1 = _output_ptr + 1;                                                \
    auto *o2 = _output_ptr + 2;                                                \
    auto *o3 = _output_ptr + 3;

#define DIV_4(arg)                                                             \
    auto res = div((arg), 4);                                                  \
    size_t res_quot = size_t(res.quot);                                        \
    size_t res_rem = size_t(res.rem);

// todo: fix, for 3d
namespace {
template <class T>
result<void> transpose_impl(const T *input, T *output, const dims_t &in_shape,
                            const dims_t &perm) {
    dims_t out_shape(in_shape.size());
    for (size_t i = 0; i < 4; i++) {
        out_shape[i] = in_shape[perm[i]];
    }
    DIV_4(out_shape[3])
    size_t move_distance =
        std::accumulate(in_shape.begin() + perm[3] + 1, in_shape.end(), 1,
                        std::multiplies<size_t>());

    auto out_img_size = out_shape[2] * out_shape[3];
    for (size_t b = 0; b < out_shape[0]; b++) {
        // #pragma omp parallel for
        // num_threads(kernels::default_kernel_context().num_threads)
        for (size_t c = 0; c < out_shape[1]; c++) {
            dims_t index(4);
            index[perm[0]] = b;
            index[perm[1]] = c;
            auto *output_ptr =
                output + b * out_shape[1] * out_img_size + c * out_img_size;
            for (size_t h = 0; h < out_shape[2]; h++) {
                index[perm[2]] = h;
                auto *input_ptr = input + linear_index(in_shape, index);
                for (size_t w = 0; w < res_quot; w++) {
                    auto *i0 = input_ptr + 0 * move_distance;
                    auto *i1 = input_ptr + 1 * move_distance;
                    auto *i2 = input_ptr + 2 * move_distance;
                    auto *i3 = input_ptr + 3 * move_distance;
                    UNFOLD_OUTPUT_ASSIGN_4(output_ptr);
                    *o0 = *i0;
                    *o1 = *i1;
                    *o2 = *i2;
                    *o3 = *i3;
                    input_ptr += 4 * move_distance;
                    output_ptr += 4;
                }
                for (size_t w = 0; w < res_rem; w++) {
                    *output_ptr = *input_ptr;
                    input_ptr += move_distance;
                    ++output_ptr;
                }
            }
        }
    }
    return ok();
}

#define TRANSPOSE_IMPL(size, type)                                             \
    case size:                                                                 \
        return transpose_impl(reinterpret_cast<const type *>(src),             \
                              reinterpret_cast<type *>(dest), in_shape, perm)
} // namespace

result<void> kernels::stackvm::optimized::transpose(
    datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const dims_t &in_shape, const dims_t &perm,
    [[maybe_unused]] const strides_t &in_strides,
    [[maybe_unused]] const strides_t &out_strides,
    [[maybe_unused]] kernel_context &context) noexcept {
    switch (runtime::get_bytes(type)) {
        TRANSPOSE_IMPL(1, uint8_t);
        TRANSPOSE_IMPL(2, uint16_t);
        TRANSPOSE_IMPL(4, uint32_t);
        TRANSPOSE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}