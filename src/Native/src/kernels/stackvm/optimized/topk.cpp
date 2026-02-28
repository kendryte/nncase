
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
#include "../reference/ref_ops.h"
#include "opt_ops.h"
#include <map>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <queue>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

result<void> nncase::kernels::stackvm::optimized::topk(
    typecode_t typecode, const gsl::byte *input, gsl::byte *output_values,
    int64_t *output_indices, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides,
    gsl::span<const size_t> output_values_shape,
    gsl::span<const size_t> output_values_strides,
    gsl::span<const size_t> output_indices_shape,
    gsl::span<const size_t> output_indices_strides, const int64_t k,
    const int32_t axis, const bool largest, const bool sorted) noexcept {
    return reference::topk(typecode, input, output_values, output_indices,
                           in_shape, in_strides, output_values_shape,
                           output_values_strides, output_indices_shape,
                           output_indices_strides, k, axis, largest, sorted);
}
