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
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

result<void> optimized::reduce(
    typecode_t typecode, nncase::runtime::stackvm::reduce_op_t op,
    const std::byte *init_value, const std::byte *input, std::byte *output,
    std::span<const size_t> in_shape, std::span<const size_t> axis,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    bool keep_dims, kernel_context &context) noexcept {
    return stackvm::reference::reduce(typecode, op, init_value, input, output,
                                      in_shape, axis, in_strides, out_strides,
                                      keep_dims, context);
}
