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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

result<void> nncase::kernels::stackvm::optimized::layer_norm(
    typecode_t typecode, const gsl::byte *input, gsl::byte *output,
    const gsl::byte *scale, const gsl::byte *bias,
    gsl::span<const size_t> in_shape, int32_t axis, float epsilon, bool use_mean) {
    return reference::layer_norm(typecode, input, output, scale, bias, in_shape,
                                 axis, epsilon, use_mean);
}
