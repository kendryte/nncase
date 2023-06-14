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
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
using namespace nncase::runtime::stackvm;

result<void> nncase::kernels::stackvm::optimized::where(
    datatype_t dt, const bool *cond, const gsl::byte *x, const gsl::byte *y,
    gsl::byte *output, gsl::span<const size_t> cond_shape,
    gsl::span<const size_t> x_shape, gsl::span<const size_t> y_shape,
    gsl::span<const size_t> out_shape, gsl::span<const size_t> cond_strides,
    gsl::span<const size_t> x_strides, gsl::span<const size_t> y_strides,
    gsl::span<const size_t> out_strides) {
    return reference::where(dt, cond, x, y, output, cond_shape, x_shape,
                            y_shape, out_shape, cond_strides, x_strides,
                            y_strides, out_strides);
}
