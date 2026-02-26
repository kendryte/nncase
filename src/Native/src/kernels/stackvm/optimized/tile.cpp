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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/util.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

result<void> nncase::kernels::stackvm::optimized::tile(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    gsl::span<const size_t> repeats) {
    return reference::tile(dt, input, output, in_shape, out_shape, in_strides,
                           out_strides, repeats);
}