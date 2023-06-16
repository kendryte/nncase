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
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

template <typename T>
result<void> constant_of_shape_impl(const T *value, T *output,
                                    gsl::span<const size_t> shape) {
    for (size_t i = 0; i < compute_size(shape); ++i) {
        output[i] = *value;
    }
    return ok();
}

#define KERNEL_IMPL(_ty)                                                       \
    return constant_of_shape_impl(IN_CAST(_ty, value), OUT_CAST(_ty, output),  \
                                  shape);

result<void> nncase::kernels::stackvm::reference::constant_of_shape(
    datatype_t dt, const gsl::byte *value, gsl::byte *output,
    gsl::span<const size_t> shape) {
    try_var(tycode, to_typecode(dt));
    TYPE_SELECT(tycode, KERNEL_IMPL);
}