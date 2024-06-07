/* Copyright 2019-2020 Canaan Inc.
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
#include "../runtime_function.h"
#include <nncase/kernels/k210/k210_kernels.h>
#include <nncase/kernels/tensor_compute.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void> k210_runtime_function::visit(const copy_options &op) noexcept {
    try_var(input, memory_at(op.input));
    try_var(output, memory_at(op.output));

    runtime_shape_t in_shape{op.in_shape.begin(), op.in_shape.end()};
    runtime_shape_t in_strides{op.in_strides.begin(), op.in_strides.end()};
    runtime_shape_t out_strides{op.out_strides.begin(), op.out_strides.end()};
    return kernels::copy(op.input.datatype,
                         reinterpret_cast<const std::byte *>(input.data()),
                         reinterpret_cast<std::byte *>(output.data()), in_shape,
                         in_strides, out_strides);
}
