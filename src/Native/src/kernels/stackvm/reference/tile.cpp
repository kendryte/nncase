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
using namespace nncase::kernels::stackvm;

template <typename T>
result<void>
tile_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
          gsl::span<const size_t> out_shape, gsl::span<const size_t> in_strides,
          gsl::span<const size_t> out_strides,
          [[maybe_unused]] gsl::span<const size_t> repeats) {
    return apply(out_shape, [&](const auto &out_index) -> result<void> {
        auto in_index = dims_t(out_index.size());
        for (size_t i = 0; i < in_shape.size(); ++i) {
            in_index[i] = out_index[i] % in_shape[i];
        }
        output[offset(out_strides, out_index)] =
            input[offset(in_strides, in_index)];
        return ok();
    });
}

#define TILE_IMPL(_ty)                                                         \
    return tile_impl(IN_CAST(_ty, input), OUT_CAST(_ty, output), in_shape,     \
                     out_shape, in_strides, out_strides, repeats);

result<void> nncase::kernels::stackvm::reference::tile(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    gsl::span<const size_t> repeats) {
    try_var(tycode, to_typecode(dt));
    TYPE_SELECT(tycode, TILE_IMPL);
}