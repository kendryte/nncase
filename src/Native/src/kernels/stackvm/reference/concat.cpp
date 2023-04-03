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
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <nncase/type.h>
#include <nncase/value.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
std::pair<size_t, size_t>
find_input_id_and_index(size_t index, const dims_t &concat_dims) noexcept {
    size_t input_id;
    for (input_id = 0;; input_id++) {
        auto input_dim = concat_dims[input_id];
        if (index < input_dim)
            break;
        index -= input_dim;
    }

    return std::make_pair(input_id, index);
}

template <class T>
result<void> concat_impl(gsl::span<const gsl::byte *const> inputs, T *output,
                         const dims_t &out_shape,
                         gsl::span<const dims_t> &in_strides,
                         const strides_t &out_strides, size_t axis,
                         const dims_t &concat_dims,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape, [&](const dims_t &out_index) -> result<void> {
        auto in_id_index =
            find_input_id_and_index(out_index[axis], concat_dims);
        auto input = reinterpret_cast<const T *>(inputs[in_id_index.first]);
        auto &sel_in_strides = in_strides[in_id_index.first];
        dims_t in_index(out_index);
        in_index[axis] = in_id_index.second;

        output[offset(out_strides, out_index)] =
            input[offset(sel_in_strides, in_index)];
        return ok();
    });
}
} // namespace

#define CONCAT_IMPL(size, type)                                                \
    case size:                                                                 \
        return concat_impl(inputs, reinterpret_cast<type *>(output),           \
                           out_shape, in_strides, out_strides, axis,           \
                           concat_dims, context)

result<void> nncase::kernels::stackvm::reference::concat(
    datatype_t type, gsl::span<const gsl::byte *const> inputs,
    gsl::byte *output, const dims_t &out_shape,
    gsl::span<const dims_t> in_strides, const strides_t &out_strides,
    size_t axis, const dims_t &concat_dims, kernel_context &context) noexcept {
    switch (runtime::get_bytes(type)) {
        CONCAT_IMPL(1, uint8_t);
        CONCAT_IMPL(2, uint16_t);
        CONCAT_IMPL(4, uint32_t);
        CONCAT_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
