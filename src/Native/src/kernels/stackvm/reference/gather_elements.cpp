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
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class IndicesT>
result<void>
gather_elements_impl(const T *input, T *output,[[maybe_unused]] const dims_t &in_shape,
                     const dims_t &out_shape, const strides_t &in_strides,
                     const strides_t &out_strides, const IndicesT *indices,
                     const dims_t &indices_shape, size_t axis,
                     NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape, [&](const dims_t &out_index) -> result<void> {
        dims_t in_index(out_index);

        auto indices_offset =
            offset(get_default_strides(indices_shape), out_index);
        in_index[axis] = indices[indices_offset];

        output[offset(out_strides, out_index)] =
            input[offset(in_strides, in_index)];
        return ok();
    });
}
} // namespace

#define GATHER_ELEMENTS_IMPL(size, type)                                       \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return gather_elements_impl(reinterpret_cast<const type *>(input), \
                                        reinterpret_cast<type *>(output),      \
                                        in_shape, out_shape, in_strides,       \
                                        out_strides, indices_value,            \
                                        indices_shape, axis, context);         \
        });

result<void> nncase::kernels::stackvm::reference::gather_elements(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    const dims_t &in_shape, const dims_t &out_shape,
    const strides_t &in_strides, const strides_t &out_strides,
    datatype_t indices_type, const gsl::byte *indices,
    const dims_t &indices_shape, size_t axis,
    kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, GATHER_ELEMENTS_IMPL);
}
