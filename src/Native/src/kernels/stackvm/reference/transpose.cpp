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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T>
result<void> transpose_impl(const T *input, T *output, const dims_t &in_shape,
                            const dims_t &perm, const strides_t &in_strides,
                            const strides_t &out_strides,
                            NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(in_shape, [&](const dims_t &index) -> result<void> {
        dims_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++)
            out_index[i] = index[perm[i]];
        output[offset(out_strides, out_index)] =
            input[offset(in_strides, index)];
        return ok();
    });
}

#define TRANSPOSE_IMPL(size, type)                                             \
    case size:                                                                 \
        return transpose_impl(reinterpret_cast<const type *>(src),             \
                              reinterpret_cast<type *>(dest), in_shape, perm,  \
                              in_strides, out_strides, context)
} // namespace

result<void> nncase::kernels::stackvm::reference::transpose(
    datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const dims_t &in_shape, const dims_t &perm, const strides_t &in_strides,
    const strides_t &out_strides, kernel_context &context) noexcept {
    switch (runtime::get_bytes(type)) {
        TRANSPOSE_IMPL(1, uint8_t);
        TRANSPOSE_IMPL(2, uint16_t);
        TRANSPOSE_IMPL(4, uint32_t);
        TRANSPOSE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
