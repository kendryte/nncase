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
result<void> slice_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                        gsl::span<const size_t> in_strides,
                        gsl::span<const size_t> out_strides, const axes_t &begins,
                        const axes_t &ends, const axes_t &strides,
                        NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        dims_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++) {
            const auto stride = strides[i];
            if (stride > 0) {
                if ((int32_t)index[i] < begins[i] ||
                    index[i] >= static_cast<size_t>(ends[i]))
                    return ok();
            } else {
                if ((int32_t)index[i] <= ends[i] ||
                    (int32_t)index[i] > begins[i])
                    return ok();
            }

            auto out_div =
                div((int32_t)(index[i] - begins[i]), (int32_t)strides[i]);
            if (out_div.rem)
                return ok();
            out_index[i] = (size_t)out_div.quot;
        }

        output[offset(out_strides, out_index)] =
            input[offset(in_strides, index)];
        return ok();
    });
}
} // namespace

#define SLICE_IMPL(size, type)                                                 \
    case size:                                                                 \
        return slice_impl(reinterpret_cast<const type *>(input),               \
                          reinterpret_cast<type *>(output), in_shape,          \
                          in_strides, out_strides, begins, ends, strides,      \
                          context)

result<void> nncase::kernels::stackvm::reference::slice(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, const axes_t &begins, const axes_t &ends,
    const axes_t &strides, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, SLICE_IMPL);
}
