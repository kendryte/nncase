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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

template <typename T>
result<void>
trilu_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
           gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_strides, long k, bool upper) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto h = index.size() - 2;
        auto w = index.size() - 1;
        if (upper) {
            auto value = w >= (h + k) ? 0 : input[offset(in_strides, index)];
            output[offset(out_strides, index)] = value;
        } else {
            auto value = w >= (h + k) ? input[offset(in_strides, index)] : 0;
            output[offset(out_strides, index)] = value;
        }
        return ok();
    });
}

#define TRILU_IMPL(size, type)                                                 \
    case size:                                                                 \
        return trilu_impl(IN_CAST(type, input), OUT_CAST(type, output),        \
                          in_shape, in_strides, out_strides, k, upper)

result<void> nncase::kernels::stackvm::reference::trilu(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, long k, bool upper) noexcept {
    switch (runtime::get_bytes(type)) {
        TRILU_IMPL(1, uint8_t);
        TRILU_IMPL(2, uint16_t);
        TRILU_IMPL(4, uint32_t);
        TRILU_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}