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
result<void> clamp_impl(const T *input, T min, T max, T *output,
                        gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
                        gsl::span<const size_t> out_strides,
                        NNCASE_UNUSED kernel_context &context) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        const auto v = input[offset(index, in_strides)];
        output[offset(index, out_strides)] = std::min(std::max(v, min), max);
        return ok();
    });
}
} // namespace

#define CLAMP_IMPL(type)                                                       \
    return clamp_impl(IN_CAST(type, input), *IN_CAST(type, min),               \
                      *IN_CAST(type, max), OUT_CAST(type, output), in_shape,   \
                      in_strides, out_strides, context);

result<void> nncase::kernels::stackvm::reference::clamp(
    typecode_t type, const gsl::byte *input, const gsl::byte *min,
    const gsl::byte *max, gsl::byte *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    TYPE_SELECT(type, CLAMP_IMPL);
}