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
#include "kernel_template.h"
#include "ref_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

namespace {
template <class T>
result<void> batchnorm_impl(const T *input, const T *scale, const T *bias,
                            const T *input_mean, const T *input_var, T *output,
                            gsl::span<const size_t> in_shape,
                            gsl::span<const size_t> in_strides,
                            gsl::span<const size_t> out_strides,
                            float epsilon) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto c = index[1];
        const auto x = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = static_cast<T>(
            (static_cast<float>(x) - static_cast<float>(input_mean[c])) /
                std::sqrt(static_cast<float>(input_var[c]) +
                          static_cast<float>(epsilon)) *
                static_cast<float>(scale[c]) +
            static_cast<float>(bias[c]));
        return ok();
    });
} // namespace

#define BATCHNORM_IMPL(type)                                                   \
    return batchnorm_impl(IN_CAST(type, input), IN_CAST(type, scale),          \
                          IN_CAST(type, bias), IN_CAST(type, input_mean),      \
                          IN_CAST(type, input_var), OUT_CAST(type, output),    \
                          in_shape, in_strides, out_strides, epsilon);

} // namespace

result<void> nncase::kernels::stackvm::reference::batchnorm(
    typecode_t typecode, const gsl::byte *input, const gsl::byte *scale,
    const gsl::byte *bias, const gsl::byte *input_mean,
    const gsl::byte *input_var, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, float epsilon) {
    TYPE_SELECT(typecode, BATCHNORM_IMPL);
}