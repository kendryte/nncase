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

#include "../../gsl-lite.hpp"
#include <apply.h>
#include <runtime_utils.h>

using namespace nncase::runtime::cpu;
namespace kernels {
namespace {
template <class TInput, class TOutput>
void cast_impl(const TInput *input, TOutput *output,
               gsl::span<const size_t> in_shape,
               gsl::span<const size_t> in_strides,
               gsl::span<const size_t> out_strides) noexcept {
    if (is_scalar(in_shape)) {
        output[0] = static_cast<TOutput>(input[0]);
        return;
    }
    return apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = static_cast<TOutput>(value);
        return;
    });
}
} // namespace

template <class TI, class TO>
void cast(const TI *input, TO *output, gsl::span<const size_t> in_shape,
          gsl::span<const size_t> in_strides,
          gsl::span<const size_t> out_strides) {
    // TODO: support other cast mode
    // TODO: support float32->bfloat16 or float32->float16
    cast_impl(input, output, in_shape, in_strides, out_strides);
    return;
}
} // namespace kernels