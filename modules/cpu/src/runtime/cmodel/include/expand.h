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
template <class T>
void expand_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                 gsl::span<const size_t> input_strides,
                 gsl::span<const size_t> out_shape,
                 gsl::span<const size_t> out_strides) noexcept {
    return apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        const auto in_index = get_reduced_offset(index, in_shape);
        output[offset(out_strides, index)] =
            input[offset(input_strides, in_index)];
        return;
    });
}
} // namespace

template <class T>
void expand(const T *input, T *output, gsl::span<const size_t> input_shape,
            gsl::span<const size_t> input_strides,
            gsl::span<const size_t> out_shape,
            gsl::span<const size_t> out_strides) {
    expand_impl(input, output, input_shape, input_strides, out_shape,
                out_strides);
    return;
}
} // namespace kernels