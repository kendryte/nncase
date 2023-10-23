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
void clamp_impl(const T *input, T min, T max, T *output,
                gsl::span<const size_t> in_shape,
                gsl::span<const size_t> in_strides,
                gsl::span<const size_t> out_strides) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto v = input[offset(index, in_strides)];
        output[offset(index, out_strides)] = static_cast<T>(
            std::min(std::max(static_cast<float>(v), static_cast<float>(min)),
                     static_cast<float>(max)));
        return;
    });
}
} // namespace

template <class T>
void clamp(const T *input, T *output, T min, T max,
           gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_strides) {
    clamp_impl(input, min, max, output, in_shape, in_strides, out_strides);
}
} // namespace kernels