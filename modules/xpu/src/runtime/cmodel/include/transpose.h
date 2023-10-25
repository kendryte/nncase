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

namespace kernels {

namespace {
template <class T>
void transpose_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                    gsl::span<const size_t> perm,
                    gsl::span<const size_t> in_strides,
                    gsl::span<const size_t> out_strides) noexcept {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        dims_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++)
            out_index[i] = index[perm[i]];
        output[offset(out_strides, out_index)] =
            input[offset(in_strides, index)];
    });
}

} // namespace

template <class T>
void transpose(const T *src, T *dest, gsl::span<const size_t> in_shape,
               gsl::span<const size_t> perm, gsl::span<const size_t> in_strides,
               gsl::span<const size_t> out_strides) noexcept {
    transpose_impl(src, dest, in_shape, perm, in_strides, out_strides);
}

} // namespace kernels