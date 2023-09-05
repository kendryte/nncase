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

#include <apply.h>
#include <gsl/gsl-lite.hpp>
#include <runtime_utils.h>

namespace kernels {

namespace {
template <class T, class IndicesT>
void gather_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                 gsl::span<const size_t> out_shape,
                 gsl::span<const size_t> in_strides,
                 gsl::span<const size_t> out_strides, const IndicesT *indices,
                 gsl::span<const size_t> indices_shape, size_t axis) noexcept {
    // scalar
    if (out_shape.size() == 0) {
        *output = input[indices[0]];
        return;
    }

    apply(out_shape, [&](gsl::span<const size_t> out_index) -> void {
        // select batch
        // [out_index.begin(), out_index.begin() + axis]
        dims_t in_index(in_shape.size());
        size_t i_index = 0;
        for (; i_index < static_cast<size_t>(axis); ++i_index) {
            in_index[i_index] = out_index[i_index];
        }

        // which index to be used in indices
        // dims_t indices_index(out_index.begin() + axis,
        //                      out_index.begin() + axis + indices_shape.size());
        // auto indices_offset =
        //     offset((indices_shape), indices_index);
        // select sub block in dim axis
        // in_index[i_index] = indices[indices_offset];
        // ++i_index;

        // select position in sub block
        for (auto o_index = axis + indices_shape.size();
             o_index < out_index.size(); ++o_index, ++i_index) {
            in_index[i_index] = out_index[o_index];
        }
        // output[offset(out_strides, out_index)] =
        //     input[offset(in_strides, in_index)];
    });
}
} // namespace

template <class T, class IndicesT>
void gather(const T *input, T *output, gsl::span<const size_t> in_shape,
            gsl::span<const size_t> in_strides,
            gsl::span<const size_t> out_shape,
            gsl::span<const size_t> out_strides, const IndicesT *indices,
            gsl::span<const size_t> indices_shape, size_t axis) noexcept {
    gather_impl(input, output, in_shape, out_shape, in_strides, out_strides,
                indices, indices_shape, axis);
}

} // namespace kernels