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
#include "runtime_utils.h"

namespace kernels {

namespace {
std::pair<size_t, size_t>
find_input_id_and_index(size_t index,
                        gsl::span<const size_t> concat_dims) noexcept {
    size_t input_id;
    for (input_id = 0;; input_id++) {
        auto input_dim = concat_dims[input_id];
        if (index < input_dim)
            break;
        index -= input_dim;
    }

    return std::make_pair(input_id, index);
}

} // namespace

template <class T>
void concat(gsl::span<const gsl::byte *> inputs, T *output,
            gsl::span<const size_t> out_shape,
            gsl::span<const strides_t> in_strides,
            gsl::span<const size_t> out_strides, size_t axis,
            gsl::span<const size_t> concat_dims) noexcept {
    return apply(out_shape, [&](gsl::span<const size_t> out_index) -> void {
        auto in_id_index =
            find_input_id_and_index(out_index[axis], concat_dims);
        auto input = reinterpret_cast<const T *>(inputs[in_id_index.first]);
        auto &sel_in_strides = in_strides[in_id_index.first];
        dims_t in_index(out_index);
        in_index[axis] = in_id_index.second;

        output[offset(out_strides, out_index)] =
            input[offset(sel_in_strides, in_index)];
    });
}

} // namespace kernels