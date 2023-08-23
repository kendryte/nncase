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
template <class T>
void slice_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                gsl::span<const size_t> in_strides,
                gsl::span<const size_t> out_strides,
                gsl::span<const int64_t> &begins,
                gsl::span<const int64_t> &ends,
                gsl::span<const int64_t> &strides) noexcept {
    apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        dims_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++) {
            const auto stride = strides[i];
            if (stride > 0) {
                if ((int32_t)index[i] < begins[i] ||
                    index[i] >= static_cast<size_t>(ends[i]))
                    return;
            } else {
                if ((int32_t)index[i] <= ends[i] ||
                    (int32_t)index[i] > begins[i])
                    return;
            }

            auto out_div =
                div((int32_t)(index[i] - begins[i]), (int32_t)strides[i]);
            if (out_div.rem)
                return;
            out_index[i] = (size_t)out_div.quot;
        }

        output[offset(out_strides, out_index)] =
            input[offset(in_strides, index)];
    });
}
} // namespace

template <class T>
void slice(const T *input, T *output, gsl::span<const size_t> in_shape,
           gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_strides, gsl::span<const int64_t> begins,
           gsl::span<const int64_t> ends,
           gsl::span<const int64_t> strides) noexcept {
    slice_impl(input, output, in_shape, in_strides, out_strides, begins, ends,
               strides);
}

} // namespace kernels