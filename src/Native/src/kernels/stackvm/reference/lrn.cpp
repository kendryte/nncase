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
#include "../shape_infer.h"
#include "ref_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_context.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels::stackvm::reference;
using namespace nncase::kernels::stackvm;

namespace {
result<void> lrn_impl(const float *input, float alpha, float beta, float bias,
                      float *output, gsl::span<const size_t> in_shape,
                      gsl::span<const size_t> in_strides,
                      gsl::span<const size_t> out_strides) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto off = offset(in_strides, index);
        const auto x = input[off];
        output[offset(out_strides, index)] =
            x / std::pow(x * alpha + bias, beta);
        return ok();
    });
}
} // namespace

result<void> nncase::kernels::stackvm::reference::lrn(
    const float *input, float alpha, float beta, float bias, int size,
    float *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides) {
    std::vector<std::unique_ptr<float[]>> tmpData;
    std::vector<dims_t> tmpShapes;
    std::vector<dims_t> tmpStrides;
    auto concat_size = 0;
    auto square_data =
        std::make_unique<float[]>(runtime::compute_size(in_shape));
    try_(reference::unary(dt_float32, runtime::stackvm::unary_op_t::square,
                          IN_BYTE_CAST(input), OUT_BYTE_CAST(output), in_shape,
                          in_strides, in_shape, in_strides));
    for (size_t i = 0; i < in_shape[1]; ++i) {
        auto beginV =
            std::max(static_cast<int64_t>(0),
                     static_cast<int64_t>(i - std::floor((size - 1) / 2)));
        auto endV =
            std::min(static_cast<int64_t>(in_shape[1] - 1),
                     static_cast<int64_t>(i + std::ceil((size - 1) / 2)));
        auto begins = axes_t{0, (int64_t)beginV, 0, 0};
        auto ends = axes_t{static_cast<int64_t>(in_shape[0]),
                           static_cast<int64_t>(endV + 1),
                           static_cast<int64_t>(in_shape[2]),
                           static_cast<int64_t>(in_shape[3])};
        auto strides = axes_t{1, 1, 1, 1};
        auto tmp_out_shape = slice_infer_shape(in_shape, begins, ends, strides);
        auto tmp_out_strides = runtime::get_default_strides(tmp_out_shape);
        auto slice_out =
            std::make_unique<float[]>(runtime::compute_size(tmp_out_shape));
        try_(slice(dt_float32, IN_BYTE_CAST(square_data.get()),
                   OUT_CAST(gsl::byte, slice_out.get()), in_shape, in_strides,
                   out_strides, begins, ends, strides,
                   default_kernel_context()));

        std::cout << "Slice Out: ";
        for (size_t j = 0; j < runtime::compute_size(tmp_out_shape); ++j) {
            std::cout << slice_out[j] << " ";
        }
        std::cout << std::endl;

        auto keep_dims = true;
        auto axes = dims_t{1};
        auto reduce_shape = reduce_infer_shape(tmp_out_shape, axes, keep_dims);
        auto reduce_size = runtime::compute_size(reduce_shape);
        concat_size += reduce_size;
        tmpData.push_back(std::make_unique<float[]>(reduce_size));
        tmpShapes.push_back(reduce_shape);
        auto reduce_out_strides = runtime::get_default_strides(reduce_shape);
        tmpStrides.push_back(reduce_out_strides);
        auto init_value = 0.f;
        try_(reference::reduce(
            dt_float32, reduce_op_t::sum, IN_CAST(gsl::byte, &init_value),
            IN_CAST(gsl::byte, slice_out.get()),
            OUT_CAST(gsl::byte, tmpData[i].get()), tmp_out_shape, axes,
            tmp_out_strides, reduce_out_strides, keep_dims));
        for (size_t k = 0; k < tmpData.size(); ++k) {
            std::cout << "tmpData[" << k << "]: ";
            for (size_t j = 0; j < runtime::compute_size(tmpShapes[k]); ++j) {
                std::cout << tmpData[k][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    auto concat_output = std::make_unique<float[]>(concat_size);
    auto concat_shape = concat_infer_shape(tmpShapes, 1);
    auto concat_strides = runtime::get_default_strides(concat_shape);
    auto concat_dims = dims_t();
    auto axis = 1;
    for (auto &tmpShape : tmpShapes) {
        concat_dims.push_back(tmpShape[axis]);
    }
    std::vector<const gsl::byte *> concat_inputs;
    for (auto &i : tmpData) {
        concat_inputs.push_back(IN_CAST(gsl::byte, i.get()));
    }
    try_(reference::concat(
        dt_float32, concat_inputs, OUT_CAST(gsl::byte, concat_output.get()),
        concat_shape, tmpStrides, concat_strides, axis, concat_dims))
        try_(lrn_impl(concat_output.get(), alpha, beta, bias, output, in_shape,
                      in_strides, out_strides));
    return ok();
}