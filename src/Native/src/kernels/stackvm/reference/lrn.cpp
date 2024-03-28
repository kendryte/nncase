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
template <typename T>
result<void> lrn_impl(const T *input, float alpha, float beta, float bias,
                      int64_t size, T *output, const T *square_sum,
                      std::span<const size_t> in_shape,
                      std::span<const size_t> in_strides,
                      std::span<const size_t> out_strides) {
    return apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        auto off = offset(in_strides, index);
        const auto x = input[off];
        const auto num = square_sum[off];
        output[offset(out_strides, index)] =
            x / static_cast<T>(std::pow(
                    static_cast<float>(num) * alpha / size + bias, beta));
        return ok();
    });
}
} // namespace

template <typename T>
result<void> lrn_impl2(typecode_t type, const T *input, float alpha, float beta,
                       float bias, int size, T *output,
                       std::span<const size_t> in_shape,
                       std::span<const size_t> in_strides,
                       std::span<const size_t> out_strides) {
    std::vector<std::unique_ptr<T[]>> tmpData;
    std::vector<dims_t> tmpShapes;
    std::vector<dims_t> tmpStrides;
    auto concat_size = 0;
    auto square_data = std::make_unique<T[]>(runtime::compute_size(in_shape));
    try_(nncase::kernels::stackvm::reference::unary(
        type, runtime::stackvm::unary_op_t::square, IN_BYTE_CAST(input),
        OUT_BYTE_CAST(square_data.get()), in_shape, in_strides, in_shape,
        in_strides));
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
            std::make_unique<T[]>(runtime::compute_size(tmp_out_shape));
        try_(slice(type, IN_BYTE_CAST(square_data.get()),
                   OUT_CAST(std::byte, slice_out.get()), in_shape, in_strides,
                   out_strides, begins, ends, strides,
                   default_kernel_context()));

        auto keep_dims = true;
        auto axes = dims_t{1};
        auto reduce_shape = reduce_infer_shape(tmp_out_shape, axes, keep_dims);
        auto reduce_size = runtime::compute_size(reduce_shape);
        concat_size += reduce_size;
        tmpData.push_back(std::make_unique<T[]>(reduce_size));
        tmpShapes.push_back(reduce_shape);
        auto reduce_out_strides = runtime::get_default_strides(reduce_shape);
        tmpStrides.push_back(reduce_out_strides);
        auto init_value = 0;
        try_(nncase::kernels::stackvm::reference::reduce(
            type, reduce_op_t::sum, IN_CAST(std::byte, &init_value),
            IN_CAST(std::byte, slice_out.get()),
            OUT_CAST(std::byte, tmpData[i].get()), tmp_out_shape, axes,
            tmp_out_strides, reduce_out_strides, keep_dims));
    }

    auto concat_output = std::make_unique<T[]>(concat_size);
    auto concat_shape = concat_infer_shape(tmpShapes, 1);
    auto concat_strides = runtime::get_default_strides(concat_shape);
    auto concat_dims = dims_t();
    auto axis = 1;
    for (auto &tmpShape : tmpShapes) {
        concat_dims.push_back(tmpShape[axis]);
    }
    std::vector<const std::byte *> concat_inputs;
    for (auto &i : tmpData) {
        concat_inputs.push_back(IN_CAST(std::byte, i.get()));
    }
    try_(nncase::kernels::stackvm::reference::concat(
        type, concat_inputs, OUT_CAST(std::byte, concat_output.get()),
        concat_shape, tmpStrides, concat_strides, axis, concat_dims))
        try_(lrn_impl(input, alpha, beta, bias, size, output,
                      concat_output.get(), in_shape, in_strides, out_strides));
    return ok();
}

#define LRN_IMPL(type)                                                         \
    return lrn_impl2(typecode, IN_CAST(type, input), alpha, beta, bias, size,  \
                     OUT_CAST(type, output), in_shape, in_strides,             \
                     out_strides);

#define TYPE_SELECT_LRN(_typecode, _impl)                                      \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_bfloat16:                                                          \
        _impl(bfloat16);                                                       \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

result<void> nncase::kernels::stackvm::reference::lrn(
    typecode_t typecode, const std::byte *input, float alpha, float beta,
    float bias, int size, std::byte *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides) {
    TYPE_SELECT_LRN(typecode, LRN_IMPL)
}
