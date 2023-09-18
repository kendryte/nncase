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
#include <cstring>
#include <limits>
#include <nncase/kernels/apply.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <unordered_map>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

template <typename T>
result<void> hardmax_impl(const T *input, gsl::span<const size_t> in_shape,
                          gsl::span<const size_t> in_strides, T *output,
                          int32_t axis) noexcept {
    // init with init_value
    auto cmp = [](T a, T b) { return a > b; };
    T init_value = std::numeric_limits<T>::lowest();
    bool keep_dims = true;
    dims_t axes{static_cast<size_t>(axis)};
    auto max_shape =
        kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    auto max_stride = get_default_strides(max_shape);
    std::unique_ptr<T[]> ptr(new T[compute_size(max_shape)]);
    try_(kernels::stackvm::apply(
        max_shape, [&](gsl::span<const size_t> index) -> result<void> {
            ptr[offset(max_stride, index)] = init_value;
            return ok();
        }));

    // collact all max indices
    std::unordered_map<size_t, size_t> out_map;
    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        size_t src_idx = offset(in_strides, index);
        const auto src = input[src_idx];
        auto out_idx =
            offset(max_stride,
                   kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = ptr[out_idx];
        auto ret = cmp(src, dst);
        if (ret) {
            out_map[out_idx] = src_idx;
            dst = src;
        }
        return ok();
    }));

    // update output with max idx as 1
    memset(static_cast<void *>(output), 0, compute_size(in_shape) * sizeof(T));
    for (auto e : out_map) {
        output[e.second] = static_cast<T>(1);
    }

    return ok();
}

#define HARDMAX_IMPL(_ty)                                                      \
    return hardmax_impl(IN_CAST(_ty, input), in_shape, in_strides,             \
                        OUT_CAST(_ty, output), axis);

result<void> hardmax_impl(typecode_t typecode, const gsl::byte *input,
                          gsl::span<const size_t> in_shape,
                          gsl::span<const size_t> in_strides, gsl::byte *output,
                          int32_t axis) noexcept {
    TYPE_SELECT(typecode, HARDMAX_IMPL)} // namespace

result<value_t> nncase::kernels::stackvm::hardmax(
    value_t input, value_t axis, value_t output,
    [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    try_output(out_mem, output, input_tensor->dtype(), input_tensor->shape());
    try_positive_axis(axis_value, axis, input_tensor);
    try_typecode(typecode, input_tensor);
    try_(hardmax_impl(typecode, input_mem, input_tensor->shape(),
                      input_tensor->strides(), out_mem, axis_value));

    return ok(output);
}
