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

namespace {
template <class TReducer, class TOutput, class T>
result<void> reduce_arg_impl(TReducer &&reducer, T init_value, const T *input,
                             TOutput *output, const dims_t &in_shape,
                             const dims_t &out_shape,
                             const strides_t &in_strides,
                             const strides_t &out_strides, const dims_t &axes,
                             bool keep_dims, bool select_last_idx,
                             NNCASE_UNUSED kernel_context &context) noexcept {
    const float epsilon = 0.000001f;

    // init with init_value
    std::unique_ptr<T[]> ptr(new T[compute_size(out_shape)]);
    try_(apply(out_shape, [&](const dims_t &index) -> result<void> {
        ptr[offset(out_strides, index)] = init_value;
        return ok();
    }));

    // collect all min/max indices
    std::unordered_map<size_t, std::vector<TOutput>> out_map;
    try_(apply(in_shape, [&](const dims_t &index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides,
                   kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = ptr[out_idx];
        auto ret = reducer(src, dst);
        if (ret) {
            out_map[out_idx].clear();
            out_map[out_idx].push_back(index[axes[0]]);
            dst = src;
        } else if constexpr (std::is_same_v<T, float>) {
            if (fabs(src - dst) < epsilon) {
                out_map[out_idx].push_back(index[axes[0]]);
            }
        }
        return ok();
    }));

    // update min/max idx
    try_(apply(out_shape, [&](const dims_t &index) -> result<void> {
        auto out_idx = offset(out_strides, index);
        output[out_idx] = select_last_idx ? out_map[out_idx].back()
                                          : out_map[out_idx].front();
        return ok();
    }));
    return ok();
}
} // namespace

#define REDUCE_ARG_IMPL(_ty)                                                   \
    {                                                                          \
        auto out_shape =                                                       \
            kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);     \
        switch (op) {                                                          \
        case reduce_arg_op_t::arg_min:                                         \
            return output_typecode == dt_int32                                 \
                       ? reduce_arg_impl(                                      \
                             [](_ty a, _ty b) { return a < b; },               \
                             std::numeric_limits<_ty>::max(),                  \
                             IN_CAST(_ty, input), OUT_CAST(int32_t, output),   \
                             in_shape, out_shape, in_strides, out_strides,     \
                             axes, keep_dims, select_last_idx, context)        \
                       : reduce_arg_impl(                                      \
                             [](_ty a, _ty b) { return a < b; },               \
                             std::numeric_limits<_ty>::max(),                  \
                             IN_CAST(_ty, input), OUT_CAST(int64_t, output),   \
                             in_shape, out_shape, in_strides, out_strides,     \
                             axes, keep_dims, select_last_idx, context);       \
        case reduce_arg_op_t::arg_max:                                         \
            return output_typecode == dt_int32                                 \
                       ? reduce_arg_impl(                                      \
                             [](_ty a, _ty b) { return a > b; },               \
                             std::numeric_limits<_ty>::lowest(),               \
                             IN_CAST(_ty, input), OUT_CAST(int32_t, output),   \
                             in_shape, out_shape, in_strides, out_strides,     \
                             axes, keep_dims, select_last_idx, context)        \
                       : reduce_arg_impl(                                      \
                             [](_ty a, _ty b) { return a > b; },               \
                             std::numeric_limits<_ty>::lowest(),               \
                             IN_CAST(_ty, input), OUT_CAST(int64_t, output),   \
                             in_shape, out_shape, in_strides, out_strides,     \
                             axes, keep_dims, select_last_idx, context);       \
        default:                                                               \
            return err(std::errc::not_supported);                              \
        }                                                                      \
    }

result<void> reduce_arg_impl(typecode_t input_typecode,
                             typecode_t output_typecode, reduce_arg_op_t op,
                             const gsl::byte *input, gsl::byte *output,
                             const dims_t &in_shape,
                             const strides_t &in_strides,
                             const strides_t &out_strides, const dims_t &axes,
                             bool keep_dims, bool select_last_idx,
                             kernel_context &context) noexcept {
    TYPE_SELECT(input_typecode, REDUCE_ARG_IMPL);
}

dims_t infer_shape(const dims_t &in_shape, int32_t axis, bool keep_dims) {
    auto new_shape = in_shape;
    if (keep_dims) {
        new_shape[axis] = 1;
    } else {
        new_shape.erase(new_shape.begin() + axis);
    }
    return new_shape;
}

result<value_t> nncase::kernels::stackvm::reduce_arg(
    reduce_arg_op_t reduce_arg_op, typecode_t dest_type, value_t input,
    value_t axis, value_t keep_dims, value_t select_last_index, value_t output,
    kernel_context &context) {
    try_input(in_mem, input);
    try_typecode(input_typecode, input_tensor);
    try_positive_axis(axis_value, axis, input_tensor);
    try_to_scalar(keep_dims_value, keep_dims, bool);
    try_to_scalar(select_last_index_value, select_last_index, bool);
    auto out_shape =
        infer_shape(input_tensor->shape(), axis_value, keep_dims_value);
    try_output(out_mem, output, dest_type, out_shape);
    auto axes = dims_t{axis_value};
    try_(reduce_arg_impl(input_typecode, dest_type, reduce_arg_op, in_mem,
                         out_mem, input_tensor->shape(),
                         input_tensor->strides(), output_tensor->strides(),
                         axes, keep_dims_value, select_last_index_value,
                         context));
    return ok(output);
}
