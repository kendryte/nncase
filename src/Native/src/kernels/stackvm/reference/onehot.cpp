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
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class T, class IndicesT>
result<void> one_hot_impl(const IndicesT *indices, T *output, const dims_t &indices_shape, const dims_t &out_shape,
    const strides_t &out_strides, NNCASE_UNUSED size_t depth, T off_value, T on_value,
    size_t axis, runtime::stackvm::one_hot_mode_t mode, NNCASE_UNUSED kernel_context &context)
{
    return apply(out_shape, [&](const dims_t &out_index) -> result<void> {
        dims_t indices_index(indices_shape.size());
        for (size_t i = 0; i < axis; ++i)
        {
            indices_index[i] = out_index[i];
        }
        for (size_t i = axis; i < indices_shape.size(); ++i)
        {
            indices_index[i] = out_index[i + 1];
        }
        auto indices_v = indices[offset(get_default_strides(indices_shape), indices_index)];
        T out_v;
        auto cur_axis_index = static_cast<int64_t>(out_index[axis]);
        if (indices_v < 0 && mode == runtime::stackvm::one_hot_mode_t::process_neg)
        {
            out_v = (indices_v + static_cast<int64_t>(out_shape[axis])) == cur_axis_index ? on_value : off_value;
        }
        else
        {
            out_v = indices_v == cur_axis_index ? on_value : off_value;
        }

        output[offset(out_strides, out_index)] = out_v;
        return ok();
    });
}
}

#define ONEHOT_IMPL(size, type)                                                                              \
    case size:                                                                                               \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { return one_hot_impl(indices_value, reinterpret_cast<type *>(output), indices_shape, out_shape, out_strides, \
            depth, reinterpret_cast<type *>(values)[0], reinterpret_cast<type *>(values)[1], axis, mode, context); });

result<void> one_hot_impl(datatype_t type, datatype_t indices_type, const gsl::byte *indices, gsl::byte *output, const dims_t &indices_shape, const dims_t &out_shape,
    const strides_t &out_strides, size_t depth, gsl::byte *values, size_t axis, runtime::stackvm::one_hot_mode_t mode, kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, ONEHOT_IMPL);
}

dims_t infer_shape(const dims_t& indices_shape, size_t depth, size_t axis) {
    auto new_shape = indices_shape;
    new_shape.insert(new_shape.begin() + axis, depth);
    return new_shape;
}

result<value_t>
nncase::kernels::stackvm::one_hot(one_hot_mode_t one_hot_mode, value_t indices,
                                  value_t depth, value_t values, value_t axis,
                                  [[maybe_unused]] value_t on_value, [[maybe_unused]] value_t off_value,
                                  value_t output, kernel_context &context) {
    try_input(onehot_values, values);
    try_var(typecode, to_typecode(values_tensor->dtype()));
    try_to_integer(depth_value, depth);
    try_input(indices_mem, indices);
    try_positive_axis(axis_value, axis, indices_tensor);
    auto out_shape = infer_shape(indices_tensor->shape(), depth_value, axis_value);
    try_output(out_mem, output, typecode, out_shape);

    try_(one_hot_impl(typecode, indices_tensor->dtype(), indices_mem, out_mem, indices_tensor->shape(),
                      output_tensor->shape(), output_tensor->strides(),
                      depth_value, onehot_values,
                      axis_value, one_hot_mode, context));
    return ok(output);
}
