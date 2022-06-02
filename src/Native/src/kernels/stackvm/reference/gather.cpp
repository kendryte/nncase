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
template <class T>
result<void> gather_impl(const T *input, T *output, const dims_t &in_shape, const dims_t &out_shape,
    const strides_t &in_strides, const strides_t &out_strides, const int32_t *indices, const dims_t &indices_shape, size_t axis,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(out_shape, [&](const dims_t &out_index) -> result<void> {
        // select batch
        // [out_index.begin(), out_index.begin() + axis]
        dims_t in_index(in_shape.size());
        size_t i_index = 0;
        for (; i_index < static_cast<size_t>(axis); ++i_index)
        {
            in_index[i_index] = out_index[i_index];
        }

        // which index to be used in indices
        dims_t indices_index(out_index.begin() + axis, out_index.begin() + axis + indices_shape.size());
        auto indices_offset = offset(get_default_strides(indices_shape), indices_index);
        // select sub block in dim axis
        in_index[i_index] = indices[indices_offset];
        ++i_index;

        // select position in sub block
        for (auto o_index = axis + indices_shape.size(); o_index < out_index.size(); ++o_index, ++i_index)
        {
            in_index[i_index] = out_index[o_index];
        }
        output[offset(out_strides, out_index)] = input[offset(in_strides, in_index)];
        return ok();
    });
}
}

#define GATHER_IMPL(size, type) \
    case size:                  \
        return gather_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, out_shape, in_strides, out_strides, indices, indices_shape, axis, context);

result<void> gather_impl(datatype_t type, const gsl::byte *input, gsl::byte *output, const dims_t &in_shape, const dims_t &out_shape,
    const strides_t &in_strides, const strides_t &out_strides, const int32_t *indices, const dims_t &indices_shape, size_t axis, kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, GATHER_IMPL);
}

dims_t infer_shape(const dims_t& in_shape, const dims_t& index_shape, int axis) {
    auto in_shape_size = in_shape.size();
    auto index_shape_size = index_shape.size();
    auto new_shape = dims_t(in_shape_size + index_shape_size);
    for (int i = 0; i < axis; ++i) {
        new_shape[i] = in_shape[i];
    }
    for (int i = 0; i < index_shape_size; ++i) {
        new_shape[axis + i] = index_shape[i];
    }
    for (int i = 0; i < in_shape_size - axis - 1; ++i) {
        new_shape[axis + index_shape_size + i] = in_shape[i];
    }
    return new_shape;
}

result<value_t> nncase::kernels::stackvm::gather(value_t input, value_t axis,
                                                 value_t index, value_t output,
                                                 kernel_context &context) {
    try_input(input_mem, input);
    try_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_to_scalar(axis_value, axis, size_t);
    auto out_shape = infer_shape(input_tensor->shape(), index_tensor->shape(), axis_value);
    try_output(out_mem, output, dtype, out_shape);
    auto indices = reinterpret_cast<const int32_t*>(index_mem);
    try_(gather_impl(typecode, input_mem, out_mem,
                        input_tensor->shape(), output_tensor->shape(),
                     input_tensor->strides(), output_tensor->strides(),
                     indices, index_tensor->shape(), axis_value, context));
    return ok(output);
}