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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class T>
result<void> onehot_impl(const int32_t *indices, T *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, NNCASE_UNUSED size_t depth, T off_value, T on_value,
    size_t axis, onehot_mode_t mode, NNCASE_UNUSED kernel_context &context)
{
    return apply(out_shape, [&](const runtime_shape_t &out_index) -> result<void>
        {
            runtime_shape_t indices_index(indices_shape.size());
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
            auto cur_axis_index = static_cast<int32_t>(out_index[axis]);
            if (indices_v < 0 && mode == onehot_process_neg)
            {
                out_v = (indices_v + static_cast<int32_t>(out_shape[axis])) == cur_axis_index ? on_value : off_value;
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
        return onehot_impl(indices, reinterpret_cast<type *>(output), indices_shape, out_shape, out_strides, \
            *reinterpret_cast<type *>(depth), *reinterpret_cast<type *>(off_value), *reinterpret_cast<type *>(on_value), axis, mode, context);

result<void> reference::onehot(datatype_t type, const int32_t *indices, gsl::byte *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, gsl::byte *depth, gsl::byte *off_value, gsl::byte *on_value, size_t axis, onehot_mode_t mode, kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, ONEHOT_IMPL);
}
