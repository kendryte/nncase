/* Copyright 2019-2020 Canaan Inc.
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
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
runtime_shape_t get_padded_shape(const runtime_shape_t &in_shape, const runtime_paddings_t &paddings)
{
    runtime_shape_t out_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++)
        out_shape[i] = (size_t)((int32_t)in_shape[i] + paddings[i].sum());
    return out_shape;
}

runtime_shape_t get_in_index(const runtime_shape_t &index, const runtime_shape_t &in_shape,
    const runtime_paddings_t &paddings, pad_mode_t mode, bool &pad_element)
{
    runtime_shape_t in_index(index.size());
    pad_element = false;
    for (size_t i = 0; i < index.size(); i++)
    {
        auto &padding = paddings[i];
        if ((int32_t)index[i] < padding.before)
        {
            pad_element = true;
            if (mode == pad_reflect)
                in_index[i] = ((size_t)padding.before - index[i]) % in_shape[i];
            else if (mode == pad_symmetric)
                in_index[i] = ((size_t)padding.before - index[i] - 1) % in_shape[i];
            else if (mode == pad_edge)
                in_index[i] = 0;
        }
        else
        {
            auto cnt_idx = (int32_t)index[i] - padding.before;
            if (cnt_idx > (int32_t)in_shape[i] - 1)
            {
                pad_element = true;
                if (mode == pad_reflect)
                    in_index[i] = ((size_t)cnt_idx - in_shape[i]) % in_shape[i];
                else if (mode == pad_symmetric)
                    in_index[i] = ((size_t)cnt_idx - in_shape[i] + 1) % in_shape[i];
                else if (mode == pad_edge)
                    in_index[i] = in_shape[i] - 1;
            }
            else
            {
                in_index[i] = (size_t)cnt_idx;
            }
        }
    }

    return in_index;
}

template <class T>
result<void> pad_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, T pad_value) noexcept
{
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        bool pad_element = false;
        auto in_index = get_in_index(index, in_shape, paddings, mode, pad_element);
        T value;
        if (!pad_element || mode != pad_constant)
            value = input[offset(in_strides, in_index)];
        else
            value = pad_value;
        output[offset(out_strides, index)] = value;
        return ok();
    });
}
}

#define PAD_IMPL(size, type) \
    case size:               \
        return pad_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, out_shape, in_strides, out_strides, paddings, mode, pad_value.as<type>())

result<void> reference::pad(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, const scalar &pad_value) noexcept
{
    auto out_shape = get_padded_shape(in_shape, paddings);
    switch (runtime::get_bytes(type))
    {
        PAD_IMPL(1, uint8_t);
        PAD_IMPL(2, uint16_t);
        PAD_IMPL(4, uint32_t);
    default:
        return err(std::errc::not_supported);
    }
}
