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
template <class T>
result<void> broadcast_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_index = kernels::detail::get_reduced_offset(index, in_shape);
        output[offset(out_strides, index)] = input[offset(in_strides, in_index)];
        return ok();
    });
}
}

#define BROADCAST_IMPL(size, type) \
    case size:                     \
        return broadcast_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_shape, out_strides, context)

result<void> reference::broadcast(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    switch (runtime::get_bytes(type))
    {
        BROADCAST_IMPL(1, uint8_t);
        BROADCAST_IMPL(2, uint16_t);
        BROADCAST_IMPL(4, uint32_t);
        BROADCAST_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
