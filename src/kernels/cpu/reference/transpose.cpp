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
result<void> transpose_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        runtime_shape_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++)
            out_index[i] = index[perm[i]];
        output[offset(out_strides, out_index)] = input[offset(in_strides, index)];
        return ok();
    });
}
}

#define TRANSPOSE_IMPL(size, type) \
    case size:                     \
        return transpose_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), in_shape, perm, in_strides, out_strides)

result<void> reference::transpose(datatype_t type, const gsl::byte *src, gsl::byte *dest, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    switch (runtime::get_bytes(type))
    {
        TRANSPOSE_IMPL(1, uint8_t);
        TRANSPOSE_IMPL(2, uint16_t);
        TRANSPOSE_IMPL(4, uint32_t);
        TRANSPOSE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
