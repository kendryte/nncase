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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
result<void> lut1d_impl(const uint8_t *input, const uint8_t *table, uint8_t *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return apply(shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = table[v];
        return ok();
    });
}
}

result<void> reference::lut1d(datatype_t type, const gsl::byte *input, const gsl::byte *table, gsl::byte *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, NNCASE_UNUSED const scalar &min, NNCASE_UNUSED const scalar &max) noexcept
{
    if (type != dt_uint8)
        return err(std::errc::not_supported);
    return lut1d_impl(reinterpret_cast<const uint8_t *>(input), reinterpret_cast<const uint8_t *>(table),
        reinterpret_cast<uint8_t *>(output), shape, in_strides, out_strides);
}
