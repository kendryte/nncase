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
result<void> copy_impl(const T *src, T *dest, const runtime_shape_t &shape, const runtime_shape_t &src_strides,
    const runtime_shape_t &dest_strides) noexcept
{
    auto src_view = xt::adapt(src, runtime::get_bytes(to_datatype<T>(), src_strides), xt::no_ownership(), shape, src_strides);
    auto dest_view = xt::adapt(dest, runtime::get_bytes(to_datatype<T>(), dest_strides), xt::no_ownership(), shape, dest_strides);
    std::copy(src_view.begin(), src_view.end(), dest_view.begin());
    return ok();
}
}

#define COPY_IMPL(size, type) \
    case size:                \
        return copy_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides)

result<void> reference::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides) noexcept
{
    switch (runtime::get_bytes(type))
    {
        COPY_IMPL(1, uint8_t);
        COPY_IMPL(2, uint16_t);
        COPY_IMPL(4, uint32_t);
        COPY_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
