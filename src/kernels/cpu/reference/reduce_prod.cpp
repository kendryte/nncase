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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template NNCASE_API result<void> reference::reduce_prod<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, const runtime_shape_t &axes) noexcept;

template <typename T>
result<void> reference::reduce_prod(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, const runtime_shape_t &axes) noexcept
{
    std::vector<size_t> xt_in_shape(in_shape.begin(), in_shape.end());
    auto in_array = xt::adapt(input, compute_size(xt_in_shape), xt::no_ownership(), xt_in_shape);

    std::vector<size_t> xt_out_shape(out_shape.begin(), out_shape.end());
    auto out_array = xt::adapt(output, compute_size(xt_out_shape), xt::no_ownership(), xt_out_shape);

    std::vector<size_t> xt_axes(axes.begin(), axes.end());
    out_array = xt::prod(in_array, xt_axes);

    return ok();
}
