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
#include <iostream>
#include <limits>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TReducer>
result<void> reduce_arg_impl(TReducer &&reducer, float init_value,
    const float *input, int64_t *output,
    const runtime_shape_t &in_shape,const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides,  const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims, NNCASE_UNUSED bool select_last_idx, NNCASE_UNUSED kernel_context &context) noexcept
{
    const float epsilon = 0.000001f;
    std::cout << "in_strides :" << std::endl;
    for (auto i : in_strides)
        std::cout << i << std::endl;

    std::cout << "out_strides :" << std::endl;
    for (auto i : out_strides)
        std::cout << i << std::endl;

    std::cout << "axes :" << std::endl;
    for (auto i : axes)
        std::cout << i << std::endl;

    std::cout << "keep_dims = " << keep_dims << ", select_last_idx = " << select_last_idx << std::endl;

    // init with init_value
    auto size = compute_size(out_shape);
    std::unique_ptr<float[]> ptr(new float[size]);
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        ptr[offset(out_strides, index)] = init_value;
        return ok();
    }));


    // collact all max/min indices
    std::unordered_map<size_t, std::vector<size_t>> out_map;
    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        // std::cout << "index shape: " << std::endl;
        // for (auto i : index)
        //     std::cout << i << std::endl;

        auto in_idx = offset(in_strides, index);
        const auto src = input[in_idx];
        const auto out_index = kernels::detail::get_reduced_offset(index, axes, keep_dims);
        auto out_idx = offset(out_strides, out_index);
        auto &dst = ptr[out_idx];
        std::cout << "in_idx = " << in_idx << ", src = " << src << ", out_idx = " << out_idx << ", dst = " << dst << std::endl;
        auto ret = reducer(src, dst);
        if (ret)
        {
            out_map[out_idx].clear();
            out_map[out_idx].push_back(index[axes[0]]);
            dst = src;
            std::cout << "out_idx = " << out_idx << "-> in_idx = " << in_idx << std::endl;
        }
        else if (abs(src - dst) < epsilon)
        {
            out_map[out_idx].push_back(in_idx);
        }
        return ok();
    }));

    // update max/min idx
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto out_idx = offset(out_strides, index);
        auto in_idx = select_last_idx ? out_map[out_idx].back() : out_map[out_idx].front();

        // TODO: how to determine the N/C/H/W index?
        // output[out_idx] = in_idx / in_strides[axes[0]];
        output[out_idx] = in_idx;
        std::cout << "in_idx = " << in_idx << ", output = " << output[out_idx]  << std::endl;
        return ok();
    }));
    return ok();
}
}


result<void> reference::reduce_arg(reduce_arg_op_t op, const float *input, int64_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims, bool select_last_idx, kernel_context &context) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    switch (op)
    {
    case reduce_arg_min:
        return reduce_arg_impl([](float a, float b) { return a < b; }, std::numeric_limits<float>::max(),
            input, output, in_shape, out_shape, in_strides, out_strides, axes, keep_dims, select_last_idx, context);
    case reduce_arg_max:
        return reduce_arg_impl([](float a, float b) { return a > b; }, std::numeric_limits<float>::min(),
            input, output, in_shape, out_shape, in_strides, out_strides, axes, keep_dims, select_last_idx, context);
    default:
        return err(std::errc::not_supported);
    }

}
