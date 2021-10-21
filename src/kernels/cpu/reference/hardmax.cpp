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
#include <unordered_map>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::hardmax<float>(const float *input, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    float *output, NNCASE_UNUSED const runtime_shape_t &out_strides, int32_t axis) noexcept;

template <typename T>
result<void> reference::hardmax(const T *input, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    T *output, NNCASE_UNUSED const runtime_shape_t &out_strides, int32_t axis) noexcept

{
    // init with init_value
    auto cmp = [](T a, T b) { return a > b; };
    T init_value = std::numeric_limits<T>::min();
    bool keep_dims = true;
    runtime_shape_t axes { static_cast<size_t>(axis) };
    auto max_shape = kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    auto max_stride = get_default_strides(max_shape);
    std::unique_ptr<T[]> ptr(new T[compute_size(max_shape)]);
    try_(reference::apply(max_shape, [&](const runtime_shape_t &index) -> result<void> {
        ptr[offset(max_stride, index)] = init_value;
        return ok();
    }));

    // collact all max indices
    std::unordered_map<size_t, size_t> out_map;
    try_(reference::apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        size_t src_idx = offset(in_strides, index);
        const auto src = input[src_idx];
        auto out_idx = offset(max_stride, kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = ptr[out_idx];
        auto ret = cmp(src, dst);
        if (ret)
        {
            out_map[out_idx] = src_idx;
            dst = src;
        }
        return ok();
    }));

    // update output with max idx as 1
    memset(static_cast<void *>(output), 0, compute_size(in_shape) * sizeof(T));
    for (auto e : out_map)
    {
        output[e.second] = static_cast<T>(1);
    }

    return ok();
}
