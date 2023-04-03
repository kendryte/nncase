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
#include "../shape_infer.h"
#include "ref_ops.h"
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {

std::vector<size_t> concat(const std::vector<std::vector<size_t>> &containers) {
    std::vector<size_t> result;
    for (size_t i = 0; i < containers.size(); ++i) {
        result.insert(result.end(), containers[i].begin(), containers[i].end());
    }
    return result;
}

template <typename T, typename Fn> std::vector<T> range_exec(int end, Fn &&f) {
    std::vector<T> vec;
    for (int i = 0; i < end; ++i) {
        vec.push_back((T)f(i));
    }
    return vec;
}

template <typename Fn> std::vector<size_t> range_exec_flatten(int end, Fn &&f) {
    auto result = range_exec<std::vector<size_t>>(end, f);
    auto flatten = concat(result);
    return flatten;
}

template <class T>
result<void>
space_to_batch_impl(datatype_t dt, const T *input, T *output,
                    const dims_t &in_shape, const dims_t &block_shape,
                    const paddings_t &paddings, const strides_t &in_strides,
                    [[maybe_unused]] const dims_t &out_shape,
                    [[maybe_unused]] const strides_t &out_strides,
                    NNCASE_UNUSED kernel_context &context) noexcept {
    auto spatial_size = block_shape.size();
    auto remain_shape_size = in_shape.size() - spatial_size - 1;
    auto new_paddings = paddings_t((1 + spatial_size + remain_shape_size));
    for (size_t i = 0; i < spatial_size; ++i) {
        new_paddings[1 + i] = paddings[i];
    }
    auto pad_out_shape =
        kernels::stackvm::pad_infer_shape(in_shape, new_paddings);
    auto pad_output = std::make_unique<float[]>(compute_size(pad_out_shape));
    auto pad_out_strides = get_default_strides(pad_out_shape);
    int64_t pad_value = 0;
    try_(kernels::stackvm::reference::pad(
        dt, IN_BYTE_CAST(input), OUT_BYTE_CAST(pad_output.get()), in_shape,
        in_strides, pad_out_strides, new_paddings,
        nncase::runtime::stackvm::pad_mode_t::constant,
        IN_BYTE_CAST(&pad_value)));

    auto batch_shape1 = std::vector{pad_out_shape[0]};
    auto spatial_shape1 = range_exec_flatten(spatial_size, [&](auto &&i) {
        return std::vector{pad_out_shape[i + 1] / block_shape[i],
                           block_shape[i]};
    });

    auto remain_shape1 = range_exec<size_t>(remain_shape_size, [&](auto &&i) {
        return pad_out_shape[1 + spatial_size + i];
    });

    //    auto remain_shape1 = concat(remain_shape1_tmp);
    auto reshapeed_shape1 = concat(std::vector<std::vector<size_t>>{
        batch_shape1, spatial_shape1, remain_shape1});
    auto perm1 =
        range_exec<size_t>(spatial_size, [&](auto &&i) { return i * 2 + 2; });
    auto perm2 = std::vector<size_t>{0};
    auto perm3 =
        range_exec<size_t>(spatial_size, [&](auto &&i) { return i * 2 + 1; });
    auto perm4 = range_exec<size_t>(
        remain_shape_size, [&](auto &&i) { return i + spatial_size * 2 + 1; });
    auto perms = std::vector<std::vector<size_t>>{perm1, perm2, perm3, perm4};
    auto perm = concat(perms);
    auto reshapeed_shape1_dims =
        dims_t(reshapeed_shape1.begin(), reshapeed_shape1.end());
    auto perm_dims = dims_t(perm.begin(), perm.end());
    auto tr_out_shape = transpose_infer_shape(reshapeed_shape1_dims, perm_dims);
    auto tr_out_stride = get_default_strides(tr_out_shape);
    try_(kernels::stackvm::reference::transpose(
        dt, IN_BYTE_CAST(pad_output.get()), OUT_BYTE_CAST(output),
        reshapeed_shape1_dims, perm_dims,
        get_default_strides(reshapeed_shape1_dims), tr_out_stride));
    return ok();
}
} // namespace

#define SPACE_TO_BATCH_IMPL(size, type)                                        \
    case size:                                                                 \
        return space_to_batch_impl(dt, reinterpret_cast<const type *>(input),  \
                                   reinterpret_cast<type *>(output), in_shape, \
                                   block_shape, paddings, in_strides,          \
                                   out_shape, out_strides, context)

result<void> nncase::kernels::stackvm::reference::space_to_batch(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    const dims_t &in_shape, const dims_t &block_shape,
    const paddings_t &paddings, const strides_t &in_strides,
    const dims_t &out_shape, const strides_t &out_strides,
    NNCASE_UNUSED kernel_context &context) {
    switch (runtime::get_bytes(dt)) {
        SPACE_TO_BATCH_IMPL(1, uint8_t);
        SPACE_TO_BATCH_IMPL(2, uint16_t);
        SPACE_TO_BATCH_IMPL(4, uint32_t);
        SPACE_TO_BATCH_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}