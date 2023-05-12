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
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class IndicesT>
result<void> scatter_nd_impl(const T *input, T *output, const dims_t &in_shape,
                             [[maybe_unused]] const IndicesT *indices,
                             const dims_t &indices_shape,
                             [[maybe_unused]] const T *updates,
                             [[maybe_unused]] const dims_t &updates_shape,
                             NNCASE_UNUSED kernel_context &context) noexcept {

    std::copy(input, input + compute_size(in_shape), output);
    auto update_indices = indices_shape;
    update_indices.pop_back();
    auto indices_strides = get_default_strides(indices_shape);
    indices_strides.pop_back();

    auto in_strides = get_default_strides(in_shape);
    for (auto i = 0; i <  in_shape.size() - indices_shape.back(); ++i) {
        in_strides.pop_back();
    }

    auto updates_strides = get_default_strides(updates_shape);
    for (auto i = 0; i < updates_shape.size() - update_indices.size(); ++i) {
        updates_strides.pop_back();
    }
    auto updates_size = 1;
    for (auto i = update_indices.size(); i < updates_shape.size(); ++i) {
        updates_size *= updates_shape[i];
    }

    // auto data_size = 1;
    // for (auto i = indices_shape.back(); i < in_shape.size(); ++i) {
    //     data_size *= in_shape[i];
    // }
    return apply(
        (const dims_t &)update_indices,
        [&]([[maybe_unused]] const dims_t &idx) -> result<void> {
            auto updates_begin = updates + offset(updates_strides, idx);

            auto data_indices_begin = indices + offset(indices_strides, idx);
            dims_t data_indices_dim;
            for (auto i = 0; i < indices_shape.back(); ++i) {
                data_indices_dim.push_back(*(data_indices_begin + i));
            }

            auto data_begin = output + offset(in_strides, data_indices_dim);

            std::copy(updates_begin, updates_begin + updates_size, data_begin);

            return ok();
        });
}
} // namespace

#define SCATTER_ND_IMPL(size, type)                                            \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return scatter_nd_impl(reinterpret_cast<const type *>(input),      \
                                   reinterpret_cast<type *>(output), in_shape, \
                                   indices_value, indices_shape,               \
                                   reinterpret_cast<const type *>(updates),    \
                                   updates_shape, context);                    \
        });

result<void> nncase::kernels::stackvm::reference::scatter_nd(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    const dims_t &in_shape, datatype_t indices_type, const gsl::byte *indices,
    const dims_t &indices_shape, const gsl::byte *updates,
    const dims_t &updates_shape, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, SCATTER_ND_IMPL);
}