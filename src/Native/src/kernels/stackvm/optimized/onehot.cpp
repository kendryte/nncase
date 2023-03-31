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
#include "opt_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <sstream>
#include <string>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
// if constexpr can be used in C++17
// but k210 runtime only support C++14
template <class T> void memset_(T *output, size_t output_size, T off_value) {
    std::fill_n(output, output_size, off_value);
}

template <>
void memset_(int32_t *output, size_t output_size, int32_t off_value) {
    memset(output, off_value, output_size);
}

template <class T, class IndicesT>
result<void> one_hot_impl(const IndicesT *indices, T *output,
                          const dims_t &indices_shape, const dims_t &out_shape,
                          NNCASE_UNUSED const dims_t &out_strides,
                          NNCASE_UNUSED size_t depth, T off_value, T on_value,
                          size_t axis, runtime::stackvm::one_hot_mode_t mode,
                          NNCASE_UNUSED kernel_context &context) {
    auto output_size = compute_size(out_shape);
    memset_(output, output_size, off_value);
    auto indices_size = compute_size(indices_shape);

    size_t out_size =
        std::accumulate(indices_shape.begin(), indices_shape.begin() + axis, 1,
                        std::multiplies<size_t>{});
    size_t inner_size =
        std::accumulate(indices_shape.begin() + axis, indices_shape.end(), 1,
                        std::multiplies<size_t>{});
    auto indices_dims = indices_shape.size();
    auto onehot_dims = indices_dims - axis;

    auto neg_max_len = static_cast<int32_t>(out_shape[axis]);

    auto set_output = [&](auto indices_v, auto offset) {
        if (indices_v < 0) {
            if (mode == runtime::stackvm::one_hot_mode_t::process_neg) {
                indices_v += neg_max_len;
            } else {
                return;
            }
        }
        output[indices_v * inner_size + offset] = on_value;
    };

    if (onehot_dims == 0) {
        // a depth is a line, set a value per line
        // next line
        for (size_t i = 0; i < indices_size; ++i, ++indices) {
            set_output(*indices, 0);
            output += depth;
        }
    } else if (onehot_dims == 1) {
        const auto x_size = indices_shape[indices_shape.size() - 1];
        // next indices_inner_size
        for (size_t i = 0; i < out_size; ++i) {
            for (size_t x = 0; x < x_size; ++x, ++indices) {
                set_output(*indices, x);
            }
            output += inner_size * depth;
        }
    } else if (onehot_dims == 2) {
        const auto y_size = indices_shape[indices_shape.size() - 2];
        const auto x_size = indices_shape[indices_shape.size() - 1];
        for (size_t i = 0; i < out_size; ++i) {
            for (size_t y = 0; y < y_size; ++y) {
                for (size_t x = 0; x < x_size; ++x, ++indices) {
                    set_output(*indices, y * x_size + x);
                }
            }
            output += inner_size * depth;
        }
    } else if (onehot_dims == 3) {
        const auto c_size = indices_shape[indices_shape.size() - 3];
        const auto y_size = indices_shape[indices_shape.size() - 2];
        const auto x_size = indices_shape[indices_shape.size() - 1];
        const auto y_block_size = out_strides[out_strides.size() - 2];
        const auto c_block_size = out_strides[out_strides.size() - 3];
        for (size_t i = 0; i < out_size; ++i) {
            for (size_t c = 0; c < c_size; ++c) {
                for (size_t y = 0; y < y_size; ++y) {
                    for (size_t x = 0; x < x_size; ++x, ++indices) {
                        set_output(*indices,
                                   c * c_block_size + y * y_block_size + x);
                    }
                }
            }
            output += inner_size * depth;
        }
    } else {
        return err(std::errc::result_out_of_range);
    }
    return ok();
}
} // namespace

#define ONEHOT_IMPL(size, type)                                                \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return one_hot_impl(                                               \
                indices_value, reinterpret_cast<type *>(output),               \
                indices_shape, out_shape, out_strides, depth,                  \
                reinterpret_cast<type *>(values)[0],                           \
                reinterpret_cast<type *>(values)[1], axis, mode, context);     \
        });

result<void> optimized::one_hot(
    datatype_t type, datatype_t indices_type, const gsl::byte *indices,
    gsl::byte *output, const dims_t &indices_shape, const dims_t &out_shape,
    const dims_t &out_strides, size_t depth, gsl::byte *values, size_t axis,
    runtime::stackvm::one_hot_mode_t mode, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, ONEHOT_IMPL);
}
