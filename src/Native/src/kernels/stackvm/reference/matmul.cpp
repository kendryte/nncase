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

namespace {
template <typename T>
result<void> matmul_unit_impl(const T *input_a, const T *input_b, T *output,
                              gsl::span<const size_t> in_a_shape,
                              gsl::span<const size_t> in_b_shape) noexcept {
    int32_t a_rows = static_cast<int32_t>(in_a_shape[0]);
    int32_t a_cols = static_cast<int32_t>(in_a_shape[1]);
    int32_t b_cols = static_cast<int32_t>(in_b_shape[1]);

    for (int32_t oy = 0; oy < a_rows; oy++) {
        for (int32_t ox = 0; ox < b_cols; ox++) {
            T value = 0;

            for (int32_t i = 0; i < a_cols; i++) {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                value += a * b;
            }

            output[oy * b_cols + ox] = value;
        }
    }

    return ok();
}

template <typename T>
result<void> matmul_impl(const T *input_a, const T *input_b, T *output,
                         gsl::span<const size_t> in_a_shape,
                         gsl::span<const size_t> in_b_shape) noexcept {
    auto new_a_shape = to_4d(in_a_shape);
    auto new_b_shape = to_4d(in_b_shape);
    auto a_unit_size = new_a_shape[2] * new_a_shape[3];
    auto b_unit_size = new_b_shape[2] * new_b_shape[3];
    auto out_unit_size = new_a_shape[2] * new_b_shape[3];

    auto batches = std::max(new_a_shape[0], new_b_shape[0]);
    auto channels = std::max(new_a_shape[1], new_b_shape[1]);
    auto ab_size = a_unit_size * new_a_shape[1];
    auto bb_size = b_unit_size * new_b_shape[1];
    auto ob_size = out_unit_size * channels;
    for (size_t n = 0; n < batches; ++n) {
        auto an = new_a_shape[0] == 1 ? 0 : n;
        auto bn = new_b_shape[0] == 1 ? 0 : n;
        for (size_t c = 0; c < channels; ++c) {
            auto ac = new_a_shape[1] == 1 ? 0 : c;
            auto bc = new_b_shape[1] == 1 ? 0 : c;
            try_(matmul_unit_impl(input_a + an * ab_size + ac * a_unit_size,
                                  input_b + bn * bb_size + bc * b_unit_size,
                                  output + n * ob_size + c * out_unit_size,
                                  dims_t{new_a_shape[2], new_a_shape[3]},
                                  dims_t{new_b_shape[2], new_b_shape[3]}));
        }
    }
    return ok();
}

template result<void>
matmul_impl<float>(const float *input_a, const float *input_b, float *output,
                   gsl::span<const size_t> in_a_shape,
                   gsl::span<const size_t> in_b_shape) noexcept;

#define MATMUL_IMPL(_ty)                                                       \
    return matmul_impl(IN_CAST(_ty, input_a), IN_CAST(_ty, input_b),           \
                       OUT_CAST(_ty, output), in_a_shape, in_b_shape);

} // namespace

result<void> nncase::kernels::stackvm::reference::matmul(
    typecode_t typecode, const gsl::byte *input_a, const gsl::byte *input_b,
    gsl::byte *output, gsl::span<const size_t> in_a_shape,
    gsl::span<const size_t> in_b_shape,
    [[maybe_unused]] kernel_context &context) noexcept {
    TYPE_SELECT(typecode, MATMUL_IMPL);
}