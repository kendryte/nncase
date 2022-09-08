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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
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
                         const dims_t &in_a_shape,
                         const dims_t &in_b_shape) noexcept {
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
                         const dims_t &in_a_shape,
                         const dims_t &in_b_shape) noexcept {
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
            try_(matmul_unit_impl(
                input_a + an * ab_size + ac * a_unit_size,
                input_b + bn * bb_size + bc * b_unit_size,
                output + n * ob_size + c * out_unit_size,
                dims_t{new_a_shape[2], new_a_shape[3]},
                dims_t{new_b_shape[2], new_b_shape[3]}));
        }
    }
    return ok();
}

template result<void> matmul_impl<float>(const float *input_a,
                                         const float *input_b, float *output,
                                         const dims_t &in_a_shape,
                                         const dims_t &in_b_shape) noexcept;

result<dims_t> infer_shape(const dims_t &lhs_shape, const dims_t &rhs_shape) {
    if (lhs_shape.size() == 2 && rhs_shape.size() == 2) {
        auto new_shape = dims_t{lhs_shape[0], rhs_shape[1]};
        return ok(new_shape);
    }
    auto big_shape = lhs_shape.size() > rhs_shape.size()
        ? lhs_shape
        : rhs_shape;
    auto new_shape = dims_t(big_shape.begin(), big_shape.begin() + (big_shape.size() - 2));
    new_shape.push_back(lhs_shape[lhs_shape.size() - 2]);
    new_shape.push_back(rhs_shape.back());
    return ok(new_shape);
}

#define MATMUL_IMPL(_ty) return matmul_impl(IN_CAST(_ty, input_a), \
    IN_CAST(_ty, input_b), OUT_CAST(_ty, output), in_a_shape, in_b_shape);

result<void> matmul_impl(typecode_t typecode, const gsl::byte *input_a, const gsl::byte *input_b, gsl::byte *output,
                         const dims_t &in_a_shape,
                         const dims_t &in_b_shape) noexcept {
    TYPE_SELECT(typecode, MATMUL_IMPL);
}
} // namespace

result<value_t>
nncase::kernels::stackvm::mat_mul(value_t lhs, value_t rhs, value_t output,
                                  [[maybe_unused]] kernel_context &context) {
    try_input(lhs_mem, lhs);
    try_input(rhs_mem, rhs);
    try_var(out_shape, infer_shape(lhs_tensor->shape(), rhs_tensor->shape()));
    try_output(out_mem, output, lhs_tensor->dtype(), out_shape);
    try_typecode(typecode, lhs_tensor);
    try_(matmul_impl(typecode, lhs_mem, rhs_mem, out_mem, lhs_tensor->shape(),
                     rhs_tensor->shape()));
    return ok(output);
}
