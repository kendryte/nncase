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

#include "../../gsl-lite.hpp"
#include <apply.h>
#include <runtime_utils.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase::runtime::cpu;
namespace kernels {
namespace {
#ifdef __riscv_vector
template <class T>
void clamp_rvv_impl(const T *input, T min, T max, T *output,
                    gsl::span<const size_t> in_shape,
                    gsl::span<const size_t> in_strides,
                    gsl::span<const size_t> out_strides) {
    auto [new_in_shape, new_in_stride] = to_nd(in_shape, in_strides, 5);
    auto [new_out_shape, new_out_stride] = to_nd(in_shape, out_strides, 5);
    for (size_t n = 0; n < new_in_shape[0]; ++n) {
        for (size_t c = 0; c < new_in_shape[1]; ++c) {
            for (size_t h = 0; h < new_in_shape[2]; ++h) {
                for (size_t w = 0; w < new_in_shape[3]; ++w) {
                    const T *in_ptr = input + n * new_in_stride[0] +
                                      c * new_in_stride[1] + h * new_in_stride[2] +
                                      w * new_in_stride[3];
                    T *out_ptr = output + n * new_out_stride[0] +
                                       c * new_out_stride[1] + h * new_out_stride[2] +
                                       w * new_out_stride[3];
                    size_t vl;
                    for (size_t i = new_in_shape[4]; i > 0; i -= vl) {
                        vl = vsetvl_e32m8(i);
                        vfloat32m8_t vx = vle32_v_f32m8(in_ptr, vl);
                        vx = vfmax_vf_f32m8(vx, min, vl);
                        vx = vfmin_vf_f32m8(vx, max, vl);
                        vse32_v_f32m8(out_ptr, vx, vl);
                        in_ptr += vl;
                        out_ptr += vl;
                    }
                }
            }
        }
    }
    return;
}
#else
template <class T>
void clamp_native_impl(const T *input, T min, T max, T *output,
                       gsl::span<const size_t> in_shape,
                       gsl::span<const size_t> in_strides,
                       gsl::span<const size_t> out_strides) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto v = input[offset(index, in_strides)];
        output[offset(index, out_strides)] = static_cast<T>(
            std::min(std::max(static_cast<float>(v), static_cast<float>(min)),
                     static_cast<float>(max)));
        return;
    });
}
#endif
} // namespace

template <class T>
void clamp(const T *input, T *output, T min, T max,
           gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_strides) {
#ifdef __riscv_vector
    clamp_rvv_impl(input, min, max, output, in_shape, in_strides, out_strides);
#else
    clamp_native_impl(input, min, max, output, in_shape, in_strides,
                      out_strides);
#endif
}
} // namespace kernels