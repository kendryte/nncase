///* Copyright 2019-2021 Canaan Inc.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//#include <nncase/kernels/stackvm/ref_ops.h>
//#include <nncase/kernels/stackvm/opt_ops.h>
//#include <nncase/kernels/kernel_utils.h>
//#include <nncase/runtime/runtime_op_utility.h>
//#if __riscv_vector
//#include <riscv_vector.h>
//#endif
//
// using namespace nncase;
// using namespace nncase::runtime;
// using namespace nncase::kernels;
// using namespace nncase::kernels::stackvm;
// using namespace nncase::kernels::stackvm::optimized;
//
// namespace
//{
//#if __riscv_vector
//
//// float
// result<void> optimized_matmul_impl(const float *input_a, const float
// *input_b, const float *bias, float *output,
//    gsl::span<const size_t> in_a_shape, gsl::span<const size_t> in_a_strides,
//    const dims_t &in_b_shape, gsl::span<const size_t> in_b_strides,
//    gsl::span<const size_t> out_shape, const dims_t out_strides,
//    value_range<float> fused_activation) noexcept
//{
//    size_t M = in_a_shape[in_a_shape.size() - 2];
//    size_t K = in_a_shape.back();
//    size_t N = in_b_shape.back();
//
//    // batch
//    size_t batch_a = 1;
//    for (size_t i = 0; i < in_a_shape.size() - 2; i++)
//        batch_a *= in_a_shape[i];
//    size_t step_a = batch_a == 1 ? 0 : in_a_strides[0];
//
//    size_t batch_b = 1;
//    for (size_t i = 0; i < in_b_shape.size() - 2; i++)
//        batch_b *= in_b_shape[i];
//    size_t step_b = batch_b == 1 ? 0 : in_b_strides[0];
//
//    size_t batch_out = 1;
//    for (size_t i = 0; i < out_shape.size() - 2; i++)
//        batch_out *= out_shape[i];
//    size_t step_out = batch_out == 1 ? 0 : out_strides[0];
//
//    size_t batch_max = std::max(batch_a, batch_b);
//
//    const float *p_batch_a = input_a;
//    const float *p_batch_b = input_b;
//    float *p_batch_out = output;
//
//    size_t vl = 0;
//    for (size_t i = 0; i < batch_max; i++)
//    {
//        const float *ptr_a = p_batch_a;
//        float *ptr_out = p_batch_out;
//        for (size_t m = 0; m < M; m++)
//        {
//            const float *pb = p_batch_b;
//            float *pc = ptr_out;
//            const float *pbias = bias;
//            for (size_t n = N; n; n -= vl)
//            {
//                vl = vsetvl_e32m8(n);
//                const float *pa = ptr_a;
//                const float *pb_vl = pb;
//
//                // init acc with bias
//                auto acc = vle32_v_f32m8(pbias, vl);
//
//                for (size_t k = 0; k < K; k++)
//                {
//                    auto vb = vle32_v_f32m8(pb_vl, vl);
//                    acc = vfmacc_vf_f32m8(acc, *pa++, vb, vl);
//                    pb_vl += N;
//                }
//
//                // update acc with act
//                acc = vfmax_vf_f32m8(vfmin_vf_f32m8(acc, fused_activation.max,
//                vl), fused_activation.min, vl);
//
//                vse32_v_f32m8(pc, acc, vl);
//                pb += vl;
//                pc += vl;
//                pbias += vl;
//            }
//            ptr_a += K;
//            ptr_out += N;
//        }
//        p_batch_a += step_a;
//        p_batch_b += step_b;
//        p_batch_out += step_out;
//    }
//
//    return ok();
//}
//#endif
//}
//
//
////template <typename T>
////result<void> optimized::matmul(const T *input_a, const T *input_b, const T
///*bias, T *output, /    gsl::span<const size_t> in_a_shape, gsl::span<const
/// size_t> in_a_strides,
/// gsl::span<const size_t> in_b_shape, /    gsl::span<const size_t>
/// in_b_strides, const dims_t
///&out_shape, gsl::span<const size_t> out_strides, /    value_range<float>
/// fused_activation) noexcept
// result<void> matmul_impl(typecode_t typecode, const gsl::byte *input_a, const
// gsl::byte *input_b, gsl::byte *output,
//                         gsl::span<const size_t> in_a_shape,
//                         gsl::span<const size_t> in_b_shape) noexcept
//{
//#if __riscv_vector
//    return optimized_matmul_impl(input_a, input_b, bias, output, in_a_shape,
//    in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
//    fused_activation);
//#endif
//
//    return kernels::stackvm::reference::matmul(typecode, input_a, input_b,
//    output, in_a_shape, in_b_shape);
//}