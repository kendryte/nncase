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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#if __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
#if __riscv_vector

size_t matmul_kernel_4x16(size_t maxk, size_t N, const float *A, size_t lda,
    const float *B, size_t ldb, const float *Bias,
    float *C, size_t ldc, bool act, float min,
    float max)
{
    auto vl = vsetvl_e32m4(N);
    const float *pa0 = A;
    const float *pa1 = A + lda;
    const float *pa2 = A + lda * 2;
    const float *pa3 = A + lda * 3;
    const float *pb = B;

    vfloat32m4_t c0_psum = act ? vle32_v_f32m4(Bias, vl) : vfmv_v_f_f32m4(0.f, vl);
    vfloat32m4_t c1_psum;
    vfloat32m4_t c2_psum;
    vfloat32m4_t c3_psum;

    asm volatile("vmv.v.v %[c1_psum], %[c0_psum] \n"
                 "vmv.v.v %[c2_psum], %[c0_psum] \n"
                 "vmv.v.v %[c3_psum], %[c0_psum] \n"
                 :
                 : [c0_psum] "vr"(c0_psum), [c1_psum] "vr"(c1_psum),
                 [c2_psum] "vr"(c2_psum), [c3_psum] "vr"(c3_psum)
                 : "memory");

    for (size_t k = 0; k < maxk; k++)
    {
        asm volatile("vle.v v0, (%[pb]) \n"
                     "flrw fa0, %[pa0], %[k], 2 \n"
                     "flrw fa1, %[pa1], %[k], 2 \n"
                     "vfmacc.vf %[c0_psum], fa0, v0 \n"
                     "vfmacc.vf %[c1_psum], fa1, v0 \n"
                     "flrw fa2, %[pa2], %[k], 2 \n"
                     "flrw fa3, %[pa3], %[k], 2 \n"
                     "vfmacc.vf %[c2_psum], fa2, v0 \n"
                     "vfmacc.vf %[c3_psum], fa3, v0 \n"
                     :
                     : [pb] "r"(pb), [k] "r"(k), [pa0] "r"(pa0), [pa1] "r"(pa1),
                     [pa2] "r"(pa2), [pa3] "r"(pa3), [c0_psum] "vr"(c0_psum),
                     [c1_psum] "vr"(c1_psum), [c2_psum] "vr"(c2_psum),
                     [c3_psum] "vr"(c3_psum)
                     : "memory", "fa0", "fa1", "fa2", "fa3", "v0");

        pb += ldb;
    }

    float *pc0 = C;
    float *pc1 = C + ldc;
    float *pc2 = C + ldc * 2;
    float *pc3 = C + ldc * 3;

    auto c0_act = vle32_v_f32m4(pc0, vl);
    auto c1_act = vle32_v_f32m4(pc1, vl);
    auto c2_act = vle32_v_f32m4(pc2, vl);
    auto c3_act = vle32_v_f32m4(pc3, vl);

    c0_act = vfadd_vv_f32m4(c0_act, c0_psum, vl);
    c1_act = vfadd_vv_f32m4(c1_act, c1_psum, vl);
    c2_act = vfadd_vv_f32m4(c2_act, c2_psum, vl);
    c3_act = vfadd_vv_f32m4(c3_act, c3_psum, vl);

    // do activation
    if (act)
    {
        c0_act = vfmax_vf_f32m4(vfmin_vf_f32m4(c0_act, max, vl), min, vl);
        c1_act = vfmax_vf_f32m4(vfmin_vf_f32m4(c1_act, max, vl), min, vl);
        c2_act = vfmax_vf_f32m4(vfmin_vf_f32m4(c2_act, max, vl), min, vl);
        c3_act = vfmax_vf_f32m4(vfmin_vf_f32m4(c3_act, max, vl), min, vl);
    }

    vse32_v_f32m4(pc0, c0_act, vl);
    vse32_v_f32m4(pc1, c1_act, vl);
    vse32_v_f32m4(pc2, c2_act, vl);
    vse32_v_f32m4(pc3, c3_act, vl);
    return vl;
}

size_t matmul_kernel_1xN(size_t maxk, size_t N, const float *A, size_t lda,
    const float *B, size_t ldb, const float *Bias,
    float *C, size_t ldc, bool act, float min, float max)
{
    auto vl = vsetvl_e32m8(N);
    const float *pa0 = A;
    const float *pb = B;

    vfloat32m8_t c0_psum = act ? vle32_v_f32m8(Bias, vl) : vfmv_v_f_f32m8(0.f, vl);

    for (size_t k = 0; k < maxk; k++)
    {
        auto vb = vle32_v_f32m8(pb, vl);
        c0_psum = vfmacc_vf_f32m8(c0_psum, *pa0++, vb, vl);
        pb += ldb;
    }

    float *pc0 = C;

    auto c0_act = vle32_v_f32m8(pc0, vl);
    c0_act = vfadd_vv_f32m8(c0_act, c0_psum, vl);

    // do activation
    if (act)
    {
        c0_act = vfmax_vf_f32m8(vfmin_vf_f32m8(c0_act, max, vl), min, vl);
    }

    vse32_v_f32m8(pc0, c0_act, vl);
    return vl;
}

void matmul_block_4x16(size_t M, size_t maxk, size_t N, const float *A,
    size_t lda, const float *B, size_t ldb,
    const float *Bias, float *C, size_t ldc, bool act,
    float min, float max)
{
    const float *pa_m = A;
    float *pc_m = C;
    size_t m = M;
    size_t k = k;

    while (m >= 4)
    {
        size_t n = N;
        float *pc_n = pc_m;
        const float *pb_n = B;
        const float *pbias_n = Bias;
        while (n)
        {
            auto vl = matmul_kernel_4x16(maxk, n, pa_m, lda, pb_n, ldb, pbias_n, pc_n,
                ldc, act, min, max);
            n -= vl;
            pb_n += vl;
            pc_n += vl;
            pbias_n += vl;
        }
        pa_m += lda * 4;
        pc_m += ldc * 4;
        m -= 4;
    }

    // m < 4

    while (m)
    {
        size_t n = N;
        float *pc_n = pc_m;
        const float *pb_n = B;
        const float *pbias_n = Bias;
        while (n)
        {
            auto vl = matmul_kernel_1xN(maxk, n, pa_m, lda, pb_n, ldb, pbias_n, pc_n,
                ldc, act, min, max);
            n -= vl;
            pb_n += vl;
            pc_n += vl;
            pbias_n += vl;
        }
        pa_m += lda;
        pc_m += ldc;
        m--;
    }
}

const size_t mc = 32;
const size_t kc = 32;

void matmul_rvv(size_t M, size_t K, size_t N, const float *A, size_t lda, const float *B, size_t ldb,
    const float *Bias, float *C, size_t ldc, float min, float max)
{
    for (size_t p = 0; p < K; p += kc)
    {
        auto pb = std::min(K - p, kc);
        for (size_t i = 0; i < M; i += mc)
        {
            auto ib = std::min(M - i, mc);
            bool act = M - i == ib;
            matmul_block_4x16(ib, pb, N, &A[i * lda + p], lda, &B[p * ldb], ldb, Bias,
                &C[i * ldc], ldc, act, min, max);
        }
    }
}

// float
result<void> optimized_matmul_impl(const float *input_a, const float *input_b, const float *bias, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t out_strides,
    value_range<float> fused_activation) noexcept
{
    size_t M = in_a_shape[in_a_shape.size() - 2];
    size_t K = in_a_shape.back();
    size_t N = in_b_shape.back();
    size_t lda = in_a_strides[in_a_shape.size() - 2];
    size_t ldb = in_b_strides[in_b_shape.size() - 2];
    size_t ldc = out_strides[out_shape.size() - 2];

    // batch
    size_t batch_a = 1;
    for (size_t i = 0; i < in_a_shape.size() - 2; i++)
        batch_a *= in_a_shape[i];
    size_t step_a = batch_a == 1 ? 0 : in_a_strides[0];

    size_t batch_b = 1;
    for (size_t i = 0; i < in_b_shape.size() - 2; i++)
        batch_b *= in_b_shape[i];
    size_t step_b = batch_b == 1 ? 0 : in_b_strides[0];

    size_t batch_out = 1;
    for (size_t i = 0; i < out_shape.size() - 2; i++)
        batch_out *= out_shape[i];
    size_t step_out = batch_out == 1 ? 0 : out_strides[0];

    size_t batch_max = std::max(batch_a, batch_b);

    const float *p_batch_a = input_a;
    const float *p_batch_b = input_b;
    float *p_batch_out = output;

    size_t vl = 0;
    for (size_t i = 0; i < batch_max; i++)
    {
        matmul_rvv(M, K, N, p_batch_a, lda, p_batch_b, ldb, bias, p_batch_out, ldc,
            fused_activation.min, fused_activation.max);
        p_batch_a += step_a;
        p_batch_b += step_b;
        p_batch_out += step_out;
    }

    return ok();
}
#endif
}

template result<void> optimized::matmul<float>(const float *input_a, const float *input_b, const float *bias, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept;

template <typename T>
result<void> optimized::matmul(const T *input_a, const T *input_b, const T *bias, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept
{
#if __riscv_vector
    return optimized_matmul_impl(input_a, input_b, bias, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides, fused_activation);
#endif

    return cpu::reference::matmul(input_a, input_b, bias, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides,
        out_shape, out_strides, fused_activation);
}