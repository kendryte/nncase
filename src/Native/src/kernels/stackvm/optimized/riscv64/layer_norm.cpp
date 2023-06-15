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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#if __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

#if __riscv_vector
#define RVV_LMUL 8
#define _STR(x) #x
#define STR(x) _STR(x)
#define _CONNECT(a, b) a##b
#define CONNECT(a, b) _CONNECT(a, b)
#define RVVSETVLI2(evl, avl, elen)                                             \
    "vsetvli " STR(evl) "," STR(avl) "," STR(elen) "," STR(                    \
        CONNECT(m, RVV_LMUL)) ";"

static float get_mean(const float *data, int n) {
    float ret;
    __asm volatile("mv a0, %[avl];"
                   "mv a1, %[input_ptr1];" RVVSETVLI2(
                       t0, a0, e32) "vmv.s.x v0, x0;"
                                    "XXXXXX%=:;" RVVSETVLI2(
                                        t0, a0, e32) "vle32.v v8, (a1);"
                                                     "sub a0,a0, t0;"
                                                     "slli t1, t0, 2;"
                                                     "vfredsum.vs v0,v8,v0;"

                                                     "add a1, a1, t1;"
                                                     "bnez a0, XXXXXX%=;"
                                                     "vfmv.f.s f0, v0;"
                                                     "fcvt.s.w f1, %[avl];"
                                                     "fdiv.s %[ret], f0, f1;"

                   : [ret] "=f"(ret)
                   : [avl] "r"(n), [input_ptr1] "r"(data)
                   : "t0", "t1", "a0", "a1", "f0", "f1", "v0", "v8");
    return ret;
}

static float get_var(const float *data, int n, float mean) {
    float ret;
    __asm volatile(

        "mv a0, %[avl];"
        "mv a1, %[input_ptr1];" RVVSETVLI2(
            t0, a0, e32) "vmv.s.x v0, x0;"

                         "vle32.v v8, (a1);"
                         "sub a0,a0, t0;"
                         "slli t1, t0, 2;"
                         "vfsub.vf v8, v8, %[mean];"
                         "vfmul.vv v8, v8, v8;"
                         "add a1, a1, t1;"
                         "beqz a0, X1_END%=;"
                         "X1_STRAT%=:;" RVVSETVLI2(
                             t0, a0, e32) "vle32.v v16, (a1);"
                                          "sub a0,a0, t0;"
                                          "slli t1, t0, 2;"
                                          "vfsub.vf v16, v16, %[mean];"
                                          "vfmacc.vv v8, v16, v16;"

                                          "add a1, a1, t1;"
                                          "bnez a0, X1_STRAT%=;"

                                          "X1_END%=:"

                                          "vfredsum.vs v0,v8,v0;"

                                          "vfmv.f.s f0, v0;"
                                          "fcvt.s.w f1, %[avl];"
                                          "fdiv.s %[ret], f0, f1;"

        : [ret] "=f"(ret)
        : [avl] "r"(n), [input_ptr1] "r"(data), [mean] "f"(mean)
        : "t0", "t1", "a0", "a1", "v0", "v8", "v16", "f0", "f1");
    return ret;
}

static void layer_norm_update1(const float *data, float *out, int len,
                               float mean, float var, const float *r1, float e,
                               const float *b) {
    float r_sqrt = 1.0f / sqrtf(var + e);
    __asm volatile(
        "mv a0, %[avl];"
        "mv a1, %[input_ptr1];"
        "mv a2, %[out];"
        "mv a3, %[scale];"
        "mv a4, %[b];"
        "layer_norm_update1%=:;" RVVSETVLI2(
            t0, a0, e32) "vle32.v v16, (a1);"
                         "vle32.v v8, (a3);"
                         "sub a0,a0, t0;"
                         "slli t1, t0, 2;"
                         "vfsub.vf v16, v16, %[mean];"
                         "add a1, a1, t1;"
                         "vfmul.vf v16, v16, %[r_sqrt];"

                         "add a3, a3, t1;"
                         "vfmul.vv v16, v8, v16;"

                         "vle32.v v8, (a4);"
                         "vfadd.vv v16, v16, v8;"
                         "add a4, a4, t1;"

                         "vse32.v v16, (a2);"
                         "add a2, a2, t1;"
                         "bnez a0, layer_norm_update1%=;"

        :
        : [avl] "r"(len), [input_ptr1] "r"(data), [mean] "f"(mean),
          [r_sqrt] "f"(r_sqrt), [b] "r"(b), [out] "r"(out), [scale] "r"(r1)
        : "t0", "t1", "a0", "a1", "a2", "v0", "v16", "a3", "a4", "v8");
}

result<void> layernorm_impl(const float *input, float *output,
                            const float *scale, const float *bias,
                            gsl::span<const size_t> in_shape, int32_t axis,
                            float epsilon) {
    if (axis < 0) {
        axis = (int)in_shape.size() + axis;
    }
    auto outer_size = 1;
    auto inner_size = 1;
    for (auto i = 0; i < axis; i++)
        outer_size *= in_shape[i];
    for (auto i = axis; i < static_cast<int>(in_shape.size()); i++)
        inner_size *= in_shape[i];

    for (int32_t batch = 0; batch < outer_size; batch++) {
        const float *src = input + batch * inner_size;
        float *dest = output + batch * inner_size;

        float mean = get_mean(src, inner_size);

        float var_data = get_var(src, inner_size, mean);

        layer_norm_update1(src, dest, inner_size, mean, var_data, scale,
                           epsilon, bias);
    }
    return ok();
}
#endif

result<void> nncase::kernels::stackvm::optimized::layer_norm(
    const float *input, float *output, const float *scale, const float *bias,
    gsl::span<const size_t> in_shape, int32_t axis, float epsilon) {
#if __riscv_vector
    return layernorm_impl(input, output, scale, bias, in_shape, axis, epsilon);
#else
    return reference::layer_norm(input, output, scale, bias, in_shape, axis,
                                 epsilon);
#endif
}
