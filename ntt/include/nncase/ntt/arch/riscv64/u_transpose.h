#pragma once
#include "nncase/ntt/apply.h"
#include "nncase/ntt/shape.h"
#include <cstring>
#include <riscv_vector.h>
#include <iostream>

namespace nncase::ntt::ukernels {

// 2D 恒等 (0,1) ：会被compress, 不需要实现了
// template <Tensor TIn, class TOut>
// class u_transpose_impl<TIn, TOut, fixed_shape_t<0, 1>, true> 

// 2D 转置 (1,0) ：使用向量加载 + 具间隔(strided)存储
template <Tensor TIn, class TOut>
class u_transpose_impl<TIn, TOut, fixed_shape_t<1, 0>, true> {
  public:
    constexpr void
    operator()([[maybe_unused]] const TIn &input, [[maybe_unused]] TOut &output,
               [[maybe_unused]] const fixed_shape_t<1, 0> &) const {
        static_assert(TIn::rank() == 2);

        using InElem = element_or_scalar_t<TIn>;
        using OutElem = typename std::decay_t<TOut>::element_type;

        const size_t M = input.shape()[0];
        const size_t N = input.shape()[1];
        if (input.strides()[1] == 1 && output.strides()[1] == 1 &&
            input.strides()[0] == N && output.strides()[0] == M) 
        {
            if constexpr (Vector<InElem>) {
                const InElem *vec_input_ptr = input.elements().data();
                InElem *vec_output_ptr = output.elements().data();
                const intptr_t out_stride_bytes = sizeof(float);

                for (size_t i = 0; i < M; ++i) {
                    const auto row = vec_input_ptr + i*N;
                    size_t remain = N * (NTT_VLEN / 32);
                    size_t j = 0;
                    size_t k = 0;
                    while (remain) {
                        size_t vl = __riscv_vsetvl_e32m1(remain);
                        vfloat32m1_t v = __riscv_vle32_v_f32m1((float*)(row+k), vl);
                        auto col_dst = vec_output_ptr + (k++)*M + i;
                        __riscv_vsse32_v_f32m1((float*)col_dst,out_stride_bytes, v,
                                               vl);
                            remain -= vl;
                            j += vl;
                        }
                }
                return;
            } else {
                const float *in_base = input.elements().data();
                float *out_base = output.elements().data();
                const intptr_t out_stride_bytes =
                    static_cast<intptr_t>(M * sizeof(float));

                for (size_t i = 0; i < M; ++i) {
                    const float *row = in_base + i*N;
                    size_t remain = N;
                    size_t j = 0;
                    while (remain) {
                        size_t vl = __riscv_vsetvl_e32m4(remain);
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(row + j, vl);
                        auto col_dst = out_base + j * M + i;
                        __riscv_vsse32_v_f32m4(col_dst, out_stride_bytes, v,
                                               vl);
                        j += vl;
                        remain -= vl;
                    }
                }
                return;
            }
        }
       
        printf("%s, %d: \n", __FILE__, __LINE__);
        for (size_t i = 0; i < input.shape()[0]; ++i)
            for (size_t j = 0; j < input.shape()[1]; ++j)
                output(j, i) = input(i, j);
    }
};
} // namespace nncase::ntt::ukernels