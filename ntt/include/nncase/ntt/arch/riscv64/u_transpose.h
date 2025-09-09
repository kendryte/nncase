#pragma once
#include <riscv_vector.h>
#include <cstring>
#include "nncase/ntt/shape.h"
#include "nncase/ntt/apply.h"

namespace nncase::ntt::ukernels {

// // 通用回退版本（任意维度、任意排列）
// template <Tensor TIn, class TOut, FixedDimensions TPerms, bool Arch>
// class u_transpose_impl {
// public:
//     constexpr void operator()(const TIn &input, TOut &output, const TPerms &) const {
//         constexpr auto rank = TIn::rank();
//         constexpr TPerms perm_const;
//         constexpr auto pos_perms = positive_axes(perm_const, rank);
//         ntt::apply(input.shape(), [&](auto index) {
//             auto out_index = generate_shape<rank>(
//                 [&](auto i) { return index[pos_perms[i]]; });
//             output(out_index) = input(index);
//         });
//     }
// };

// 2D 恒等 (0,1) ：直接行向量复制（如需可进一步判断完全连续后整体 memcpy）
// template <Tensor TIn, class TOut>
// class u_transpose_impl<TIn, TOut, fixed_shape_t<0, 1>, true> {
// public:
//     constexpr void operator()([[maybe_unused]]const TIn &input, [[maybe_unused]]TOut &output,
//                               [[maybe_unused]]const fixed_shape_t<0,1> &) const {
//                                 printf("riscv64 u_transpose_impl<0,1>\n");

//         // static_assert(TIn::rank() == 2);
//         // using TinElem = typename TIn::element_type;
//         // using ToutElem = typename std::decay_t<TOut>::element_type;

//         // auto M = input.shape()[0];
//         // auto N = input.shape()[1];

//         // // 若内存布局一致且元素类型相同，可一次性 memcpy
//         // if constexpr (std::is_same_v<TinElem, ToutElem>) {
//         //     bool contiguous =
//         //         input.strides()[1] == 1 &&
//         //         input.strides()[0] == N &&
//         //         output.strides()[1] == 1 &&
//         //         output.strides()[0] == N;
//         //     if (contiguous) {
//         //         std::memcpy(output.elements().data(),
//         //                     input.elements().data(),
//         //                     sizeof(TinElem) * M * N);
//         //         return;
//         //     }
//         // }

//         // // 行复制（当元素为 float 且行内连续时使用向量指令）
//         // if constexpr (std::is_same_v<TinElem, float> &&
//         //               std::is_same_v<ToutElem, float>) {
//         //     bool row_contig_in  = (input.strides()[1] == 1);
//         //     bool row_contig_out = (output.strides()[1] == 1);
//         //     if (row_contig_in && row_contig_out) {
//         //         const float *in_ptr  = input.elements().data();
//         //         float *out_ptr       = output.elements().data();
//         //         size_t in_row_stride  = input.strides()[0];
//         //         size_t out_row_stride = output.strides()[0];
//         //         for (size_t i = 0; i < M; ++i) {
//         //             size_t remain = N;
//         //             const float *src = in_ptr + i * in_row_stride;
//         //             float *dst = out_ptr + i * out_row_stride;
//         //             while (remain) {
//         //                 size_t vl = vsetvl_e32m1(remain);
//         //                 vfloat32m1_t v = vle32_v_f32m1(src, vl);
//         //                 vse32_v_f32m1(dst, v, vl);
//         //                 src += vl;
//         //                 dst += vl;
//         //                 remain -= vl;
//         //             }
//         //         }
//         //         return;
//         //     }
//         // }

//         // // 回退
//         // for (size_t i = 0; i < M; ++i)
//         //     for (size_t j = 0; j < N; ++j)
//         //         output(i, j) = input(i, j);
//     }
// };

// 2D 转置 (1,0) ：使用向量加载 + 具间隔(strided)存储
template <Tensor TIn, class TOut>
class u_transpose_impl<TIn, TOut, fixed_shape_t<1, 0>, true> {
public:
    constexpr void operator()([[maybe_unused]]const TIn &input, [[maybe_unused]]TOut &output,
                              [[maybe_unused]]const fixed_shape_t<1,0> &) const {
                                printf("riscv64 u_transpose_impl<0,1>\n");
                static_assert(TIn::rank() == 2);
        using InElem  = typename TIn::element_type;
        using OutElem = typename std::decay_t<TOut>::element_type;
printf("%s, %d: \n", __FILE__, __LINE__);
#ifdef __riscv_vector
printf("%s, %d: \n", __FILE__, __LINE__);
        {  
            if constexpr (std::is_same_v<InElem, vector<float>> && std::is_same_v<OutElem, vector<float>>) 
                {
                    printf("%s, %d: \n", __FILE__, __LINE__);
                }
        }
        // if constexpr (std::is_same_v<InElem, float> && std::is_same_v<OutElem, float>) {
            const size_t M = input.shape()[0];
            const size_t N = input.shape()[1];
            // 要求输入/输出都是行主连续
            if (input.strides()[1] == 1 && output.strides()[1] == 1 &&
                input.strides()[0] == N && output.strides()[0] == M) {

                const float *in_base = input.elements().data();
                // float *out_base = output.elements().data();
                const intptr_t out_stride_bytes = static_cast<intptr_t>(M * sizeof(float));

                for (size_t i = 0; i < M; ++i) {
                    auto row = &input(i,0);
                    size_t remain = N;
                    size_t j = 0;
                    while (remain) {
                        size_t vl = __riscv_vsetvl_e32m1(remain);
                        vfloat32m1_t v = __riscv_vle32_v_f32m1(row + j, vl);
                        auto col_dst = &output(j,i); // 输出矩阵列首地址
                        __riscv_vsse32_v_f32m1(col_dst, out_stride_bytes, v, vl);
                        j += vl;
                        remain -= vl;
                    }
                }
                return;
            }
        // }
#endif
printf("%s, %d: \n", __FILE__, __LINE__);
        // 标量回退
        for (size_t i = 0; i < input.shape()[0]; ++i)
            for (size_t j = 0; j < input.shape()[1]; ++j)
                output(j, i) = input(i, j);
    }

};

} // namespace nncase::ntt::ukernels