#pragma once
#include "nncase/ntt/apply.h"
#include "nncase/ntt/shape.h"
#include <cstring>
#include <iostream>
#include <riscv_vector.h>
// namespace nncase::ntt {
namespace nncase::ntt::ukernels {

// 2D 恒等 (0,1) ：会被compress, 不需要实现了
// template <Tensor TIn, class TOut>
// class u_transpose_impl<TIn, TOut, fixed_shape_t<0, 1>, true>

// 2D 转置 (1,0) ：使用向量加载 + 具间隔(strided)存储
// template <Tensor TIn, class TOut>
// class u_transpose_impl<TIn, TOut, fixed_shape_t<1, 0>, true> {
//   public:
//     constexpr void
//     operator()([[maybe_unused]] const TIn &input, [[maybe_unused]] TOut &output,
//                [[maybe_unused]] const fixed_shape_t<1, 0> &) const {
//         static_assert(TIn::rank() == 2);

//         using InElem = element_or_scalar_t<TIn>;
//         using OutElem = typename std::decay_t<TOut>::element_type;

//         const size_t M = input.shape()[0];
//         const size_t N = input.shape()[1];

//         if constexpr (Vector<InElem>) {
//             const InElem *vec_input_ptr = input.elements().data();
//             InElem *vec_output_ptr = output.elements().data();
//             if (input.strides()[1] == 1 && output.strides()[1] == 1 &&
//                 input.strides()[0] == N && output.strides()[0] == M) {
//                 const intptr_t out_stride_bytes = sizeof(float);

//                 for (size_t i = 0; i < M; ++i) {
//                     const auto row = vec_input_ptr + i * N;
//                     size_t remain = N * (NTT_VLEN / 32);
//                     size_t j = 0;
//                     while (remain) {
//                         size_t vl = __riscv_vsetvl_e32m1(remain);
//                         vfloat32m1_t v =
//                             __riscv_vle32_v_f32m1((float *)(row + j), vl);
//                         auto col_dst = vec_output_ptr + (j++) * M + i;
//                         __riscv_vsse32_v_f32m1((float *)col_dst,
//                                                out_stride_bytes, v, vl);
//                         remain -= vl;
//                     }
//                 }
//                 return;
//             } else {
//             }
//         } else {
//             const float *in_base = input.elements().data();
//             float *out_base = output.elements().data();
//             if (input.strides()[1] == 1 && output.strides()[1] == 1 &&
//                 input.strides()[0] == N && output.strides()[0] == M) {
//                 printf("%s, %d: \n", __FILE__, __LINE__);
//                 printf("strides: in(%zu, %zu), out(%zu, %zu)\n",
//                        input.strides()[0], input.strides()[1],
//                        output.strides()[0], output.strides()[1]);
//                 const intptr_t out_stride_bytes =
//                     static_cast<intptr_t>(M * sizeof(float));

//                 for (size_t i = 0; i < M; ++i) {
//                     const float *row = in_base + i * N;
//                     size_t remain = N;
//                     size_t j = 0;
//                     while (remain) {
//                         size_t vl = __riscv_vsetvl_e32m4(remain);
//                         vfloat32m4_t v = __riscv_vle32_v_f32m4(row + j, vl);
//                         auto col_dst = out_base + j * M + i;
//                         __riscv_vsse32_v_f32m4(col_dst, out_stride_bytes, v,
//                                                vl);
//                         j += vl;
//                         remain -= vl;
//                     }
//                 }
//                 return;
//             } else {
//                 printf("%s, %d: \n", __FILE__, __LINE__);
//                 const intptr_t out_stride_bytes =
//                     static_cast<intptr_t>(output.strides()[1] * sizeof(float));
//                 const intptr_t in_stride_bytes =
//                     static_cast<intptr_t>(input.strides()[1] * sizeof(float));
//                 for (size_t i = 0; i < M; ++i) {
//                     const float *row = in_base + i * N;
//                     float *col_dst = out_base + i * output.strides()[0];
//                     size_t remain = N;
//                     size_t j = 0;
//                     while (remain) {
//                         size_t vl = __riscv_vsetvl_e32m4(remain);
//                         vfloat32m4_t v = __riscv_vlse32_v_f32m4(row + j, in_stride_bytes, vl);
//                         col_dst = col_dst + j * output.strides()[1];
//                         __riscv_vsse32_v_f32m4(col_dst, out_stride_bytes, v,
//                                                vl);
//                         j += vl;
//                         remain -= vl;
//                     }
//                 }
//                 return;
//             }
//         }

//         printf("%s, %d: \n", __FILE__, __LINE__);
//         for (size_t i = 0; i < input.shape()[0]; ++i)
//             for (size_t j = 0; j < input.shape()[1]; ++j)
//                 output(j, i) = input(i, j);
//     }
// };
// } // namespace nncase::ntt::ukernels

// 通用接口，用于处理不确定维度的transpose，中间保留部分函数调用，来实现rvv优化
/*
template <int NUM_AXES, int UNROLL, typename T>
void transpose_v3(const T* data_in, T* data_out, 
                 const Array<uint32_t, NUM_AXES> perm_strides, 
                 const Array<FastIntDivider<uint32_t>, NUM_AXES>& out_strides, 
                 const size_t all_cnt) {
    // 这个版本的优化主要是针对CUDA的内存访问模式
    // 在C++中，我们保持算法逻辑但使用更适合CPU的实现方式
    for (size_t offset = 0; offset < all_cnt; offset += UNROLL) {
        uint32_t out_offset_reg[UNROLL];
        uint32_t in_offset_reg[UNROLL];
        T ld_reg[UNROLL];
        
        for (int i = 0; i < UNROLL; ++i) {
            out_offset_reg[i] = offset + i;
            in_offset_reg[i] = 0;
        }

        for (int j = 0; j < NUM_AXES; ++j) {
            for (int i = 0; i < UNROLL; ++i) {
                if (out_offset_reg[i] >= all_cnt) continue;
                QuotientMod<uint32_t> d = out_strides[j].divmod(out_offset_reg[i]);
                in_offset_reg[i] += d.quotient * perm_strides[j];
                out_offset_reg[i] = d.mod;
            }
        }

        // 读取数据
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i < all_cnt) {
                ld_reg[i] = data_in[in_offset_reg[i]];
            }
        }
        
        // 写入数据
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i < all_cnt) {
                data_out[offset + i] = ld_reg[i];
            }
        }
    }
}
*/
template <Tensor TIn, class TOut, FixedDimensions TPerms>
void u_transpose_rvv_impl(const TIn &input, TOut &output, const TPerms &perms) {
    constexpr size_t R = TIn::rank();
    using perms_t = std::decay_t<TPerms>;
    using OutRef = std::decay_t<TOut>;

    // 2D 特化调度：perms == (1,0)
    // if constexpr (R == 2) {
    //     if constexpr (std::is_same_v<perms_t, fixed_shape_t<1, 0>>) {
    //         ukernels::u_transpose_impl<TIn, OutRef, fixed_shape_t<1, 0>, true>{}(input, output, perms);
    //         return;
    //     }
    // }

    // 判断是否恒等 perm
    bool is_identity = true;
    for (size_t i = 0; i < R; ++i)
        if (perms[i] != i) { is_identity = false; break; }
    if (is_identity) {
        // 直接拷贝（保持简单，利用逐元素）
        // 可根据需要再加 contiguous 判断后用 memcpy
        std::array<size_t, R> idx{};
        // 递归生成
        // 通用递归枚举
        auto rec_id = [&](auto &&self, size_t axis) -> void {
            if (axis == R) {
                // 展开调用
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    output(idx[Is]...) = input(idx[Is]...);
                }(std::make_index_sequence<R>{});
                return;
            }
            const auto dim = input.shape()[axis];
            for (size_t v = 0; v < dim; ++v) {
                idx[axis] = v;
                self(self, axis + 1);
            }
        };
        rec_id(rec_id, 0);
        return;
    }

    // 通用 N-D 转置
    std::array<size_t, R> out_idx{};
    std::array<size_t, R> in_idx{};

    // 递归枚举输出坐标，映射输入
    auto assign_leaf = [&](const std::array<size_t, R> &out_idx_local) {
        // 构造 in_idx
        for (size_t ax = 0; ax < R; ++ax)
            in_idx[perms[ax]] = out_idx_local[ax];
        // 展开调用
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            output(out_idx_local[Is]...) = input(in_idx[Is]...);
        }(std::make_index_sequence<R>{});
    };

    auto rec = [&](auto &&self, size_t axis) -> void {
        if (axis == R) {
            assign_leaf(out_idx);
            return;
        }
        // 输出第 axis 维长度 = 输入 perms[axis] 对应原维长度
        const size_t dim = input.shape()[perms[axis]];
        for (size_t v = 0; v < dim; ++v) {
            out_idx[axis] = v;
            self(self, axis + 1);
        }
    };
    rec(rec, 0);
}


} // namespace nncase::ntt