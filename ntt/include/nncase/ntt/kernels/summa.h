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
#pragma once
#include "../apply.h"
#include "matmul.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor_traits.h"
#include <numeric>
#include <type_traits>

using namespace nncase;
using namespace nncase::ntt;

namespace nncase::ntt {
template <bool AccumulateC = false, bool TransposedA = false,
          bool TransposedB = false, ShardedTensor TLhs, ShardedTensor TRhs,
          class TOut, FixedDimensions LhsVectorizedAxes,
          FixedDimensions LhsPadedNums, FixedDimensions RhsVectorizedAxes,
          FixedDimensions RhsPadedNums>
void summa(const TLhs &lhs, const TRhs &rhs, TOut &&output,
           [[maybe_unused]] LhsVectorizedAxes lhsVectorizedAxes = fixed_shape_v<>,
           [[maybe_unused]] LhsPadedNums lhsPadedNums = fixed_shape_v<>,
           [[maybe_unused]] RhsVectorizedAxes rhsVectorizedAxes = fixed_shape_v<>,
           [[maybe_unused]] RhsPadedNums rhsPadedNums = fixed_shape_v<>) {
    static_assert(TransposedA == false && TransposedB == false,
                  "not supported for now");
    using TLhsElem = typename TLhs::element_type;
    using TRhsElem = typename TRhs::element_type;
    using TOutElem = typename std::decay_t<TOut>::element_type;

    using lhs_mesh_type = typename TLhs::mesh_type;
    using rhs_mesh_type = typename TRhs::mesh_type;

    auto lhs_local_tensor = lhs.local();
    auto rhs_local_tensor = rhs.local();

    auto lhs_local_index = lhs_mesh_type::local_index();
    auto rhs_local_index = rhs_mesh_type::local_index();

    constexpr auto lhs_rank = TLhs::rank();
    constexpr auto rhs_rank = TRhs::rank();
    constexpr auto out_rank = std::decay_t<TOut>::rank();

    constexpr auto lhs_k_dim = lhs_rank - 1_dim;
    constexpr auto rhs_k_dim = rhs_rank - 2_dim;

    const auto K_lhs = lhs.shape()[lhs_k_dim];
    const auto K_rhs = rhs.shape()[rhs_k_dim];
    static_assert(K_lhs == K_rhs,
                  "K dimensions must match for matrix multiplication");
    const auto K = K_lhs;

    // using last two meshes(b&t) for cpu, x&y for xpu
    auto local_K_lhs = lhs_local_tensor.shape()[lhs_k_dim];
    auto local_K_rhs = rhs_local_tensor.shape()[rhs_k_dim];

    const auto lcm_K = ntt::lcm(local_K_lhs, local_K_rhs);
    auto C = output.local();

    // TODO: start from different k_offset can reduce confliction
    for (dim_t k_offset = 0; k_offset < K; k_offset += lcm_K) {
        if (local_K_lhs >= local_K_rhs) {
            const dim_t lhs_k_mesh_idx = k_offset / local_K_lhs;
            const auto lhs_shard_index =
                lhs_local_index.template replace_at<-1>(lhs_k_mesh_idx);
            const auto A_remote = lhs.remote(lhs_shard_index);

            for (dim_t rhs_k_offset = 0; rhs_k_offset < lcm_K;
                 rhs_k_offset += local_K_rhs) {
                const dim_t global_rhs_k = k_offset + rhs_k_offset;
                const dim_t rhs_k_mesh_idx = global_rhs_k / local_K_rhs;
                const auto rhs_shard_index =
                    rhs_local_index.template replace_at<-2>(rhs_k_mesh_idx);
                const auto B = rhs.remote(rhs_shard_index);
                const auto A_offset =
                    make_zeros_shape<lhs_rank>().template replace_at<-1>(
                        rhs_k_offset);
                const auto A_shape =
                    A_remote.shape().template replace_at<-1>(local_K_rhs);
                const auto A = A_remote.view(A_offset, A_shape);

                if (global_rhs_k == 0) {
                    ntt::matmul<AccumulateC, TransposedA, TransposedB>(
                        A, B, C, lhsVectorizedAxes, lhsPadedNums, rhsVectorizedAxes,
                        rhsPadedNums);
                } else {
                    ntt::matmul<true, TransposedA, TransposedB>(
                        A, B, C, lhsVectorizedAxes, lhsPadedNums, rhsVectorizedAxes,
                        rhsPadedNums);
                }
            }
        } else {
            const auto rhs_k_mesh_idx = k_offset / local_K_rhs;
            const auto rhs_shard_index =
                rhs_local_index.template replace_at<-2>(rhs_k_mesh_idx);
            auto B_remote = rhs.remote(rhs_shard_index);

            for (dim_t lhs_k_offset = 0; lhs_k_offset < lcm_K;
                 lhs_k_offset += local_K_lhs) {
                const dim_t global_lhs_k = k_offset + lhs_k_offset;
                const dim_t lhs_k_mesh_idx = global_lhs_k / local_K_lhs;
                auto lhs_shard_index =
                    lhs_local_index.template replace_at<-1>(lhs_k_mesh_idx);
                const auto A = lhs.remote(lhs_shard_index);
                const auto B_offset =
                    make_zeros_shape<rhs_rank>().template replace_at<-2>(
                        lhs_k_offset);
                const auto B_shape =
                    B_remote.shape().template replace_at<-2>(local_K_lhs);
                const auto B = B_remote.view(B_offset, B_shape);
                if (global_lhs_k == 0) {
                    ntt::matmul<AccumulateC, TransposedA, TransposedB>(
                        A, B, C, lhsVectorizedAxes, lhsPadedNums, rhsVectorizedAxes,
                        rhsPadedNums);
                } else {
                    ntt::matmul<true, TransposedA, TransposedB>(
                        A, B, C, lhsVectorizedAxes, lhsPadedNums, rhsVectorizedAxes,
                        rhsPadedNums);
                }
            }
        }
    }

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
    // TODO: remove this when summa tiling is ready
    if constexpr (Vector<TLhsElem> && Vector<TRhsElem>) {
        if constexpr (TLhsElem::shape_type::rank() == 2 &&
                      TRhsElem::shape_type::rank() == 2 &&
                      std::is_same_v<typename TOutElem::shape_type,
                                     fixed_shape_t<64, 64>> &&
                      (std::is_same_v<float, typename TLhsElem::element_type> ||
                       std::is_same_v<half, typename TLhsElem::element_type>)) {

            ntt::apply(C.shape(), [&](auto index) {
                auto data = (float *)C(index).buffer().data();
                float tmp[64 * 64];
                for (int i = 0; i < 64; i++) {
                    std::memcpy(tmp + i * 2 * 32, data + i * 32,
                                32 * sizeof(float));
                    std::memcpy(tmp + (i * 2 + 1) * 32, data + (i + 64) * 32,
                                32 * sizeof(float));
                }

                std::memcpy(data, tmp, 64 * 64 * sizeof(float));
            });
        }
    }
#endif
}
} // namespace nncase::ntt
