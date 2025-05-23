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
#include "../shape_infer/matmul.h"
#include "../tensor.h"
#include "../ukernels.h"
#include "matmul.h"
#ifdef NNCASE_XPU_MODULE
#include "nncase/ntt/arch/xpu/topology.h"
#else
#include "nncase/ntt/arch/cpu/topology.h"
#endif
#include <numeric>
#include <type_traits>

using namespace nncase;
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;

namespace nncase::ntt {
template <bool AccumulateC = false, bool TransposedA = false,
          bool TransposedB = false, class TLhs, class TRhs, class TOut,
          FixedDimensions LhsPackedAxes, FixedDimensions LhsPadedNums,
          FixedDimensions RhsPackedAxes, FixedDimensions RhsPadedNums>
void summa(const TLhs &lhs, const TRhs &rhs, TOut &&output,
           [[maybe_unused]] LhsPackedAxes lhsPackedAxes = fixed_shape_v<>,
           [[maybe_unused]] LhsPadedNums lhsPadedNums = fixed_shape_v<>,
           [[maybe_unused]] RhsPackedAxes rhsPackedAxes = fixed_shape_v<>,
           [[maybe_unused]] RhsPadedNums rhsPadedNums = fixed_shape_v<>) {
    static_assert(TransposedA == false && TransposedB == false,
                  "not supported for now");
    using TLhsElem = typename TLhs::local_tensor_type::element_type;
    using TRhsElem = typename TRhs::local_tensor_type::element_type;
    using TOutElem =
        typename std::decay_t<TOut>::local_tensor_type::element_type;

    using lhs_mesh_type = typename TLhs::mesh_type;
    using rhs_mesh_type = typename TRhs::mesh_type;

    using lhs_global_shape = typename TLhs::shape_type;
    using rhs_global_shape = typename TRhs::shape_type;

    auto lhs_local_tensor = lhs.local();
    auto rhs_local_tensor = rhs.local();
    auto out_local_tensor = output.local();
    using out_local_strides =
        typename std::decay_t<TOut>::local_tensor_type::strides_type;

    auto lhs_local_index = lhs_mesh_type::local_index();
    auto rhs_local_index = rhs_mesh_type::local_index();

    constexpr size_t lhs_rank = lhs_global_shape::rank();
    constexpr size_t rhs_rank = rhs_global_shape::rank();

    constexpr size_t lhs_k_dim = lhs_rank - 1;
    constexpr size_t rhs_k_dim = rhs_rank - 2;

    constexpr size_t K_lhs = lhs_global_shape::at(lhs_k_dim);
    constexpr size_t K_rhs = rhs_global_shape::at(rhs_k_dim);
    static_assert(K_lhs == K_rhs,
                  "K dimensions must match for matrix multiplication");
    constexpr size_t K = K_lhs;

    // using last two meshes(b&t) for cpu, x&y for xpu
    constexpr size_t mesh_rank = lhs_mesh_type::rank();
    auto local_K_lhs = lhs_local_tensor.shape().at(lhs_k_dim);
    auto local_K_rhs = rhs_local_tensor.shape().at(rhs_k_dim);

    auto lcm_K = std::lcm(local_K_lhs, local_K_rhs);
    constexpr auto LhsRank = TLhs::local_tensor_type::rank();
    constexpr auto RhsRank = TRhs::local_tensor_type::rank();
    constexpr auto OutRank = std::decay_t<TOut>::local_tensor_type::rank();
    const auto CShape = out_local_tensor.shape().template slice<0, OutRank>();
    const auto CStrides =
        out_local_tensor.strides().template slice<0, OutRank>();

    auto C = ntt::make_tensor_view<TOutElem>(output.local().buffer().data(),
                                             CShape, CStrides);

    // TODO: start from different k_offset can reduce confliction
    for (size_t k_offset = 0; k_offset < K; k_offset += lcm_K) {
        if (local_K_lhs >= local_K_rhs) {
            size_t lhs_k_mesh_idx = k_offset / local_K_lhs;
            auto lhs_mesh_index = lhs_local_index;
            lhs_mesh_index.at(mesh_rank - 1) = lhs_k_mesh_idx;
            auto A_remote =
                lhs.template remote<lhs_mesh_type::scope>(lhs_mesh_index);

            for (size_t rhs_k_offset = 0; rhs_k_offset < lcm_K;
                 rhs_k_offset += local_K_rhs) {
                size_t global_rhs_k = k_offset + rhs_k_offset;
                size_t rhs_k_mesh_idx = global_rhs_k / local_K_rhs;
                auto rhs_mesh_index = rhs_local_index;
                rhs_mesh_index.at(mesh_rank - 2) = rhs_k_mesh_idx;
                auto B_remote =
                    rhs.template remote<rhs_mesh_type::scope>(rhs_mesh_index);

                const auto AShape = lhs_local_tensor.shape()
                                        .template slice<0, LhsRank - 1>()
                                        .append(local_K_rhs);
                const auto AStrides =
                    A_remote.strides().template slice<0, LhsRank>();
                const auto BShape =
                    rhs_local_tensor.shape().template slice<0, RhsRank>();
                const auto BStrides =
                    B_remote.strides().template slice<0, RhsRank>();

                auto A = ntt::make_tensor_view<TLhsElem>(
                    A_remote.buffer().data() + rhs_k_offset, AShape, AStrides);
                auto B = ntt::make_tensor_view<TRhsElem>(
                    B_remote.buffer().data(), BShape, BStrides);

                if (global_rhs_k == 0) {
                    ntt::matmul<AccumulateC, TransposedA, TransposedB>(
                        A, B, C, lhsPackedAxes, lhsPadedNums, rhsPackedAxes,
                        rhsPadedNums);
                } else {
                    ntt::matmul<true, TransposedA, TransposedB>(
                        A, B, C, lhsPackedAxes, lhsPadedNums, rhsPackedAxes,
                        rhsPadedNums);
                }
            }
        } else {
            size_t rhs_k_mesh_idx = k_offset / local_K_rhs;
            auto rhs_mesh_index = rhs_local_index;
            rhs_mesh_index.at(mesh_rank - 2) = rhs_k_mesh_idx;
            auto B_remote =
                rhs.template remote<rhs_mesh_type::scope>(rhs_mesh_index);

            for (size_t lhs_k_offset = 0; lhs_k_offset < lcm_K;
                 lhs_k_offset += local_K_lhs) {
                size_t global_lhs_k = k_offset + lhs_k_offset;
                size_t lhs_k_mesh_idx = global_lhs_k / local_K_lhs;
                auto lhs_mesh_index = lhs_local_index;
                lhs_mesh_index.at(mesh_rank - 1) = lhs_k_mesh_idx;
                auto A_remote =
                    lhs.template remote<rhs_mesh_type::scope>(lhs_mesh_index);

                const auto BShape = rhs_local_tensor.shape()
                                        .template slice<0, RhsRank - 1>()
                                        .append(local_K_lhs);
                const auto BStrides =
                    B_remote.strides().template slice<0, RhsRank>();

                const auto AShape =
                    lhs_local_tensor.shape().template slice<0, LhsRank>();
                const auto AStrides =
                    A_remote.strides().template slice<0, LhsRank>();
                auto B = ntt::make_tensor_view<TRhsElem>(
                    B_remote.buffer().data() +
                        lhs_k_offset * BStrides.template at<rhs_rank - 2>(),
                    BShape, BStrides);
                auto A = ntt::make_tensor_view<TLhsElem>(
                    A_remote.buffer().data(), AShape, AStrides);

                if (global_lhs_k == 0) {
                    ntt::matmul<AccumulateC, TransposedA, TransposedB>(
                        A, B, C, lhsPackedAxes, lhsPadedNums, rhsPackedAxes,
                        rhsPadedNums);
                } else {
                    ntt::matmul<true, TransposedA, TransposedB>(
                        A, B, C, lhsPackedAxes, lhsPadedNums, rhsPackedAxes,
                        rhsPadedNums);
                }
            }
        }
    }

    // TODO: remove this when summa tiling is ready
    if constexpr (Vector<TLhsElem> && Vector<TRhsElem>) {
        if constexpr (TLhsElem::shape_type::rank() == 2 &&
                      TRhsElem::shape_type::rank() == 2 &&
                      TOutElem::shape_type::at(0) == 64 &&
                      TOutElem::shape_type::at(1) == 64 &&
                      (std::is_same_v<float, typename TLhsElem::element_type> ||
                       std::is_same_v<half, typename TLhsElem::element_type>)) {

            ntt::apply(CShape, [&](auto index) {
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
}
} // namespace nncase::ntt
