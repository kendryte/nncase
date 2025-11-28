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
#include "../shape.h"
#include "../tensor.h"
#include "../tensor_traits.h"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "../caching.h"
#include "binary.h"
#include "matmul.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "reduce.h"
#include "unary.h"
#include <type_traits>

// This version follows the Evaluator logic in src/Nncase.Evaluator/NN/SparseExperts.cs
//
// Shapes:
//  input q:                     [seq_len, hidden_size]
//  router_expert_ids:           [seq_len, num_top_k] - already selected topk expert indices
//  router_expert_weights:       [seq_len, num_top_k] - router weights for selected experts
//  moeExpertGateProjW:          [num_expert, moe_intermediate_size, hidden_size]
//  moeExpertGateProjScale:      [num_expert, moe_intermediate_size, 1] or [num_expert, 1]
//  moeExpertGateInputScale:     [num_expert, 1] (optional, can be empty/null)
//  moeExpertUpProjW:            [num_expert, moe_intermediate_size, hidden_size]
//  moeExpertUpProjScale:        [num_expert, moe_intermediate_size, 1] or [num_expert, 1]
//  moeExpertUpInputScale:       [num_expert, 1] (optional, can be empty/null)
//  moeExpertDownProjW:          [num_expert, hidden_size, moe_intermediate_size]
//  moeExpertDownProjScale:      [num_expert, hidden_size, 1] or [num_expert, 1]
//  moeExpertDownInputScale:     [num_expert, 1] (optional, can be empty/null)
//  output:                      [seq_len, hidden_size]
//
// Processing: Loop over experts (not tokens), gather tokens assigned to each expert.

namespace nncase::ntt {

namespace detail {

template <Tensor TQ, Tensor TRouterIds, Tensor TRouterWeights,
          Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TUpInputScale, Tensor TUpProjW, Tensor TUpProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale,
          class TOut>
void sparse_experts_impl(const TQ &q,
                         const TRouterIds &topk_indices,
                         const TRouterWeights &topk_probs,
                         const TGateInputScale &moeExpertGateInputScale,
                         const TGateProjW &moeExpertGateProjW,
                         const TGateProjScale &moeExpertGateProjScale,
                         const TDownInputScale &moeExpertDownInputScale,
                         const TDownProjW &moeExpertDownProjW,
                         const TDownProjScale &moeExpertDownProjScale,
                         const TUpInputScale &moeExpertUpInputScale,
                         const TUpProjW &moeExpertUpProjW,
                         const TUpProjScale &moeExpertUpProjScale,
                         size_t hidden_size,
                         size_t moe_intermediate_size,
                         size_t /* num_expert */,
                         size_t num_top_k,
                         size_t /* chunk_size */,
                         TOut &output) {
    using ElemType = typename TQ::element_type;
    const auto seq_len = q.shape()[0_dim];

    // Initialize output to zero
    for (size_t i = 0; i < seq_len; i++)
        for (size_t h = 0; h < hidden_size; h++) output(i, h) = (ElemType)0;

    // For each token, accumulate expert contributions.
    for (size_t i = 0; i < seq_len; i++) {
        // take input vector
        // For each top expert
        for (size_t k = 0; k < num_top_k; k++) {
            int32_t expert = topk_indices(i, k);
            if (expert < 0) continue;
            auto prob = topk_probs(i, k);
            // --- MLP ---
            // gate/up: [moe_intermediate_size, hidden_size]
            // down:    [hidden_size, moe_intermediate_size]
            // scales: match output dim of corresponding matmul

            // gate
            std::vector<float> gate(moe_intermediate_size);
            constexpr bool gate_scale_is_2d = moeExpertGateProjScale.shape().rank() == 2;
            float gate_scale_val = 1.f;
            if constexpr(gate_scale_is_2d)
            {
                gate_scale_val = (float)moeExpertGateProjScale(expert, 0);
            }
            
            // Check if gate input scale exists (not empty)
            constexpr bool gate_input_scale_is_2d = moeExpertGateInputScale.rank() == 2;
            float gate_input_scale_val = 1.f;
            if constexpr(gate_input_scale_is_2d)
            {
                gate_input_scale_val = (float)moeExpertGateInputScale(expert, 0);
            }
            
            for (size_t d = 0; d < moe_intermediate_size; d++) {
                float acc = 0.f;
                for (size_t h = 0; h < hidden_size; h++) {
                    auto input_val = (float)q(i, h);
                    // Apply input scaling if available

                    if constexpr(gate_input_scale_is_2d) {
                        input_val /= gate_input_scale_val;
                    } else {
                        input_val /= (float)moeExpertGateInputScale(expert, h, 0);
                    }

                    acc += input_val * (float)moeExpertGateProjW(expert, d, h);
                }
                if constexpr(gate_scale_is_2d)
                {
                    acc *= (gate_scale_val * gate_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertGateProjScale(expert, d, 0) * gate_input_scale_val);
                }

                // silu
                auto sig = sigmoid(acc);
                gate[d] = sig * acc; // silu(x) = sigmoid(x) * x
            }
            // up
            std::vector<float> up(moe_intermediate_size);
            // Check if up proj scale is 2D [num_expert, 1] or 3D [num_expert, moe_intermediate_size, 1]
            constexpr bool up_scale_is_2d = (moeExpertUpProjScale.rank() == 2);

            float up_scale_val = 1.f;
            if constexpr(up_scale_is_2d)
            {
                up_scale_val = (float)moeExpertUpProjScale(expert, 0);
            }

            constexpr bool up_input_scale_is_2d = moeExpertUpInputScale.rank() == 2;
            float up_input_scale_val = 1.f;
            if constexpr(up_input_scale_is_2d)
            {
                up_input_scale_val = (float)moeExpertUpInputScale(expert, 0);
            }

            for (size_t d = 0; d < moe_intermediate_size; d++) {
                float acc = 0.f;
                for (size_t h = 0; h < hidden_size; h++) {
                    float input_val = (float)q(i, h);
                    if constexpr(up_input_scale_is_2d) {
                        input_val /= up_input_scale_val;
                    } else {
                        input_val /= (float)moeExpertUpInputScale(expert, h, 0);
                    }

                    acc += input_val * (float)moeExpertUpProjW(expert, d, h);
                }
                if constexpr(up_scale_is_2d)
                {
                    acc *= (up_scale_val * up_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertUpProjScale(expert, d, 0) * up_input_scale_val);
                }

                up[d] = acc;
            }
            // down input = gate * up (elementwise)
            // down: (gate*up)[moe_intermediate_size] @ downW[hidden_size, moe_intermediate_size]
            // Check if down proj scale is 2D [num_expert, 1] or 3D [num_expert, hidden_size, 1]
            constexpr bool down_scale_is_2d = (moeExpertDownProjScale.rank() == 2);
            float down_scale_val = 1.f;
            if constexpr(down_scale_is_2d)
            {
                down_scale_val = (float)moeExpertDownProjScale(expert, 0);
            }
            // Check if down input scale exists (not empty)
            constexpr bool down_input_scale_is_2d = moeExpertDownInputScale.rank() == 2;
            float down_input_scale_val = 1.f;
            if constexpr(down_input_scale_is_2d)
            {
                down_input_scale_val = (float)moeExpertDownInputScale(expert, 0);
            }
            
            for (size_t h = 0; h < hidden_size; h++) {
                float acc = 0.f;
                for (size_t d = 0; d < moe_intermediate_size; d++) {
                    float down_in = gate[d] * up[d];
                    // Apply input scaling if available
                    if constexpr(down_input_scale_is_2d) {
                        down_in /= down_input_scale_val;
                    } else {
                        down_in /= (float)moeExpertDownInputScale(expert, d, 0);
                    }

                    acc += down_in * (float)moeExpertDownProjW(expert, h, d);
                }
                if constexpr(down_scale_is_2d)
                {
                    acc *= (down_scale_val * down_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertDownProjScale(expert, h, 0) * down_input_scale_val);
                }
                output(i, h) += (ElemType)(prob * acc); // accumulate
            }
        }
    }
}

} // namespace detail

template <Tensor TQ, Tensor TRouterIds, Tensor TRouterWeights,
          Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale,
          Tensor TUpInputScale, Tensor TUpProjW, Tensor TUpProjScale,
          class TOut>
void sparse_experts(const TQ &q,
                   const TRouterIds &router_expert_ids,
                   const TRouterWeights &router_expert_weights,
                   const TGateInputScale &moeExpertGateInputScale,
                   const TGateProjW &moeExpertGateProjW,
                   const TGateProjScale &moeExpertGateProjScale,
                   const TDownInputScale &moeExpertDownInputScale,
                   const TDownProjW &moeExpertDownProjW,
                   const TDownProjScale &moeExpertDownProjScale,
                   const TUpInputScale &moeExpertUpInputScale,
                   const TUpProjW &moeExpertUpProjW,
                   const TUpProjScale &moeExpertUpProjScale,
                   TOut &&output,
                   size_t hidden_size,
                   size_t moe_intermediate_size,
                   size_t num_expert,
                   size_t num_top_k,
                   size_t chunk_size) noexcept {
    detail::sparse_experts_impl(q, router_expert_ids, router_expert_weights,
                           moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale,
                           moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale,
                           moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale,
                           hidden_size, moe_intermediate_size,
                           num_expert, num_top_k, chunk_size, output);

}

} // namespace nncase::ntt
