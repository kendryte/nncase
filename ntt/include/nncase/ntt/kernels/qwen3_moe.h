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

// A naive reference implementation of Qwen3 MoE forward pass.
// This version focuses on correctness first (no vectorization / sharding
// specializations yet). It follows the Evaluator logic in
// src/Nncase.Evaluator/NN/Qwen3MoE.cs
//
// Shapes assumptions (per expert):
//  gate/up proj weight:   [num_expert, moe_intermediate_size, hidden_size]
//  gate/up proj scale:    [num_expert, moe_intermediate_size, 1] or [num_expert, 1]
//  gate/up input scale:   [num_expert, hidden_size, 1] or [num_expert, 1] (optional, can be empty)
//  down proj weight:      [num_expert, hidden_size, moe_intermediate_size]
//  down proj scale:       [num_expert, hidden_size, 1] or [num_expert, 1]
//  down input scale:      [num_expert, moe_intermediate_size, 1] or [num_expert, 1] (optional, can be empty)
//  gate (router) weight:  [num_expert, hidden_size]
//  input q:               [seq_len, hidden_size]
//  output:                [seq_len, hidden_size]
//
// All tensors are assumed contiguous in row-major order per existing ntt tensor
// semantics.
//
// InputScale handling: If InputScale tensors are not empty, they are applied 
// by dividing the input before matrix multiplication, then the result is 
// multiplied by the corresponding proj scale after matrix multiplication.
// Formula: output = (input / inputScale) @ weight * projScale
//
// NOTE: The evaluator groups tokens by expert for efficiency. Here we process
// token -> topk experts (simpler, less optimal but clearer). Can be optimized
// later by grouping per expert and batching matmuls.

namespace nncase::ntt {

namespace detail {

template <class T> inline T sigmoid(T x) noexcept { return (T)1 / ((T)1 + (T)std::exp((double)-x)); }

template <Tensor TQ, Tensor TGateW, Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale, Tensor TUpInputScale, Tensor TUpProjW,
          Tensor TUpProjScale, class TOut>
void qwen3_moe_impl(const TQ &q, const TGateW &moeGateW,
                    const TGateInputScale &moeExpertGateInputScale,
                    const TGateProjW &moeExpertGateProjW,
                    const TGateProjScale &moeExpertGateProjScale,
                    const TDownInputScale &moeExpertDownProjInputScale,
                    const TDownProjW &moeExpertDownProjW,
                    const TDownProjScale &moeExpertDownProjScale,
                    const TUpInputScale &moeExpertUpProjInputScale,
                    const TUpProjW &moeExpertUpProjW,
                    const TUpProjScale &moeExpertUpProjScale,
                    size_t hidden_size, size_t /*intermediate_size*/,
                    size_t moe_intermediate_size, size_t num_expert,
                    size_t num_top_k, size_t is_norm_topk_prob, TOut &output) {
    using EndTElem = typename TQ::element_type;

    const auto seq_len = q.shape()[0_dim];

    // 1. router logits : [seq_len, num_expert] = q [seq_len, hidden_size] @ gateW [num_expert, hidden_size]
    std::vector<float> router_logits(seq_len * num_expert);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t e = 0; e < num_expert; e++) {
            float acc = 0.f;
            for (size_t h = 0; h < hidden_size; h++) {
                acc += (float)q(i, h) * (float)moeGateW(e, h);
            }
            router_logits[i * num_expert + e] = acc; // cast to float already in evaluator, kept same type here.
        }
    }

    // 2. softmax along experts
    std::vector<float> router_probs(router_logits); // copy
    for (size_t i = 0; i < seq_len; i++) {
        const auto *row_in = &router_logits[i * num_expert];
        auto *row_out = &router_probs[i * num_expert];
        // max
        auto m = row_in[0];
        for (size_t e = 1; e < num_expert; e++) m = ntt::max(m, row_in[e]);
        // exp and sum
        float sum = 0.f;
        for (size_t e = 0; e < num_expert; e++) {
            float v = row_in[e] - m;
            v = std::exp(v);
            row_out[e] = v;
            sum += v;
        }
        // normalize
        auto inv_sum = 1.f / sum;
        for (size_t e = 0; e < num_expert; e++) row_out[e] = row_out[e] * inv_sum;
    }

    // 3. TopK per token
    std::vector<int32_t> topk_indices(seq_len * num_top_k, -1);
    std::vector<float> topk_probs(seq_len * num_top_k, 0.f);
    for (size_t i = 0; i < seq_len; i++) {
        // Maintain a small local buffer (value, index)
        // Use simple insertion selection for small num_top_k.
        for (size_t e = 0; e < num_expert; e++) {
            float p = router_probs[i * num_expert + e];
            // find position
            size_t pos = num_top_k;
            for (size_t k = 0; k < num_top_k; k++) {
                if (pos == num_top_k && (topk_indices[i * num_top_k + k] == -1 || p > topk_probs[i * num_top_k + k])) {
                    pos = k; break; }
            }
            if (pos < num_top_k) {
                // shift right
                for (size_t k = num_top_k - 1; k > pos; k--) {
                    topk_probs[i * num_top_k + k] = topk_probs[i * num_top_k + k - 1];
                    topk_indices[i * num_top_k + k] = topk_indices[i * num_top_k + k - 1];
                }
                topk_probs[i * num_top_k + pos] = p;
                topk_indices[i * num_top_k + pos] = (int32_t)e;
            }
        }
        // optional renormalization
        if (is_norm_topk_prob) {
            float sum = 0.f;
            for (size_t k = 0; k < num_top_k; k++) sum += topk_probs[i * num_top_k + k];
            float inv = sum != 0.f ? 1.f / sum : 0.f;
            for (size_t k = 0; k < num_top_k; k++) topk_probs[i * num_top_k + k] *= inv;
        }
    }

    // 4. Zero initialize output
    for (size_t i = 0; i < seq_len; i++)
        for (size_t h = 0; h < hidden_size; h++) output(i, h) = (EndTElem)0;

    // 5. For each token, accumulate expert contributions.
    for (size_t i = 0; i < seq_len; i++) {
        // take input vector
        // For each top expert
        for (size_t k = 0; k < num_top_k; k++) {
            int32_t expert = topk_indices[i * num_top_k + k];
            if (expert < 0) continue;
            auto prob = topk_probs[i * num_top_k + k];
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

            constexpr bool up_input_scale_is_2d = moeExpertUpProjInputScale.rank() == 2;
            float up_input_scale_val = 1.f;
            if constexpr(up_input_scale_is_2d)
            {
                up_input_scale_val = (float)moeExpertUpProjInputScale(expert, 0);
            }

            for (size_t d = 0; d < moe_intermediate_size; d++) {
                float acc = 0.f;
                for (size_t h = 0; h < hidden_size; h++) {
                    float input_val = (float)q(i, h);
                    if constexpr(up_input_scale_is_2d) {
                        input_val /= up_input_scale_val;
                    } else {
                        input_val /= (float)moeExpertUpProjInputScale(expert, h, 0);
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
            constexpr bool down_input_scale_is_2d = moeExpertDownProjInputScale.rank() == 2;
            float down_input_scale_val = 1.f;
            if constexpr(down_input_scale_is_2d)
            {
                down_input_scale_val = (float)moeExpertDownProjInputScale(expert, 0);
            }
            
            for (size_t h = 0; h < hidden_size; h++) {
                float acc = 0.f;
                for (size_t d = 0; d < moe_intermediate_size; d++) {
                    float down_in = gate[d] * up[d];
                    // Apply input scaling if available
                    if constexpr(down_input_scale_is_2d) {
                        down_in /= down_input_scale_val;
                    } else {
                        down_in /= (float)moeExpertDownProjInputScale(expert, d, 0);
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
                output(i, h) += (EndTElem)(prob * acc); // accumulate
            }
        }
    }
}

} // namespace detail

// Public API wrapper.
// All tensor rank/shape validation intentionally omitted here (assumed valid
// upstream). Can be added if needed.
template <Tensor TQ, Tensor TGateW, Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale, Tensor TUpInputScale, Tensor TUpProjW, Tensor TUpProjScale, class TOut>
void qwen3_moe(const TQ &q, const TGateW &moeGateW,
               const TGateInputScale &moeExpertGateInputScale,
                const TGateProjW &moeExpertGateProjW,
               const TGateProjScale &moeExpertGateProjScale,
               const TDownInputScale &moeExpertDownInputScale,
               const TDownProjW &moeExpertDownProjW,
               const TDownProjScale &moeExpertDownProjScale,
               const TUpInputScale &moeExpertUpInputScale,
               const TUpProjW &moeExpertUpProjW,
               const TUpProjScale &moeExpertUpProjScale,
               TOut &&output, size_t /* layer_id */, size_t hidden_size,
               size_t intermediate_size, size_t moe_intermediate_size,
               size_t num_expert, size_t num_top_k, size_t is_norm_topk_prob) noexcept {
    detail::qwen3_moe_impl(q, moeGateW, 
                           moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale,
                           moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale,
                           moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale,
                           hidden_size,
                           intermediate_size, moe_intermediate_size,
                           num_expert, num_top_k, is_norm_topk_prob, output);

}

} // namespace nncase::ntt
