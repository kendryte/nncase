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
//  gate/up proj scale:    [num_expert, moe_intermediate_size, 1]
//  down proj weight:      [num_expert, hidden_size, moe_intermediate_size]
//  down proj scale:       [num_expert, hidden_size, 1]
//  gate (router) weight:  [num_expert, hidden_size]
//  input q:               [seq_len, hidden_size]
//  output:                [seq_len, hidden_size]
//
// All tensors are assumed contiguous in row-major order per existing ntt tensor
// semantics.
//
// NOTE: The evaluator groups tokens by expert for efficiency. Here we process
// token -> topk experts (simpler, less optimal but clearer). Can be optimized
// later by grouping per expert and batching matmuls.

namespace nncase::ntt {

namespace detail {

template <class T> inline T sigmoid(T x) noexcept { return (T)1 / ((T)1 + (T)std::exp((double)-x)); }

template <Tensor TQ, Tensor TGateW, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownProjW, Tensor TDownProjScale, Tensor TUpProjW,
          Tensor TUpProjScale, class TOut>
void qwen3_moe_impl(const TQ &q, const TGateW &moeGateW,
                    const TGateProjW &moeExpertGateProjW,
                    const TGateProjScale &moeExpertGateProjScale,
                    const TDownProjW &moeExpertDownProjW,
                    const TDownProjScale &moeExpertDownProjScale,
                    const TUpProjW &moeExpertUpProjW,
                    const TUpProjScale &moeExpertUpProjScale,
                    size_t hidden_size, size_t /*intermediate_size*/,
                    size_t moe_intermediate_size, size_t num_expert,
                    size_t num_top_k, size_t is_norm_topk_prob, TOut &output) {
    using TElem = typename TQ::element_type;

    const auto seq_len = q.shape()[0_dim];

    // 1. router logits : [seq_len, num_expert] = q [seq_len, hidden_size] @ gateW [num_expert, hidden_size]
    std::vector<TElem> router_logits(seq_len * num_expert);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t e = 0; e < num_expert; e++) {
            TElem acc = (TElem)0;
            for (size_t h = 0; h < hidden_size; h++) {
                acc += q(i, h) * moeGateW(e, h);
            }
            router_logits[i * num_expert + e] = acc; // cast to float already in evaluator, kept same type here.
        }
    }

    // 2. softmax along experts
    std::vector<TElem> router_probs(router_logits); // copy
    for (size_t i = 0; i < seq_len; i++) {
        const TElem *row_in = &router_logits[i * num_expert];
        TElem *row_out = &router_probs[i * num_expert];
        // max
        TElem m = row_in[0];
        for (size_t e = 1; e < num_expert; e++) m = ntt::max(m, row_in[e]);
        // exp and sum
        TElem sum = (TElem)0;
        for (size_t e = 0; e < num_expert; e++) {
            TElem v = row_in[e] - m;
            v = (TElem)std::exp((double)v);
            row_out[e] = v;
            sum += v;
        }
        // normalize
        TElem inv_sum = (TElem)1 / sum;
        for (size_t e = 0; e < num_expert; e++) row_out[e] = row_out[e] * inv_sum;
    }

    // 3. TopK per token
    std::vector<int32_t> topk_indices(seq_len * num_top_k, -1);
    std::vector<TElem> topk_probs(seq_len * num_top_k, (TElem)0);
    for (size_t i = 0; i < seq_len; i++) {
        // Maintain a small local buffer (value, index)
        // Use simple insertion selection for small num_top_k.
        for (size_t e = 0; e < num_expert; e++) {
            TElem p = router_probs[i * num_expert + e];
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
            TElem sum = (TElem)0;
            for (size_t k = 0; k < num_top_k; k++) sum += topk_probs[i * num_top_k + k];
            TElem inv = sum != (TElem)0 ? (TElem)1 / sum : (TElem)0;
            for (size_t k = 0; k < num_top_k; k++) topk_probs[i * num_top_k + k] *= inv;
        }
    }

    // 4. Zero initialize output
    for (size_t i = 0; i < seq_len; i++)
        for (size_t h = 0; h < hidden_size; h++) output(i, h) = (TElem)0;

    // 5. For each token, accumulate expert contributions.
    for (size_t i = 0; i < seq_len; i++) {
        // take input vector
        // For each top expert
        for (size_t k = 0; k < num_top_k; k++) {
            int32_t expert = topk_indices[i * num_top_k + k];
            if (expert < 0) continue;
            TElem prob = topk_probs[i * num_top_k + k];
            // --- MLP ---
            // gate/up: [moe_intermediate_size, hidden_size]
            // down:    [hidden_size, moe_intermediate_size]
            // scales: match output dim of corresponding matmul

            // gate
            std::vector<TElem> gate(moe_intermediate_size);
            for (size_t d = 0; d < moe_intermediate_size; d++) {
                TElem acc = (TElem)0;
                for (size_t h = 0; h < hidden_size; h++) {
                    acc += q(i, h) * moeExpertGateProjW(expert, d, h);
                }
                acc *= moeExpertGateProjScale(expert, d, 0);
                // silu
                TElem sig = sigmoid(acc);
                gate[d] = sig * acc; // silu(x) = sigmoid(x) * x
            }
            // up
            std::vector<TElem> up(moe_intermediate_size);
            for (size_t d = 0; d < moe_intermediate_size; d++) {
                TElem acc = (TElem)0;
                for (size_t h = 0; h < hidden_size; h++) {
                    acc += q(i, h) * moeExpertUpProjW(expert, d, h);
                }
                acc *= moeExpertUpProjScale(expert, d, 0);
                up[d] = acc;
            }
            // down input = gate * up (elementwise)
            // down: (gate*up)[moe_intermediate_size] @ downW[hidden_size, moe_intermediate_size]
            for (size_t h = 0; h < hidden_size; h++) {
                TElem acc = (TElem)0;
                for (size_t d = 0; d < moe_intermediate_size; d++) {
                    TElem down_in = gate[d] * up[d];
                    acc += down_in * moeExpertDownProjW(expert, h, d);
                }
                acc *= moeExpertDownProjScale(expert, h, 0);
                output(i, h) += prob * acc; // accumulate
            }
        }
    }
}

} // namespace detail

// Public API wrapper.
// All tensor rank/shape validation intentionally omitted here (assumed valid
// upstream). Can be added if needed.
template <Tensor TQ, Tensor TGateW, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownProjW, Tensor TDownProjScale, Tensor TUpProjW,
          Tensor TUpProjScale, class TOut>
void qwen3_moe(const TQ &q, const TGateW &moeGateW,
               const TGateProjW &moeExpertGateProjW,
               const TGateProjScale &moeExpertGateProjScale,
               const TDownProjW &moeExpertDownProjW,
               const TDownProjScale &moeExpertDownProjScale,
               const TUpProjW &moeExpertUpProjW,
               const TUpProjScale &moeExpertUpProjScale,
               TOut &&output, size_t layer_id, size_t hidden_size,
               size_t intermediate_size, size_t moe_intermediate_size,
               size_t num_expert, size_t num_top_k, size_t is_norm_topk_prob) noexcept {
    detail::qwen3_moe_impl(q, moeGateW, moeExpertGateProjW,
                           moeExpertGateProjScale, moeExpertDownProjW,
                           moeExpertDownProjScale, moeExpertUpProjW,
                           moeExpertUpProjScale, hidden_size,
                           intermediate_size, moe_intermediate_size,
                           num_expert, num_top_k, is_norm_topk_prob, output);
    
}

} // namespace nncase::ntt
