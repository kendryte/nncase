// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Importer
{
    public class DeepSeekV3 : HuggingFaceModel
    {
        public override Tuple<Expr, Expr> DecodeLayer(
            int count,
            Expr hiddenStates,
            Expr pastKeyValues,
            Tuple<Expr, Expr> positionEmbeddings)
        {
            var residual = hiddenStates;
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.input_layernorm.weight");

            // TODO: using `config.attn_implementation` to choose attention implementation
            // self attention
            (hiddenStates, pastKeyValues) = LLMSelfAttention(
                count,
                hiddenStates,
                pastKeyValues,
                positionEmbeddings);
            hiddenStates = residual + hiddenStates;

            // fully Connected
            residual = hiddenStates;
            hiddenStates = LLMLayerNorm(
                hiddenStates,
                $"model.layers.{count}.post_attention_layernorm.weight");

            if (Config.GetNestedValue<string>("n_routed_experts") != null
                && count >= Config.GetNestedValue<int>("first_k_dense_replace")
                && count % Config.GetNestedValue<int>("moe_layer_freq") == 0)
            {
                throw new NotImplementedException("MOE is not supported yet");
            }
            else
            {
                hiddenStates = LLMMlp(count, hiddenStates);
            }

            hiddenStates = residual + hiddenStates;

            var output = hiddenStates;

            return System.Tuple.Create<Expr, Expr>(output, pastKeyValues);
        }
    }
}
