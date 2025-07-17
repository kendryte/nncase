// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.Importer
{
    public class Qwen3 : HuggingFaceModel
    {
        public override Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Dimension seqLen, Dimension headDim)
        {
            var hidden_shape = new RankedShape(seqLen, -1L, headDim);

            var qProjW = GetWeight($"model.layers.{count}.self_attn.q_proj.weight")!;
            var qProjB = GetWeight($"model.layers.{count}.self_attn.q_proj.bias");

            var ifScaleQ = GetWeight($"model.layers.{count}.self_attn.q_proj.input_scale");
            var wScaleQ = GetWeight($"model.layers.{count}.self_attn.q_proj.weight_scale");
            var queryStates = Linear(hiddenStates, qProjW, qProjB, ifScaleQ, wScaleQ, $"model.layers.{count}.self_attn.q_proj");
            queryStates = IR.F.Tensors.Reshape(queryStates, hidden_shape);
            queryStates = LLMLayerNorm(queryStates, $"model.layers.{count}.self_attn.q_norm.weight");

            // batch_size, num_heads, seq_len, head_dim
            queryStates = IR.F.Tensors.Transpose(queryStates, new long[] { 1, 0, 2 });

            var kProjW = GetWeight($"model.layers.{count}.self_attn.k_proj.weight")!;
            var kProjB = GetWeight($"model.layers.{count}.self_attn.k_proj.bias");

            var ifScaleK = GetWeight($"model.layers.{count}.self_attn.k_proj.input_scale");
            var wScaleK = GetWeight($"model.layers.{count}.self_attn.k_proj.weight_scale");
            var keyStates = Linear(hiddenStates, kProjW, kProjB, ifScaleK, wScaleK, $"model.layers.{count}.self_attn.k_proj");
            keyStates = IR.F.Tensors.Reshape(keyStates, hidden_shape);
            keyStates = LLMLayerNorm(keyStates, $"model.layers.{count}.self_attn.k_norm.weight");
            keyStates = IR.F.Tensors.Transpose(keyStates, new long[] { 1, 0, 2 });

            var vProjW = GetWeight($"model.layers.{count}.self_attn.v_proj.weight")!;
            var vProjB = GetWeight($"model.layers.{count}.self_attn.v_proj.bias");

            var ifScaleV = GetWeight($"model.layers.{count}.self_attn.v_proj.input_scale");
            var wScaleV = GetWeight($"model.layers.{count}.self_attn.v_proj.weight_scale");
            var valueStates = Linear(hiddenStates, vProjW, vProjB, ifScaleV, wScaleV, $"model.layers.{count}.self_attn.v_proj");
            valueStates = IR.F.Tensors.Reshape(valueStates, hidden_shape);
            valueStates = IR.F.Tensors.Transpose(valueStates, new long[] { 1, 0, 2 });
            return System.Tuple.Create(queryStates, keyStates, valueStates);
        }
    }
}
