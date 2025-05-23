// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Xml;
using CommunityToolkit.HighPerformance;
using DryIoc;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Buffer = System.Buffer;
using F = Nncase.IR.F;
using Tuple = System.Tuple;

namespace Nncase.Importer
{
    public class Qwen3 : HuggingFaceModel
    {
        public override Tuple<Call, Call, Call> QKVCompute(int count, Expr hiddenStates, Dimension batchSize, Dimension seqLen, Dimension headDim)
        {
            var hidden_shape = new RankedShape(batchSize, seqLen, -1L, headDim);

            var qProjW = Context!.ConstTensors![$"model.layers.{count}.self_attn.q_proj.weight"];
            Tensor? qProjB = null;
            if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.q_proj.bias"))
            {
                qProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.q_proj.bias"];
            }

            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.q_proj.input_scale", out var ifScaleQ);
            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.q_proj.weight_scale", out var wScaleQ);
            var queryStates = Linear(hiddenStates, qProjW, qProjB, ifScaleQ, wScaleQ, $"model.layers.{count}.self_attn.q_proj");
            queryStates = IR.F.Tensors.Reshape(queryStates, hidden_shape);
            queryStates = LLMLayerNorm(queryStates, $"model.layers.{count}.self_attn.q_norm.weight");

            // batch_size, num_heads, seq_len, head_dim
            queryStates = IR.F.Tensors.Transpose(queryStates, new long[] { 0, 2, 1, 3 });

            var kProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.weight"];
            Tensor? kProjB = null;
            if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.k_proj.bias"))
            {
                kProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.k_proj.bias"];
            }

            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.k_proj.input_scale", out var ifScaleK);
            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.k_proj.weight_scale", out var wScaleK);
            var keyStates = Linear(hiddenStates, kProjW, kProjB, ifScaleK, wScaleK, $"model.layers.{count}.self_attn.k_proj");
            keyStates = IR.F.Tensors.Reshape(keyStates, hidden_shape);
            keyStates = LLMLayerNorm(keyStates, $"model.layers.{count}.self_attn.k_norm.weight");
            keyStates = IR.F.Tensors.Transpose(keyStates, new long[] { 0, 2, 1, 3 });

            var vProjW = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.weight"];
            Tensor? vProjB = null;
            if (Context.ConstTensors!.ContainsKey($"model.layers.{count}.self_attn.v_proj.bias"))
            {
                vProjB = Context.ConstTensors![$"model.layers.{count}.self_attn.v_proj.bias"];
            }

            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.v_proj.input_scale", out var ifScaleV);
            Context.ConstTensors!.TryGetValue($"model.layers.{count}.self_attn.v_proj.weight_scale", out var wScaleV);
            var valueStates = Linear(hiddenStates, vProjW, vProjB, ifScaleV, wScaleV, $"model.layers.{count}.self_attn.v_proj");
            valueStates = IR.F.Tensors.Reshape(valueStates, hidden_shape);
            valueStates = IR.F.Tensors.Transpose(valueStates, new long[] { 0, 2, 1, 3 });
            return System.Tuple.Create(queryStates, keyStates, valueStates);
        }
    }
}
