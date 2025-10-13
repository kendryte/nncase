// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class Qwen3MoEEvaluator : ITypeInferencer<Qwen3MoE>, ICostEvaluator<Qwen3MoE>, IEvaluator<Qwen3MoE>
{
    public IRType Visit(ITypeInferenceContext context, Qwen3MoE target)
    {
        /*
        public static readonly ParameterInfo Q = new(typeof(Qwen3MoE), 0, "q", ParameterKind.Input);
        public static readonly ParameterInfo MoeGateW = new(typeof(Qwen3MoE), 1, "MoeGateW", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertDownProjW = new(typeof(Qwen3MoE), 2, "MoeExpertDownProjW", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertDownProjScale = new(typeof(Qwen3MoE), 3, "MoeExpertDownProjScale", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertGateProjW = new(typeof(Qwen3MoE), 4, "MoeExpertGateProjW", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertGateProjScale = new(typeof(Qwen3MoE), 5, "MoeExpertGateProjScale", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertUpProjW = new(typeof(Qwen3MoE), 6, "MoeExpertUpProjW", ParameterKind.Input);
        public static readonly ParameterInfo MoeExpertUpProjScale = new(typeof(Qwen3MoE), 7, "MoeExpertUpProjScale", ParameterKind.Input);
        */
        var q = context.CheckArgumentType<IRType>(target, Qwen3MoE.Q);

        return q switch
        {
            DistributedType dq => Visit(context, target, dq),
            TensorType tq => Visit(context, target, tq),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, Qwen3MoE target)
    {
        var qType = context.GetArgumentType<IRType>(target, Qwen3MoE.Q);
        var returnType = context.GetReturnType<IRType>();

        // cycles = softmax((q @ k^t) + mask) @ v.
        return new()
        {
            // todo kv cache
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(qType),

            // todo [CostFactorNames.CPUCycles].
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public IValue Visit(IEvaluateContext context, Qwen3MoE target)
    {
        var q = context.GetOrtArgumentValue(target, Qwen3MoE.Q);
        var qType = q.DataType;
        q = q.Cast(OrtDataType.Float);
        var moeGateW = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeGateW).AsTensor());

        var moeExpertDownInputScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertDownInputScale).AsTensor());
        var moeExpertDownProjW = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertDownProjW).AsTensor());
        var moeExpertDownProjScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertDownProjScale).AsTensor());

        var moeExpertGateInputScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertGateInputScale).AsTensor());
        var moeExpertGateProjW = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertGateProjW).AsTensor());
        var moeExpertGateProjScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertGateProjScale).AsTensor());

        var moeExpertUpInputScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertUpInputScale).AsTensor());
        var moeExpertUpProjW = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertUpProjW).AsTensor());
        var moeExpertUpProjScale = GetOrtTensor(context.GetArgumentValue(target, Qwen3MoE.MoeExpertUpProjScale).AsTensor());

        var layerId = target.LayerId;
        var hiddenSize = target.HiddenSize;
        var intermediateSize = target.IntermediateSize;
        var moeIntermediateSize = target.MoEIntermediateSize;
        var numExpert = target.NumExpert;
        var numTopK = target.NumTopK;
        var isNormTopkProb = target.IsNormTopkProb;
        Console.WriteLine($"Qwen3MoE: layerId={layerId}, hiddenSize={hiddenSize}, intermediateSize={intermediateSize}, moeIntermediateSize={moeIntermediateSize}, numExpert={numExpert}, numTopK={numTopK}, isNormTopkProb={isNormTopkProb}");

        Console.WriteLine($"Qwen3MoE: q.shape={string.Join(" ", q.Shape)}, moeGateW.shape={string.Join(" ", moeGateW.Shape)}, " +
            $"moeExpertDownInputScale.shape={string.Join(" ", moeExpertDownInputScale.Shape)}, moeExpertDownProjW.shape={string.Join(" ", moeExpertDownProjW.Shape)}, " +
            $"moeExpertDownProjScale.shape={string.Join(" ", moeExpertDownProjScale.Shape)}, " +
            $"moeExpertGateInputScale.shape={string.Join(" ", moeExpertGateInputScale.Shape)}, moeExpertGateProjW.shape={string.Join(" ", moeExpertGateProjW.Shape)}, " +
            $"moeExpertGateProjScale.shape={string.Join(" ", moeExpertGateProjScale.Shape)}, " +
            $"moeExpertUpInputScale.shape={string.Join(" ", moeExpertUpInputScale.Shape)}, moeExpertUpProjW.shape={string.Join(" ", moeExpertUpProjW.Shape)}, " +
            $"moeExpertUpProjScale.shape={string.Join(" ", moeExpertUpProjScale.Shape)}");

        // var (seqLen, hiddenDim) = (q.Shape[0], q.Shape[1]);
        var seqLen = q.Shape[0];
        var routerLogits = OrtKI.Einsum(new[] { q, moeGateW }, "ls,ds->ld");
        routerLogits = OrtKI.Cast(routerLogits, (int)OrtDataType.Float);
        var routerWeights = OrtKI.Softmax(routerLogits, -1);

        var topkRes = OrtKI.TopK(routerWeights, OrtKISharp.Tensor.MakeTensor(new[] { numTopK }, new[] { 1L }), -1L, 1L, 1L);
        routerWeights = topkRes[0];
        var selectedExperts = topkRes[1];

        if (isNormTopkProb == 1L)
        {
            routerWeights = OrtKI.Div(routerWeights, OrtKI.ReduceSum(routerWeights, Tensor.FromArray(new[] { -1L }).ToOrtTensor(), keepdims: 1L, 0L));
        }

        routerWeights = OrtKI.Cast(routerWeights, (long)q.DataType);

        var finalHiddenStates = OrtKISharp.Tensor.MakeTensor(
            Enumerable.Range(0, (int)(seqLen * hiddenSize)).Select(i => 0).ToArray());

        finalHiddenStates = OrtKI.Reshape(finalHiddenStates, OrtKISharp.Tensor.MakeTensor(new[] { seqLen, hiddenSize }), 0L);
        finalHiddenStates = OrtKI.Cast(finalHiddenStates, (long)q.DataType);

        var expertMask = OrtKI.OneHot(selectedExperts, numExpert, Tensor.From(new[] { 0L, 1L }).ToOrtTensor(), -1L);
        expertMask = OrtKI.Cast(expertMask, (long)q.DataType);
        expertMask = OrtKI.Transpose(expertMask, [2L, 1L, 0L]); // [num_experts, topk, seq_length]

        for (var expertIndex = 0L; expertIndex < numExpert; expertIndex++)
        {
            var singleExpertMask = OrtKI.Slice(expertMask, new[] { expertIndex }, new[] { expertIndex + 1L }, new[] { 0L }, new[] { 1L }); // [num_experts -> 1, topk, seq_length]
            singleExpertMask = OrtKI.Squeeze(singleExpertMask, new[] { 0L }); // [topk, seq_length]
            var nonZero = OrtKI.NonZero(singleExpertMask).ToArray<long>();
            var idx = nonZero[..(nonZero.Length / 2)];
            var topX = nonZero[(nonZero.Length / 2)..];
            if (nonZero.Length == 0)
            {
                continue; // 没有被选中的专家
            }

            var currentState = OrtKI.Gather(q, topX, 0);

            // prepare expertMaskReduceSum
            var expertMaskReduceSum = OrtKI.ReduceSum(singleExpertMask, Tensor.FromArray(new[] { 0L, 1L }).ToOrtTensor(), keepdims: 0L, 0L);

            // // prepare q
            // var qExpand = OrtKI.Unsqueeze(currentState, new[] { 0L });

            // prepare gate matmul
            var gateInputScale = SliceAndSqueeze(moeExpertGateInputScale, expertIndex);
            var gateProjW = SliceAndSqueeze(moeExpertGateProjW, expertIndex);
            var gateProjScale = SliceAndSqueeze(moeExpertGateProjScale, expertIndex);

            // prepare up matmul
            var upInputScale = SliceAndSqueeze(moeExpertUpInputScale, expertIndex);
            var upProjW = SliceAndSqueeze(moeExpertUpProjW, expertIndex);
            var upProjScale = SliceAndSqueeze(moeExpertUpProjScale, expertIndex);

            // prepare down matmul
            var downInputScale = SliceAndSqueeze(moeExpertDownInputScale, expertIndex);
            var downProjW = SliceAndSqueeze(moeExpertDownProjW, expertIndex);
            var downProjScale = SliceAndSqueeze(moeExpertDownProjScale, expertIndex);

            // MLP
            var expertOutput = MLP(currentState, gateInputScale, gateProjW, gateProjScale, upInputScale, upProjW, upProjScale, downInputScale, downProjW, downProjScale, hiddenSize, moeIntermediateSize);

            var weightsForSeq = OrtKI.Gather(routerWeights, Tensor.FromArray(topX).ToOrtTensor(), 0L); // [N, topk]

            var idx2D = OrtKI.Unsqueeze(Tensor.FromArray(idx).ToOrtTensor(), new[] { -1L }); // [N,1]
            var selectedWeights = OrtKI.GatherElements(weightsForSeq, idx2D, 1L); // [N,1]

            expertOutput = OrtKI.Mul(expertOutput, selectedWeights); // [N, hidden]

            var updates = OrtKI.Cast(expertOutput, (long)q.DataType); // [N, hidden]
            var idxCol = OrtKI.Unsqueeze(Tensor.FromArray(topX).ToOrtTensor(), new[] { -1L });        // [N, 1]

            var indices = OrtKI.Tile(idxCol, Tensor.FromArray(new[] { 1L, hiddenSize }).ToOrtTensor()); // [N, hidden]

            finalHiddenStates = OrtKI.ScatterElements(finalHiddenStates, indices, updates, 0L, "add");
        }

        finalHiddenStates = OrtKI.Cast(finalHiddenStates, (long)qType);

        return finalHiddenStates.ToValue();
    }

    private OrtKISharp.Tensor GetOrtTensor(Tensor tensor)
    {
        return IR.F.Tensors.Cast(tensor, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
    }

    private OrtKISharp.Tensor SliceAndSqueeze(OrtKISharp.Tensor tensor, long index)
    {
        // Slices the tensor at the specified index and squeezes the first dimension (expert_batch: 1).
        var slicedTensor = OrtKI.Slice(tensor, new[] { index }, new[] { index + 1L }, new[] { 0L }, new[] { 1L });
        return OrtKI.Squeeze(slicedTensor, new[] { 0L });
    }

    private OrtKISharp.Tensor MLP(OrtKISharp.Tensor q, OrtKISharp.Tensor? gateInputScale, OrtKISharp.Tensor gateProjW, OrtKISharp.Tensor gateProjScale, OrtKISharp.Tensor? upInputScale, OrtKISharp.Tensor upProjW, OrtKISharp.Tensor upProjScale, OrtKISharp.Tensor? downInputScale, OrtKISharp.Tensor downProjW, OrtKISharp.Tensor downProjScale, long hiddenSize, long moeIntermediateSize)
    {
        // gate_proj(q)
        // q: [seq_len, hidden_size]
        // gateInputScale: null or [1]
        // gateW: [hidden_size, moe_intermediate_size]
        // gateWScale: [moe_intermediate_size, 1] or [1]
        var gateStates = Matmul(q, gateInputScale, gateProjW, gateProjScale); // [seq_len, moe_intermediate_size]

        // silu(gate)
        var gateType = gateStates.DataType;
        gateStates = OrtKI.Cast(gateStates, (long)OrtDataType.Float);
        gateStates = OrtKI.Sigmoid(gateStates) * gateStates; // [seq_len, moe_intermediate_size]
        gateStates = OrtKI.Cast(gateStates, (long)gateType);

        // up_proj(q)
        // upW: [hidden_size, moe_intermediate_size]
        var upStates = Matmul(q, upInputScale, upProjW, upProjScale); // [seq_len, moe_intermediate_size]

        // silu(gate(q)) * up(q)
        var downInput = OrtKI.Mul(gateStates, upStates); // [seq_len, moe_intermediate_size]

        // Down(silu(gate(q)) * up(q))
        var downStates = Matmul(downInput, downInputScale, downProjW, downProjScale); // [seq_len, moe_intermediate_size]
        return downStates;
    }

    private OrtKISharp.Tensor Matmul(OrtKISharp.Tensor q, OrtKISharp.Tensor? inputScale, OrtKISharp.Tensor projW, OrtKISharp.Tensor projScale)
    {
        if (inputScale != null)
        {
            q = OrtKI.Div(q, inputScale.Cast(OrtDataType.Float));
        }

        // gateProjScale = OrtKI.Reshape(gateProjScale, OrtKISharp.Tensor.MakeTensor(new[] { 1L, moeIntermediateSize, 1L }), 0L);
        // gateProjW = OrtKI.Mul(gateProjW, gateProjScale);
        var states = OrtKI.Einsum(new[] { q.Cast(OrtDataType.Float), projW.Cast(OrtDataType.Float) }, "hs,ds->hd");
        if (projScale.Rank == 1)
        {
            states = OrtKI.Mul(states, projScale.Cast(OrtDataType.Float)); // [seq_len,     moe_intermediate_size]
        }
        else
        {
            var scale = OrtKI.Transpose(projScale, new[] { 1L, 0L }); // [1, moe_intermediate_size]
            states = OrtKI.Mul(states, scale.Cast(OrtDataType.Float)); // [seq_len, moe_intermediate_size]
        }

        if (inputScale != null)
        {
            states = OrtKI.Mul(states, inputScale.Cast(OrtDataType.Float));
        }

        states = states.Cast(q.DataType); // [seq_len, moe_intermediate_size]
        return states;
    }

    private IRType Visit(ITypeInferenceContext context, Qwen3MoE target, TensorType q)
    {
        return q;
    }

    private IRType Visit(ITypeInferenceContext context, Qwen3MoE target, DistributedType q)
    {
        // TODO: Handle distributed type inference
        // For now, we just return the type as is.
        return q;
    }
}
