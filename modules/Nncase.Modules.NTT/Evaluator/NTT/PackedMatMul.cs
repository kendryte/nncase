// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class VectorizedMatMulEvaluator : IEvaluator<VectorizedMatMul>, ITypeInferencer<VectorizedMatMul>, ICostEvaluator<VectorizedMatMul>
{
    public IValue Visit(IEvaluateContext context, VectorizedMatMul target)
    {
        var lhs = context.GetOrtArgumentValue(target, VectorizedMatMul.Lhs); // [x,m/32,k/32,m',k']
        var rhs = context.GetOrtArgumentValue(target, VectorizedMatMul.Rhs); // [x,k/32,n/32,k',n']

        var outRank = context.CurrentCall.CheckedShape.Rank;
        var outLanes = Array.Empty<int>();
        var outShape = Array.Empty<long>();
        var axes = Array.Empty<int>();
        var (lm, lk) = target.TransposeA ? (lhs.Rank - target.RhsVectorizedAxes.Count - 1, lhs.Rank - target.RhsVectorizedAxes.Count - 2) : (lhs.Rank - target.LhsVectorizedAxes.Count - 2, lhs.Rank - target.LhsVectorizedAxes.Count - 1);
        var (rk, rn) = target.TransposeB ? (rhs.Rank - target.RhsVectorizedAxes.Count - 1, rhs.Rank - target.RhsVectorizedAxes.Count - 2) : (rhs.Rank - target.RhsVectorizedAxes.Count - 2, rhs.Rank - target.RhsVectorizedAxes.Count - 1);
        if (target.LhsVectorizedAxes.Count == 0 && target.RhsVectorizedAxes.Count == 1)
        {
            outLanes = new[] { (int)rhs.Shape[^1] };
            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
            axes = new[] { outRank - 1 };
        }
        else if (target.LhsVectorizedAxes.Count == 1 && target.RhsVectorizedAxes.Count == 0)
        {
            outLanes = new[] { (int)lhs.Shape[^1] };
            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
            axes = new[] { outRank - 2 };
        }
        else if (target.LhsVectorizedAxes.Count == 1 && target.RhsVectorizedAxes.Count == 1)
        {
            if (target.LhsVectorizedAxes[0] == lk && target.RhsVectorizedAxes[0] == rk)
            {
                outLanes = Array.Empty<int>();
                axes = Array.Empty<int>();
            }
            else
            {
                outLanes = new[] { (int)lhs.Shape[^1], (int)rhs.Shape[^1] };
                axes = new[] { outRank - 2, outRank - 1 };
            }

            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
        }
        else if (target.LhsVectorizedAxes.Count == 1 && target.RhsVectorizedAxes.Count == 2)
        {
            outLanes = new[] { (int)rhs.Shape[^1] };
            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
            axes = new[] { outRank - 1 };
        }
        else if (target.LhsVectorizedAxes.Count == 2 && target.RhsVectorizedAxes.Count == 1)
        {
            outLanes = new[] { (int)lhs.Shape[^2] };
            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
            axes = new[] { outRank - 2 };
        }
        else if (target.LhsVectorizedAxes.Count == 2 && target.RhsVectorizedAxes.Count == 2)
        {
            outLanes = new[] { (int)lhs.Shape[^2], (int)rhs.Shape[^1] };
            outShape = new[] { lhs.Shape[lm], rhs.Shape[rn] };
            axes = new[] { outRank - 2, outRank - 1 };
        }
        else
        {
            throw new NotImplementedException("VectorizedMatMul with more than 2 vectorized axes is not supported.");
        }

        var maxRank = System.Math.Max(lhs.Shape.Length - target.LhsVectorizedAxes.Count, rhs.Shape.Length - target.RhsVectorizedAxes.Count);
        outShape = Enumerable.Repeat(1L, maxRank - lhs.Shape.Length + target.LhsVectorizedAxes.Count).Concat(lhs.Shape.SkipLast(2 + target.LhsVectorizedAxes.Count)).
         Zip(Enumerable.Repeat(1L, maxRank - rhs.Shape.Length + target.RhsVectorizedAxes.Count).Concat(rhs.Shape.SkipLast(2 + target.RhsVectorizedAxes.Count))).
         Select(p => System.Math.Max(p.First, p.Second)).
         Concat(outShape).ToArray();

        foreach (var axis in target.LhsVectorizedAxes.Reverse())
        {
            lhs = lhs.Devectorize(axis);
        }

        foreach (var axis in target.RhsVectorizedAxes.Reverse())
        {
            rhs = rhs.Devectorize(axis);
        }

        if (target.TransposeA)
        {
            var perm = Enumerable.Range(0, lhs.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            lhs = OrtKI.Transpose(lhs, perm);
        }

        if (target.TransposeB)
        {
            var perm = Enumerable.Range(0, rhs.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            rhs = OrtKI.Transpose(rhs, perm);
        }

        var matmul = Math.MatMulEvaluator.InferValue(lhs.DataType.ToDataType(), lhs.ToTensor(), rhs.ToTensor()).AsTensor().ToOrtTensor();
        if (outLanes.Length > 0)
        {
            foreach (var (lane, axis) in outLanes.Zip(axes))
            {
                matmul = matmul.Vectorize(lane, axis);
            }
        }

        return Value.FromTensor(Tensor.FromBytes(outLanes.Length == 0 ? DataTypes.Float32 : new VectorType(DataTypes.Float32, outLanes), matmul.BytesBuffer.ToArray(), outShape));
    }

    public IRType Visit(ITypeInferenceContext context, VectorizedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, VectorizedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, VectorizedMatMul.Rhs);

        IRType rType;
        string? errorMessage = null;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                {
                    var dimInfo = target.GetDimInfo(a.TensorType.Shape.Rank, b.TensorType.Shape.Rank);
                    (var lhsVectorizeKind, var rhsVectorizeKind) = target.GetVectorizeKind(a.TensorType.Shape.Rank, b.TensorType.Shape.Rank);
                    bool vectorizeK = lhsVectorizeKind == VectorizedMatMul.VectorizeKind.K && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.K;
                    rType = Math.MatMulEvaluator.VisitDistributedType(a, b, vectorizeK, dimInfo, target.TransposeB, target.OutputDataType);
                    if (target.FusedReduce)
                    {
                        rType = Math.MatMulEvaluator.ConvertPartialToBroadcast((DistributedType)rType);
                    }
                }

                break;
            case (TensorType a, TensorType b):
                {
                    var dimInfo = target.GetDimInfo(a.Shape.Rank, b.Shape.Rank);
                    (var lhsVectorizeKind, var rhsVectorizeKind) = target.GetVectorizeKind(a.Shape.Rank, b.Shape.Rank);
                    bool vectorizeK = lhsVectorizeKind == VectorizedMatMul.VectorizeKind.K && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.K;
                    rType = Math.MatMulEvaluator.VisitTensorType(a, b, vectorizeK, dimInfo, target.OutputDataType);
                }

                break;
            default:
                rType = new InvalidType($"lhs: {lhs}, rhs: {rhs}, in {target.DisplayProperty()} not support: {errorMessage}");
                break;
        }

        return rType;
    }

    public Cost Visit(ICostEvaluateContext context, VectorizedMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, VectorizedMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, VectorizedMatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: Shape lhsShape })
        {
            var k = target.TransposeA ? lhsShape.Rank - 2 : lhsShape.Rank - 1;
            macPerElement = lhsShape[k].IsFixed ? (uint)lhsShape[k].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            var k = target.TransposeA ? distributedType.TensorType.Shape.Rank - 2 : distributedType.TensorType.Shape.Rank - 1;
            macPerElement = lhsType.Shape[k].IsFixed ? (uint)lhsType.Shape[k].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }
}
