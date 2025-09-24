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
        var lhs = context.GetArgumentValueAsTensor(target, VectorizedMatMul.Lhs); // [x,m/32,k/32]<m',k'>
        var rhs = context.GetArgumentValueAsTensor(target, VectorizedMatMul.Rhs); // [x,k/32,n/32]<k',n'>

        lhs = Evaluator.Tensors.UnpackEvaluator.UnpackImpl(lhs, target.LhsVectorizedAxes);
        rhs = Evaluator.Tensors.UnpackEvaluator.UnpackImpl(rhs, target.RhsVectorizedAxes);

        if (target.TransposeA)
        {
            var perm = Enumerable.Range(0, lhs.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            lhs = lhs.Transpose(perm);
        }

        if (target.TransposeB)
        {
            var perm = Enumerable.Range(0, rhs.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            rhs = rhs.Transpose(perm);
        }

        var matmul = Math.MatMulEvaluator.InferValue(lhs.ElementType, lhs, rhs, outputDataType: target.OutputDataType, scale: context.GetArgumentValue(target, VectorizedMatMul.Scale)).AsTensor().ToOrtTensor();

        TensorType outputType = context.CurrentCall.CheckedType switch
        {
            DistributedType dt => dt.TensorType,
            TensorType t => t,
            _ => throw new ArgumentOutOfRangeException(string.Empty),
        };

        if (outputType.DType is VectorType ov)
        {
            matmul = matmul.Pack(0, ov.Lanes, target.GetOutVectorizeAxes(lhs.Rank, rhs.Rank));
        }

        return matmul.ToValue(outputType.DType);
    }

    public IRType Visit(ITypeInferenceContext context, VectorizedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, VectorizedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, VectorizedMatMul.Rhs);
        var scale = context.CheckArgumentType<IRType>(target, VectorizedMatMul.Scale);
        IRType rType;
        string? errorMessage = null;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                {
                    var dimInfo = target.GetDimInfo(a.TensorType.Shape.Rank, b.TensorType.Shape.Rank);
                    (var lhsVectorizeKind, var rhsVectorizeKind) = target.GetVectorizeKind(a.TensorType.Shape.Rank, b.TensorType.Shape.Rank);
                    bool vectorizeK = lhsVectorizeKind == VectorizedMatMul.VectorizeKind.K && rhsVectorizeKind == VectorizedMatMul.VectorizeKind.K;
                    rType = Math.MatMulEvaluator.VisitDistributedType(a, b, scale, vectorizeK, dimInfo, target.TransposeB, target.OutputDataType);
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
                    rType = Math.MatMulEvaluator.VisitTensorType(a, b, scale, vectorizeK, dimInfo, target.OutputDataType);
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
