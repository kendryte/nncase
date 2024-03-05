// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class PackedMatMulEvaluator : IEvaluator<PackedMatMul>, ITypeInferencer<PackedMatMul>, ICostEvaluator<PackedMatMul>
{
    public IValue Visit(IEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetOrtArgumentValue(target, PackedMatMul.Lhs); // [x,m/32,k/32,m',k']
        var rhs = context.GetOrtArgumentValue(target, PackedMatMul.Rhs); // [x,k/32,n/32,k',n']
        var outLanes = new[] { (int)lhs.Shape[^2], (int)rhs.Shape[^1] };
        var outshape = new[] { (int)lhs.Shape[^4], (int)rhs.Shape[^3] };
        var maxRank = System.Math.Max(lhs.Shape.Length, rhs.Shape.Length);
        outshape = Enumerable.Repeat(1L, maxRank - lhs.Shape.Length).Concat(lhs.Shape.SkipLast(4)).
         Zip(Enumerable.Repeat(1L, maxRank - rhs.Shape.Length).Concat(rhs.Shape.SkipLast(4))).
         Select(p => (int)System.Math.Max(p.First, p.Second)).
         Concat(outshape).ToArray();

        foreach (var axis in target.LhsPackedAxes.Reverse())
        {
            lhs = lhs.Unpack(axis);
        }

        foreach (var axis in target.RhsPackedAxes.Reverse())
        {
            rhs = rhs.Unpack(axis);
        }

        // lhs = OrtKI.Unsqueeze(lhs, new long[] { -4, -1 }); // [x,m/32,k/32, 1  , m' ,k', 1 ]
        // rhs = OrtKI.Unsqueeze(rhs, new long[] { -6, -3 }); // [x, 1  ,k/32,n/32, 1  ,k', n']
        // var matmul = OrtKI.Mul(lhs, rhs); // [x, m/32,k/32,n/32,m',k',n']
        // matmul = OrtKI.ReduceSum(matmul, new long[] { -2, -5 }, 0, 1);
        var matmul = OrtKI.MatMul(lhs, rhs);
        foreach (var (lane, axis) in outLanes.Zip(new[] { -2 + outshape.Length, -1 + outshape.Length }))
        {
            matmul = matmul.Pack(lane, axis);
        }

        return Value.FromTensor(Tensor.FromBytes(new VectorType(DataTypes.Float32, outLanes), matmul.BytesBuffer.ToArray(), outshape));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Rhs);

        bool CheckPackAxes(Shape lhs, Shape rhs)
        {
            if (target.LhsPackedAxes.Count != 2 || target.RhsPackedAxes.Count != 2)
            {
                return false;
            }

            if (target.LhsPackedAxes[0] != lhs.Rank - 2 || target.LhsPackedAxes[1] != lhs.Rank - 1)
            {
                return false;
            }

            if (target.RhsPackedAxes[0] != rhs.Rank - 2 || target.RhsPackedAxes[1] != rhs.Rank - 1)
            {
                return false;
            }

            return true;
        }

        IRType rType;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                if (!CheckPackAxes(a.TensorType.Shape, b.TensorType.Shape))
                {
                    goto ERROR;
                }

                rType = Math.MatMulEvaluator.VisitDistributedType(a, b);

                break;
            case (TensorType a, TensorType b):
                if (!CheckPackAxes(a.Shape, b.Shape))
                {
                    goto ERROR;
                }

                rType = Math.MatMulEvaluator.VisitTensorType(a, b);
                break;
            default:
            ERROR: rType = new InvalidType($"{lhs} {rhs} not support");
                break;
        }

        return rType;
    }

    public Cost Visit(ICostEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedMatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: Shape lhsShape })
        {
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            macPerElement = lhsType.Shape[^1].IsFixed ? (uint)lhsType.Shape[^1].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }
}
