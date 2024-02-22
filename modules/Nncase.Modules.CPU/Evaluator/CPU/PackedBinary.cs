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

public sealed class PackedBinaryEvaluator : IEvaluator<PackedBinary>, ITypeInferencer<PackedBinary>, ICostEvaluator<PackedBinary>
{
    public IValue Visit(IEvaluateContext context, PackedBinary target)
    {
        var lhs = context.GetOrtArgumentValue(target, PackedBinary.Lhs); // [x,m/32,k/32,m',k']
        var rhs = context.GetOrtArgumentValue(target, PackedBinary.Rhs); // [x,k/32,n/32,k',n']
        var lanes = new[] { (int)lhs.Shape[^2], (int)rhs.Shape[^1] };
        var outshape = new[] { (int)lhs.Shape[^4], (int)rhs.Shape[^3] };
        var maxRank = System.Math.Max(lhs.Shape.Length, rhs.Shape.Length);
        outshape = Enumerable.Repeat(1L, maxRank - lhs.Shape.Length).Concat(lhs.Shape.SkipLast(4)).
         Zip(Enumerable.Repeat(1L, maxRank - rhs.Shape.Length).Concat(rhs.Shape.SkipLast(4))).
         Select(p => (int)System.Math.Max(p.First, p.Second)).
         Concat(outshape).ToArray();

        lhs = OrtKI.Unsqueeze(lhs, new long[] { -4, -1 }); // [x,m/32,k/32, 1  , m' ,k', 1 ]
        rhs = OrtKI.Unsqueeze(rhs, new long[] { -6, -3 }); // [x, 1  ,k/32,n/32, 1  ,k', n']
        var matmul = OrtKI.Mul(lhs, rhs); // [x, m/32,k/32,n/32,m',k',n']
        matmul = OrtKI.ReduceSum(matmul, new long[] { -2, -5 }, 0, 1);

        return Value.FromTensor(Tensor.FromBytes(new VectorType(DataTypes.Float32, lanes), matmul.BytesBuffer.ToArray(), outshape));
    }

    public IRType Visit(ITypeInferenceContext context, PackedBinary target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedBinary.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedBinary.Rhs);

        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => Visit(context, target, a, b),
            (TensorType a, TensorType b) => Visit(context, target, a, b),
            _ => new InvalidType("not support"),
        };
    }

    private IRType Visit(ITypeInferenceContext context, PackedBinary target, TensorType a, TensorType b)
    {
        int GetDim(Shape s, int lane, int axis, int pad)
        {
            return (s[axis].FixedValue * lane) - pad;
        }

        var rank = System.Math.Max(a.Shape.Rank, b.Shape.Rank);
        var leftA = rank - a.Shape.Rank;
        var leftB = rank - b.Shape.Rank;
        int ToA(int i) => leftB - leftA + i;
        int ToB(int i) => leftA - leftB + i;
        // the pack can't on the broadcast axis
        switch (a.DType, b.DType)
        {
            case (VectorType va, VectorType vb):
                {
                    for (int i = 0; i < va.Lanes.Count; i++)
                    {
                        ToA(target.LhsPackedAxes[i])
                    }
                    // for (int i = 0; i < ; i++)
                    // {

                    // }
                }
                break;
            case (VectorType va, PrimType pb):
                {

                }
                break;
            default:
                break;
        };
    }

    private IRType Visit(ITypeInferenceContext context, PackedBinary target, DistributedType a, DistributedType b) => throw new NotImplementedException();

    public Cost Visit(ICostEvaluateContext context, PackedBinary target)
    {
        var lhs = context.GetArgumentType<IRType>(target, PackedBinary.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, PackedBinary.Rhs);
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
