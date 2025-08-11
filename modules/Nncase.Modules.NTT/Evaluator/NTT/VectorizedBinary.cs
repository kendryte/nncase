// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class VectorizedBinaryEvaluator : IEvaluator<VectorizedBinary>, ITypeInferencer<VectorizedBinary>, ICostEvaluator<VectorizedBinary>
{
    internal enum DimKind : int
    {
        E, // elemwise
        B, // broadcast
    }

    public IValue Visit(IEvaluateContext context, VectorizedBinary target)
    {
        var a = context.GetOrtArgumentValue(target, VectorizedBinary.Lhs);
        var b = context.GetOrtArgumentValue(target, VectorizedBinary.Rhs);
        _ = System.Math.Max(target.LhsVectorizedAxes.Count, target.RhsVectorizedAxes.Count);

        var maxLaneSize = System.Math.Max(target.LhsVectorizedAxes.Count, target.RhsVectorizedAxes.Count);
        if (target.LhsVectorizedAxes.Count < maxLaneSize)
        {
            a = OrtKI.Unsqueeze(a, Enumerable.Range(-maxLaneSize, maxLaneSize - target.LhsVectorizedAxes.Count).Select(a => (long)a).ToArray());
        }

        if (target.RhsVectorizedAxes.Count < maxLaneSize)
        {
            b = OrtKI.Unsqueeze(b, Enumerable.Range(-maxLaneSize, maxLaneSize - target.RhsVectorizedAxes.Count).Select(a => (long)a).ToArray());
        }

        var binary = target.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            BinaryOp.Max => OrtKI.Max([a, b]),
            _ => throw new ArgumentOutOfRangeException(target.BinaryOp.ToString()),
        };

        var outShape = context.Evaluate(context.CurrentCall.CheckedShape).AsTensor().ToArray<long>();
        return Value.FromTensor(Tensor.FromBytes(context.CurrentCall.CheckedDataType, binary.BytesBuffer.ToArray(), outShape));
    }

    public IRType Visit(ITypeInferenceContext context, VectorizedBinary target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, VectorizedBinary.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, VectorizedBinary.Rhs);

        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => Visit(target, a, b),
            (TensorType a, TensorType b) => Visit(target, a, b),
            _ => new InvalidType("not support"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, VectorizedBinary target)
    {
        var lhs = context.GetArgumentType<IRType>(target, VectorizedBinary.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, VectorizedBinary.Rhs);
        var outputType = context.GetReturnType<IRType>();

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: RankedShape lhsShape })
        {
            macPerElement = !lhsShape.IsScalar && lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
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

    private IRType Visit(VectorizedBinary target, TensorType a, TensorType b)
    {
#if false
        var rank = System.Math.Max(a.Shape.Rank, b.Shape.Rank);
        var outShape = new Dimension[rank];
        var lhsOrginShape = a.Shape.ToArray();
        var rhsOrginShape = b.Shape.ToArray();
        for (int i = 0; i < target.LhsVectorizedAxes.Count; i++)
        {
            lhsOrginShape[target.LhsVectorizedAxes[i]] = (lhsOrginShape[target.LhsVectorizedAxes[i]] * ((VectorType)a.DType).Lanes[i]) - target.LhsPadedNums[i];
        }

        for (int i = 0; i < target.RhsVectorizedAxes.Count; i++)
        {
            rhsOrginShape[target.RhsVectorizedAxes[i]] = (rhsOrginShape[target.RhsVectorizedAxes[i]] * ((VectorType)b.DType).Lanes[i]) - target.RhsPadedNums[i];
        }

        var orginKinds = new DimKind[rank];

        for (int i = -1; i >= -rank; i--)
        {
            var aAxis = a.Shape.Rank + i;
            var bAxis = b.Shape.Rank + i;
            switch (aAxis, bAxis)
            {
                case ( < 0, _):
                    outShape[rank + i] = b.Shape[bAxis];
                    orginKinds[rank + i] = DimKind.B;
                    break;
                case (_, < 0):
                    outShape[rank + i] = a.Shape[aAxis];
                    orginKinds[rank + i] = DimKind.B;
                    break;
                case ( >= 0, >= 0):
                    switch (lhsOrginShape[aAxis], rhsOrginShape[bAxis])
                    {
                        case (Dimension l, Dimension r) when l == r:
                            outShape[rank + i] = a.Shape[aAxis];
                            orginKinds[rank + i] = DimKind.E;
                            break;
                        case (Dimension l, _) when l.IsFixed && l.FixedValue == 1:
                            outShape[rank + i] = b.Shape[bAxis];
                            orginKinds[rank + i] = DimKind.B;
                            break;
                        case (_, Dimension r) when r.IsFixed && r.FixedValue == 1:
                            outShape[rank + i] = a.Shape[aAxis];
                            orginKinds[rank + i] = DimKind.B;
                            break;
                        default:
                            return new InvalidType("vectorized binary not support dim");
                    }

                    break;
                default:
                    throw new NotSupportedException();
            }
        }

        // second check the dtype.
        DataType dataType;
        switch (a.DType, b.DType)
        {
            case (VectorType va, VectorType vb):
                {
                    var lanes = System.Math.Max(va.Lanes.Count, vb.Lanes.Count);
                    var valid = true;
                    for (int i = -1; i >= -lanes; --i)
                    {
                        var ai = va.Lanes.Count + i;
                        var bi = vb.Lanes.Count + i;
                        switch (ai, bi)
                        {
                            case ( < 0, _):
                                valid &= orginKinds[target.RhsVectorizedAxes[bi] - b.Shape.Rank + rank] == DimKind.B && rhsOrginShape[target.RhsVectorizedAxes[bi]] != 1;
                                break;
                            case (_, < 0):
                                valid &= orginKinds[target.LhsVectorizedAxes[ai] - a.Shape.Rank + rank] == DimKind.B && lhsOrginShape[target.LhsVectorizedAxes[ai]] != 1;
                                break;
                            case ( >= 0, >= 0):
                                var laxis = target.LhsVectorizedAxes[ai] - a.Shape.Rank + rank;
                                var raxis = target.RhsVectorizedAxes[bi] - b.Shape.Rank + rank;
                                valid &= lhsOrginShape[target.LhsVectorizedAxes[ai]] == rhsOrginShape[target.RhsVectorizedAxes[bi]] && laxis == raxis && orginKinds[laxis] == orginKinds[raxis] && orginKinds[raxis] == DimKind.E;
                                break;
                        }
                    }

                    if (valid)
                    {
                        dataType = va.Lanes.Count >= vb.Lanes.Count ? va : vb;
                    }
                    else
                    {
                        return new InvalidType("can't vectorize on the broadcast axis!");
                    }
                }

                break;
            case (VectorType va, PrimType pb):
                if (va.ElemType != pb)
                {
                    return new InvalidType("Shape Can't Broadcast");
                }

                dataType = va;
                break;
            case (PrimType pa, VectorType vb):
                if (vb.ElemType != pa)
                {
                    return new InvalidType("Shape Can't Broadcast");
                }

                dataType = vb;
                break;
            default:
                return new InvalidType("Shape Can't Broadcast");
        }

        return new TensorType(dataType, outShape);
#endif

        var broadcastType = TypeInference.BroadcastType(a, b);
        return broadcastType;
    }

    private IRType Visit(VectorizedBinary target, DistributedType a, DistributedType b)
    {
        if (a.Placement != b.Placement)
        {
            return new InvalidType("lhs rhs have different placement");
        }

        var rType = Visit(target, a.TensorType, b.TensorType);
        if (rType is not TensorType tensorType)
        {
            return rType;
        }

        return Math.BinaryEvaluator.CheckSBP(target.BinaryOp, tensorType, a, b);
    }
}
#pragma warning restore SA1008 // Opening parenthesis should be spaced correctly
