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
using Nncase.IR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class PackedBinaryEvaluator : IEvaluator<PackedBinary>, ITypeInferencer<PackedBinary>, ICostEvaluator<PackedBinary>
{
    internal enum DimKind : int
    {
        E, // elemwise
        B, // broadcast
    }

    public IValue Visit(IEvaluateContext context, PackedBinary target)
    {
        var a = context.GetOrtArgumentValue(target, PackedBinary.Lhs);
        var b = context.GetOrtArgumentValue(target, PackedBinary.Rhs);
        _ = System.Math.Max(target.LhsPackedAxes.Count, target.RhsPackedAxes.Count);

        switch (target.LhsPackedAxes.Count, target.RhsPackedAxes.Count)
        {
            case (2, 1):
                b = OrtKI.Unsqueeze(b, new long[] { -2 });
                break;
            case (1, 2):
                a = OrtKI.Unsqueeze(a, new long[] { -2 });
                break;
            default:
                break;
        }

        var binary = target.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            _ => throw new ArgumentOutOfRangeException(target.BinaryOp.ToString()),
        };

        return Value.FromTensor(Tensor.FromBytes(context.CurrentCall.CheckedDataType, binary.BytesBuffer.ToArray(), context.CurrentCall.CheckedShape));
    }

    public IRType Visit(ITypeInferenceContext context, PackedBinary target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedBinary.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedBinary.Rhs);

        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => Visit(target, a, b),
            (TensorType a, TensorType b) => Visit(target, a, b),
            _ => new InvalidType("not support"),
        };
    }

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

    private IRType Visit(PackedBinary target, TensorType a, TensorType b)
    {
        var rank = System.Math.Max(a.Shape.Rank, b.Shape.Rank);
        var outShape = new int[rank];
        var lhsOrginShape = a.Shape.ToValueArray();
        var rhsOrginShape = b.Shape.ToValueArray();
        for (int i = 0; i < target.LhsPackedAxes.Count; i++)
        {
            lhsOrginShape[target.LhsPackedAxes[i]] = (lhsOrginShape[target.LhsPackedAxes[i]] * ((VectorType)a.DType).Lanes[i]) - target.LhsPadedNums[i];
        }

        for (int i = 0; i < target.RhsPackedAxes.Count; i++)
        {
            rhsOrginShape[target.RhsPackedAxes[i]] = (rhsOrginShape[target.RhsPackedAxes[i]] * ((VectorType)b.DType).Lanes[i]) - target.RhsPadedNums[i];
        }

        var orginKinds = new DimKind[rank];

        for (int i = -1; i >= -rank; i--)
        {
            var aAxis = a.Shape.Rank + i;
            var bAxis = b.Shape.Rank + i;
            switch (aAxis, bAxis)
            {
                case ( < 0, _):
                    outShape[rank + i] = b.Shape[bAxis].FixedValue;
                    orginKinds[rank + i] = DimKind.B;
                    break;
                case (_, < 0):
                    outShape[rank + i] = a.Shape[aAxis].FixedValue;
                    orginKinds[rank + i] = DimKind.B;
                    break;
                case ( >= 0, >= 0):
                    switch (lhsOrginShape[aAxis], rhsOrginShape[bAxis])
                    {
                        case (int l, int r) when l == r:
                            outShape[rank + i] = a.Shape[aAxis].FixedValue;
                            orginKinds[rank + i] = DimKind.E;
                            break;
                        case (1, _):
                            outShape[rank + i] = b.Shape[bAxis].FixedValue;
                            orginKinds[rank + i] = DimKind.B;
                            break;
                        case (_, 1):
                            outShape[rank + i] = a.Shape[aAxis].FixedValue;
                            orginKinds[rank + i] = DimKind.B;
                            break;
                        default:
                            return new InvalidType("packed binary not support dim");
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
                                valid &= orginKinds[target.RhsPackedAxes[bi] - b.Shape.Rank + rank] == DimKind.B && rhsOrginShape[target.RhsPackedAxes[bi]] != 1;
                                break;
                            case (_, < 0):
                                valid &= orginKinds[target.LhsPackedAxes[ai] - a.Shape.Rank + rank] == DimKind.B && lhsOrginShape[target.LhsPackedAxes[ai]] != 1;
                                break;
                            case ( >= 0, >= 0):
                                var laxis = target.LhsPackedAxes[ai] - a.Shape.Rank + rank;
                                var raxis = target.RhsPackedAxes[bi] - b.Shape.Rank + rank;
                                valid &= lhsOrginShape[target.LhsPackedAxes[ai]] == rhsOrginShape[target.RhsPackedAxes[bi]] && laxis == raxis && orginKinds[laxis] == orginKinds[raxis] && orginKinds[raxis] == DimKind.E;
                                break;
                        }
                    }

                    if (valid)
                    {
                        dataType = va.Lanes.Count >= vb.Lanes.Count ? va : vb;
                    }
                    else
                    {
                        return new InvalidType("can't pack on the broadcast axis!");
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
    }

    private IRType Visit(PackedBinary target, DistributedType a, DistributedType b)
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
