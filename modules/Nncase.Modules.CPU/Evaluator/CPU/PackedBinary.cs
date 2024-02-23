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
    public IValue Visit(IEvaluateContext context, PackedBinary target)
    {
        var a = context.GetOrtArgumentValue(target, PackedBinary.Lhs);
        var b = context.GetOrtArgumentValue(target, PackedBinary.Rhs);
        var pRank = System.Math.Max(target.LhsPackedAxes.Count, target.RhsPackedAxes.Count);

        switch (target.LhsPackedAxes.Count, target.RhsPackedAxes.Count)
        {
            case (2, 1):
                b = OrtKI.Unsqueeze(b, new long[] { -2 });
                break;
            case (1, 2):
                a = OrtKI.Unsqueeze(a, new long[] { -2 });
                break;
            case (2, 0):
                b = OrtKI.Unsqueeze(b, new long[] { -1, -1 });
                break;
            case (0, 2):
                a = OrtKI.Unsqueeze(a, new long[] { -1, -1 });
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
        Shape outShape = Shape.Scalar;
        var leftA = rank - a.Shape.Rank;
        var leftB = rank - b.Shape.Rank;

        bool CheckDimBroadCast(int dimA, int dimB, out int dimOut)
        {
            dimOut = System.Math.Max(dimA, dimB);
            return (dimA, dimB) switch
            {
                (int a, int b) when a == b => true,
                (1, _) => true,
                (_, 1) => true,
                _ => false,
            };
        }

        bool CheckBroadCast([MaybeNullWhen(false)] out Shape shape)
        {
            shape = null!;
            var dims = new Dimension[rank];
            for (int i = -1; i >= -rank; i--)
            {
                var aAxis = a.Shape.Rank + i;
                var bAxis = b.Shape.Rank + i;
                switch (aAxis, bAxis)
                {
                    case ( < 0, _):
                        dims[rank + i] = b.Shape[bAxis];
                        break;
                    case (_, < 0):
                        dims[rank + i] = a.Shape[aAxis];
                        break;
                    case ( >= 0, >= 0):
                        if (CheckDimBroadCast(a.Shape[aAxis].FixedValue, b.Shape[bAxis].FixedValue, out int dimOut))
                        {
                            dims[rank + i] = dimOut;
                        }
                        else
                        {
                            return false;
                        }

                        break;
                    default:
                        throw new NotSupportedException();
                }
            }

            shape = new Shape(dims);
            return true;
        }

        // first check shape can broadcast
        switch (a.Shape.Rank, b.Shape.Rank)
        {
            case (0, 0):
                break;
            case ( > 0, 0):
                outShape = a.Shape;
                break;
            case (0, > 0):
                outShape = b.Shape;
                break;
            case ( > 0, > 0):
                if (CheckBroadCast(out var oShape))
                {
                    outShape = oShape;
                }
                else
                {
                    goto ERROR;
                }

                break;
            default:
            ERROR: return new InvalidType("Shape Can't Broadcast");
        }

        // second check the dtype.
        DataType dataType;
        switch (a.DType, b.DType)
        {
            case (VectorType va, VectorType vb):
                {
                    var valid = true;
                    for (int i = -1; i >= System.Math.Max(va.Lanes.Count, vb.Lanes.Count); --i)
                    {
                        var ai = va.Lanes.Count + i;
                        var bi = vb.Lanes.Count + i;
                        switch (ai, bi)
                        {
                            case ( < 0, _):
                            case (_, < 0):
                                break;
                            case ( >= 0, >= 0):
                                var adim = (a.Shape[target.LhsPackedAxes[ai]].FixedValue * va.Lanes[ai]) - target.LhsPadedNums[ai];
                                var bdim = (b.Shape[target.LhsPackedAxes[bi]].FixedValue * vb.Lanes[bi]) - target.RhsPadedNums[bi];
                                valid &= adim == bdim && adim != 1;

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
