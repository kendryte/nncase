﻿// Copyright (c) Canaan Inc. All rights reserved.
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

        var outRank = context.CurrentCall.CheckedShape.Rank;
        var outLanes = Array.Empty<int>();
        var outShape = Array.Empty<int>();
        var axes = Array.Empty<int>();
        if (target.LhsPackedAxes.Count == 0 && target.RhsPackedAxes.Count == 1)
        {
            outLanes = new[] { (int)rhs.Shape[^1] };
            outShape = new[] { (int)lhs.Shape[^2], (int)rhs.Shape[^2] };
            axes = new[] { outRank - 1 };
        }
        else if (target.LhsPackedAxes.Count == 1 && target.RhsPackedAxes.Count == 0)
        {
            outLanes = new[] { (int)lhs.Shape[^1] };
            outShape = new[] { (int)lhs.Shape[^3], (int)rhs.Shape[^1] };
            axes = new[] { outRank - 2 };
        }
        else if (target.LhsPackedAxes.Count == 1 && target.RhsPackedAxes.Count == 1)
        {
            if (target.LhsPackedAxes[0] == lhs.Shape.Length - 2 && target.RhsPackedAxes[0] == rhs.Shape.Length - 3)
            {
                outLanes = Array.Empty<int>();
                axes = Array.Empty<int>();
            }
            else
            {
                outLanes = new[] { (int)lhs.Shape[^1], (int)rhs.Shape[^1] };
                axes = new[] { outRank - 2, outRank - 1 };
            }

            outShape = new[] { (int)lhs.Shape[^3], (int)rhs.Shape[^2] };
        }
        else if (target.LhsPackedAxes.Count == 1 && target.RhsPackedAxes.Count == 2)
        {
            outLanes = new[] { (int)rhs.Shape[^1] };
            outShape = new[] { (int)lhs.Shape[^3], (int)rhs.Shape[^3] };
            axes = new[] { outRank - 1 };
        }
        else if (target.LhsPackedAxes.Count == 2 && target.RhsPackedAxes.Count == 1)
        {
            outLanes = new[] { (int)lhs.Shape[^2] };
            outShape = new[] { (int)lhs.Shape[^4], (int)rhs.Shape[^2] };
            axes = new[] { outRank - 2 };
        }
        else if (target.LhsPackedAxes.Count == 2 && target.RhsPackedAxes.Count == 2)
        {
            outLanes = new[] { (int)lhs.Shape[^2], (int)rhs.Shape[^1] };
            outShape = new[] { (int)lhs.Shape[^4], (int)rhs.Shape[^3] };
            axes = new[] { outRank - 2, outRank - 1 };
        }
        else
        {
            throw new NotImplementedException("PackedMatMul with more than 2 packed axes is not supported.");
        }

        var maxRank = System.Math.Max(lhs.Shape.Length - target.LhsPackedAxes.Count, rhs.Shape.Length - target.RhsPackedAxes.Count);
        outShape = Enumerable.Repeat(1L, maxRank - lhs.Shape.Length + target.LhsPackedAxes.Count).Concat(lhs.Shape.SkipLast(2 + target.LhsPackedAxes.Count)).
         Zip(Enumerable.Repeat(1L, maxRank - rhs.Shape.Length + target.RhsPackedAxes.Count).Concat(rhs.Shape.SkipLast(2 + target.RhsPackedAxes.Count))).
         Select(p => (int)System.Math.Max(p.First, p.Second)).
         Concat(outShape).ToArray();

        foreach (var axis in target.LhsPackedAxes.Reverse())
        {
            lhs = lhs.Unpack(axis);
        }

        foreach (var axis in target.RhsPackedAxes.Reverse())
        {
            rhs = rhs.Unpack(axis);
        }

        var matmul = OrtKI.MatMul(lhs, rhs);
        if (outLanes.Length > 0)
        {
            foreach (var (lane, axis) in outLanes.Zip(axes))
            {
                matmul = matmul.Pack(lane, axis);
            }
        }

        return Value.FromTensor(Tensor.FromBytes(outLanes.Length == 0 ? DataTypes.Float32 : new VectorType(DataTypes.Float32, outLanes), matmul.BytesBuffer.ToArray(), outShape));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Rhs);

        bool CheckPackAxes(Shape lhs, Shape rhs)
        {
            bool valid = true;
            switch (target.LhsPackedAxes.Count, target.RhsPackedAxes.Count)
            {
                case (0, 1):
                    if (target.RhsPackedAxes[0] != rhs.Rank - 1)
                    {
                        valid = false;
                    }

                    break;
                case (1, 0):
                    if (target.LhsPackedAxes[0] != lhs.Rank - 2)
                    {
                        valid = false;
                    }

                    break;
                case (1, 1):
                    if (!((target.LhsPackedAxes[0] == lhs.Rank - 1 && target.RhsPackedAxes[0] == rhs.Rank - 2) ||
                        (target.LhsPackedAxes[0] == lhs.Rank - 2 && target.RhsPackedAxes[0] == rhs.Rank - 1)))
                    {
                        valid = false;
                    }

                    break;
                case (1, 2):
                    if (target.LhsPackedAxes[0] != lhs.Rank - 1 || target.RhsPackedAxes[0] != rhs.Rank - 2 || target.RhsPackedAxes[1] != rhs.Rank - 1)
                    {
                        valid = false;
                    }

                    break;
                case (2, 1):
                    if (target.LhsPackedAxes[0] != lhs.Rank - 2 || target.LhsPackedAxes[1] != lhs.Rank - 1 || target.RhsPackedAxes[0] != rhs.Rank - 2)
                    {
                        valid = false;
                    }

                    break;
                case (2, 2):
                    if (target.LhsPackedAxes[0] != lhs.Rank - 2 || target.LhsPackedAxes[1] != lhs.Rank - 1)
                    {
                        valid = false;
                    }

                    if (target.RhsPackedAxes[0] != rhs.Rank - 2 || target.RhsPackedAxes[1] != rhs.Rank - 1)
                    {
                        valid = false;
                    }

                    break;
                default:
                    valid = false;
                    break;
            }

            return valid;
        }

        IRType rType;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                if (!CheckPackAxes(a.TensorType.Shape, b.TensorType.Shape))
                {
                    goto ERROR;
                }

                {
                    bool packingK = target.LhsPackedAxes.Count == 1 && target.RhsPackedAxes.Count == 1 &&
                     target.LhsPackedAxes[0] == a.TensorType.Shape.Rank - 1 && target.RhsPackedAxes[0] == b.TensorType.Shape.Rank - 2;
                    rType = Math.MatMulEvaluator.VisitDistributedType(a, b, packingK);
                }

                break;
            case (TensorType a, TensorType b):
                if (!CheckPackAxes(a.Shape, b.Shape))
                {
                    goto ERROR;
                }

                {
                    bool packingK = target.LhsPackedAxes.Count == 1 && target.RhsPackedAxes.Count == 1 &&
                     target.LhsPackedAxes[0] == a.Shape.Rank - 1 && target.RhsPackedAxes[0] == b.Shape.Rank - 2;
                    rType = Math.MatMulEvaluator.VisitTensorType(a, b, packingK);
                }

                break;
            default:
            ERROR: rType = new InvalidType($"lhs: {lhs}, rhs: {rhs} not support");
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
