// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class PackedMatMulEvaluator : IEvaluator<PackedMatMul>, ITypeInferencer<PackedMatMul>, ICostEvaluator<PackedMatMul>
{
    public IValue Visit(IEvaluateContext context, PackedMatMul target)
    {
        var lhs = context.GetOrtArgumentValue(target, PackedMatMul.Lhs); // [x, k, m]
        var rhs = context.GetArgumentValueAsTensor(target, PackedMatMul.Rhs); // [x, n/4/8, k, 4, 8]
        var rhsOrt = rhs.ToOrtTensor();

        var rhsVectorType = (VectorType)rhs.ElementType;
        var nr = rhsVectorType.Lanes[0];
        var lanes = rhsVectorType.Lanes[1];
        var outRank = context.CurrentCall.CheckedShape.Rank;

        // 1. Unpack B
        var rN = rhs.Rank - 2;
        rhsOrt = rhsOrt.Unpack(rhsVectorType.Lanes.Count, [rN, rN]);

        // 2. Transpose B
        {
            var perm = Enumerable.Range(0, rhsOrt.Rank).Select(i => (long)i).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            rhsOrt = OrtKI.Transpose(rhsOrt, perm);
        }

        var matmul = Math.MatMulEvaluator.InferValue(lhs.DataType.ToDataType(), lhs.ToTensor(), rhsOrt.ToTensor()).AsTensor().ToOrtTensor();
        var cN = matmul.Rank - 1;
        matmul = matmul.Pack(0, [nr, lanes], [cN, cN]);
        return matmul.ToValue(new VectorType(DataTypes.Float32, [nr, lanes]));
    }

    public IRType Visit(ITypeInferenceContext context, PackedMatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, PackedMatMul.Rhs);

        IRType rType;
        string? errorMessage = null;
        switch (lhs, rhs)
        {
            case (DistributedType a, DistributedType b):
                {
                    var bVectorType = (VectorType)b.TensorType.DType;
                    var nr = bVectorType.Lanes[0];
                    var unpackedB = b with { TensorType = UnpackedBType(b.TensorType) };
                    var dimInfo = VectorizedMatMul.GetDimInfo(false, true, a.TensorType.Shape.Rank, unpackedB.TensorType.Shape.Rank);
                    rType = Math.MatMulEvaluator.VisitDistributedType(a, unpackedB, dimInfo: dimInfo, transB: true, outputDataType: target.OutputDataType);
                    if (rType is not DistributedType drType)
                    {
                        return rType;
                    }

                    if (target.FusedReduce)
                    {
                        drType = (DistributedType)Math.MatMulEvaluator.ConvertPartialToBroadcast(drType);
                    }

                    rType = drType with { TensorType = (TensorType)TypeInference.PackType(drType.TensorType, [nr], [b.TensorType.Shape.Rank - 1]) };
                }

                break;
            case (TensorType a, TensorType b):
                {
                    var bVectorType = (VectorType)b.DType;
                    var nr = bVectorType.Lanes[0];
                    var unpackedB = UnpackedBType(b);
                    var dimInfo = VectorizedMatMul.GetDimInfo(false, true, a.Shape.Rank, unpackedB.Shape.Rank);
                    rType = Math.MatMulEvaluator.VisitTensorType(a, unpackedB, dimInfo: dimInfo, outputDataType: target.OutputDataType);
                    rType = TypeInference.PackType((TensorType)rType, [nr], [b.Shape.Rank - 1]);
                }

                break;
            default:
                rType = new InvalidType($"lhs: {lhs}, rhs: {rhs}, in {target.DisplayProperty()} not support: {errorMessage}");
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
            var k = lhsShape.Rank - 1;
            macPerElement = lhsShape[k].IsFixed ? (uint)lhsShape[k].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            var k = distributedType.TensorType.Shape.Rank - 1;
            macPerElement = lhsType.Shape[k].IsFixed ? (uint)lhsType.Shape[k].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }

    private TensorType UnpackedBType(TensorType tensorType)
    {
        var vectorType = (VectorType)tensorType.DType;
        var nr = vectorType.Lanes[0];
        var lanes = vectorType.Lanes[1];
        var newShape = tensorType.Shape.ToArray();
        newShape[^2] *= nr;
        return tensorType with { DType = vectorType with { Lanes = [lanes] }, Shape = newShape };
    }
}
