// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using QLinearMatMul = Nncase.IR.Math.QLinearMatMul;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="QLinearMatMul"/>.
/// </summary>
public class QLinearMatMulEvaluator : IEvaluator<QLinearMatMul>, ITypeInferencer<QLinearMatMul>, ICostEvaluator<QLinearMatMul>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, QLinearMatMul qLinearMatMul)
    {
        var lhs = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.Lhs).Cast(OrtDataType.Float);
        var rhs = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.Rhs).Cast(OrtDataType.Float);
        var lhsScale = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.LhsScale).Cast(OrtDataType.Float);
        var lhsZeroPoint = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.LhsZeroPoint).Cast(OrtDataType.Float);
        var rhsScale = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.RhsScale).Cast(OrtDataType.Float);
        var rhsZeroPoint = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.RhsZeroPoint).Cast(OrtDataType.Float);
        var outputScale = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.OutputScale).Cast(OrtDataType.Float);
        var outputZeroPoint = context.GetOrtArgumentValue(qLinearMatMul, QLinearMatMul.OutputZeroPoint).Cast(OrtDataType.Float);
        var outputDataType = qLinearMatMul.OutputDataType;

        var deqLhs = (lhs - lhsZeroPoint) * lhsScale;
        var deqRhs = (rhs - rhsZeroPoint) * rhsScale;
        var result = (OrtKI.MatMul(deqLhs, deqRhs) * outputScale) + outputZeroPoint;
        result = result.Cast(outputDataType.ToOrtType());

        return Value.FromTensor(result.ToTensor());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, QLinearMatMul target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, QLinearMatMul.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, QLinearMatMul.Rhs);
        var outputDataType = target.OutputDataType;
        return MatMulEvaluator.VisitTensorType(lhs, rhs, false, null, outputDataType);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, QLinearMatMul target)
    {
        // FIXME: add cost of dequant & quant
        var lhs = context.GetArgumentType<IRType>(target, QLinearMatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, QLinearMatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: RankedShape lhsShape })
        {
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            lhsShape = (RankedShape)lhsType.Shape;
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }
}
