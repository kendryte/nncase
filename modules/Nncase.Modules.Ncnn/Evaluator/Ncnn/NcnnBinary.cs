// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnBinary"/>.
/// </summary>
public class NcnnBinaryEvaluator : IEvaluator<NcnnBinary>, ITypeInferencer<NcnnBinary>, ICostEvaluator<NcnnBinary>, IShapeEvaluator<NcnnBinary>, IMetricEvaluator<NcnnBinary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnBinary binary)
    {
        var inputA = context.GetOrtArgumentValue(binary, NcnnBinary.InputA);
        var inputB = context.GetOrtArgumentValue(binary, NcnnBinary.InputB);
        var opType = binary.OpType;

        // return OrtKI. (input, dim).ToValue();
        switch (opType)
        {
            case BinaryOperationType.ADD:
                return OrtKI.Add(inputA, inputB).ToValue();
            case BinaryOperationType.SUB:
                return OrtKI.Sub(inputA, inputB).ToValue();
            case BinaryOperationType.MUL:
                return OrtKI.Mul(inputA, inputB).ToValue();
            case BinaryOperationType.DIV:
                return OrtKI.Div(inputA, inputB).ToValue();

            // case BinaryOperationType.MAX:
            //     return System.Math.Min(inputA, inputB).ToValue();
            // case BinaryOperationType.MIN:
            //     return OrtKI.Min(inputA, inputB).ToValue();

            // TODO: trunc
            default:
                throw new NotSupportedException("Ncnn unary ops");
        }
    }

    public IRType Visit(NcnnBinary target, TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnBinary target)
    {
        return target.LorR switch
        {
            1 => Visit(context.CheckArgumentType<TensorType>(target, NcnnBinary.InputA), new TensorType(context.CheckArgumentType<TensorType>(target, NcnnBinary.InputA).DType, target.ConstShape)),
            2 => Visit(new TensorType(context.CheckArgumentType<TensorType>(target, NcnnBinary.InputA).DType, target.ConstShape), context.CheckArgumentType<TensorType>(target, NcnnBinary.InputA)),
            0 => Visit(context.CheckArgumentType<TensorType>(target, NcnnBinary.InputA), context.CheckArgumentType<TensorType>(target, NcnnBinary.InputB)),
            _ => throw new NotSupportedException("Never reach here, LorR without fourth situation."),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnBinary target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnBinary target)
    {
        var lhsType = context.GetArgumentType<TensorType>(target, NcnnBinary.InputA);
        var rhsType = context.GetArgumentType<TensorType>(target, NcnnBinary.InputB);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(lhsType) + CostUtility.GetMemoryAccess(rhsType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, (int)MetricUtility.GetBinaryFLOPs(MapBinaryOp(target.OpType))),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnBinary target)
    {
        var lhs = context.GetArgumentShape(target, NcnnBinary.InputA);
        var rhs = context.GetArgumentShape(target, NcnnBinary.InputB);
        return ShapeExprUtility.BroadcastShape(lhs, rhs);
    }

    private static BinaryOp MapBinaryOp(BinaryOperationType binaryOp) =>
        binaryOp switch
        {
            BinaryOperationType.ADD => BinaryOp.Add,
            BinaryOperationType.SUB => BinaryOp.Sub,
            BinaryOperationType.MUL => BinaryOp.Mul,
            BinaryOperationType.DIV => BinaryOp.Div,
            BinaryOperationType.MAX => BinaryOp.Max,
            BinaryOperationType.MIN => BinaryOp.Min,
            BinaryOperationType.POW => BinaryOp.Pow,

            // _ => null,

            // unsupported Binary ops
            // BinaryOp.Mod =>
            // BitwiseAnd
            // BitwiseOr
            // BitwiseXor
            // LogicalAnd
            // LogicalOr
            // LogicalXor
            // LeftShift
            // RightShift
            // => BinaryOperationType.RSUB,
            // => BinaryOperationType.RDIV,
            // => BinaryOperationType.RPOW,
            // => BinaryOperationType.ATAN2,
            // => BinaryOperationType.RATAN2,
        };

    private IRType Visit(TensorType inputA, TensorType inputB)
    {
        return TypeInference.BroadcastType(inputA, inputB);
    }
}
