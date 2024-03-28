// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnUnary"/>.
/// </summary>
public class NcnnUnaryEvaluator : IEvaluator<NcnnUnary>, ITypeInferencer<NcnnUnary>, ICostEvaluator<NcnnUnary>, IShapeEvaluator<NcnnUnary>, IMetricEvaluator<NcnnUnary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnUnary unary)
    {
        var input = context.GetOrtArgumentValue(unary, NcnnUnary.Input);
        var opType = unary.OpType;

        // return OrtKI. (input, dim).ToValue();
        switch (opType)
        {
            case UnaryOperationType.ABS:
                return OrtKI.Abs(input).ToValue();
            case UnaryOperationType.NEG:
                return OrtKI.Neg(input).ToValue();
            case UnaryOperationType.FLOOR:
                return OrtKI.Floor(input).ToValue();
            case UnaryOperationType.CEIL:
                return OrtKI.Ceil(input).ToValue();
            case UnaryOperationType.SQUARE:
                return OrtKI.Square(input).ToValue();
            case UnaryOperationType.SQRT:
                return OrtKI.Sqrt(input).ToValue();
            case UnaryOperationType.RSQRT:
                return OrtKI.Rsqrt(input).ToValue();
            case UnaryOperationType.EXP:
                return OrtKI.Exp(input).ToValue();
            case UnaryOperationType.LOG:
                return OrtKI.Log(input).ToValue();
            case UnaryOperationType.SIN:
                return OrtKI.Sin(input).ToValue();
            case UnaryOperationType.COS:
                return OrtKI.Cos(input).ToValue();
            case UnaryOperationType.TAN:
                return OrtKI.Tan(input).ToValue();
            case UnaryOperationType.ASIN:
                return OrtKI.Asin(input).ToValue();
            case UnaryOperationType.ACOS:
                return OrtKI.Acos(input).ToValue();
            case UnaryOperationType.ATAN:
                return OrtKI.Atan(input).ToValue();
            case UnaryOperationType.RECIPROCAL:
                return OrtKI.Reciprocal(input).ToValue();
            case UnaryOperationType.TANH:
                return OrtKI.Tanh(input).ToValue();
            case UnaryOperationType.LOG10:
                double ln10 = 2.3025850929940456840179914546844;
                return OrtKI.Div(OrtKI.Log(input), OrtKISharp.Tensor.FromScalar(ln10)).ToValue();
            case UnaryOperationType.ROUND:
                return OrtKI.Round(input).ToValue();

            // TODO: trunc
            default:
                throw new NotSupportedException("Ncnn unary ops");
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnUnary target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnUnary.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnUnary target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnUnary target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnUnary.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (inner * 2) + (inputF * (MetricUtility.SubFLOPs + MetricUtility.ExpFLOPs + MetricUtility.DivFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnUnary target) => context.GetArgumentShape(target, NcnnUnary.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
