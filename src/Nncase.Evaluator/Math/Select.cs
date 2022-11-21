// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Select"/>.
/// </summary>

[PatternMatch.PatternFunctionalGenerator, TypeInferGenerator, EvaluatorGenerator]
public partial class SelectEvaluator : IEvaluator<Select>, ITypeInferencer<Select>, IOpPrinter<Select>
{
    /// <inheritdoc />
    IValue Visit(bool Predicate, IValue TrueValue, IValue FalseValue)
    {
        return Predicate ? TrueValue : FalseValue;
    }

    /// <inheritdoc/>
    IRType Visit(TensorType Predicate, IRType TrueValue, IRType FalseValue)
    {
        if (TrueValue is TensorType true_type && FalseValue is TensorType false_type)
        {
            if (true_type.DType != false_type.DType || true_type.Shape != false_type.Shape)
                return new InvalidType($"TrueValue.DType {true_type.DType.GetDisplayName()} != FalseValue.DType {false_type.DType.GetDisplayName()}");
        }
        else if (TrueValue is AnyType || FalseValue is AnyType)
        {
            return AnyType.Default;
        }
        return TrueValue;
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Select target, bool ILmode)
    {
        var condition = context.GetArgument(target, Select.Predicate);
        var true_value = context.GetArgument(target, Select.TrueValue);
        var false_value = context.GetArgument(target, Select.FalseValue);

        return $"({condition} ? {true_value} : {false_value})";
    }
}
