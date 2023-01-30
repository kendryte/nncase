// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Condition"/>.
/// </summary>
[PatternMatch.PatternFunctionalGenerator]
[TypeInferGenerator]
public partial class ConditionEvaluator : IEvaluator<Condition>, ITypeInferencer<Condition>, IOpPrinter<Condition>
{
    /// <inheritdoc/>
    // IValue Visit(bool Predicate, IValue Value)
    public IValue Visit(IEvaluateContext context, Condition cond)
    {
        var predicate = context.GetArgumentValueAsTensor(cond, Condition.Predicate);
        var value = context.GetArgumentValue(cond, Condition.Value);
        var b = predicate.ToScalar<bool>();
        if (!b)
        {
            throw new ArgumentOutOfRangeException($"predicate = {b}");
        }

        return value;
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Condition target, bool iLmode)
    {
        var condition = context.GetArgument(target, Condition.Predicate);
        var true_value = context.GetArgument(target, Condition.Value);
        return $"Condition({condition}, {true_value})";
    }

    private IRType Visit(TensorType predicate, TensorType value)
    {
        if (!predicate.IsScalar && predicate.DType != DataTypes.Boolean)
        {
            return new InvalidType($"Predicate {predicate} is not bool");
        }

        return value;
    }
}
