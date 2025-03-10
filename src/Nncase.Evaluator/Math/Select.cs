﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Select"/>.
/// </summary>
[PatternMatch.PatternFunctionalGenerator]
[EvaluatorGenerator]
public partial class SelectEvaluator : IEvaluator<Select>, ITypeInferencer<Select>, IOpPrinter<Select>
{
    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Select target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Script) | context.Flags.HasFlag(PrinterFlags.Inline))
        {
            var condition = context.GetArgument(target, Select.Predicate);
            var true_value = context.GetArgument(target, Select.TrueValue);
            var false_value = context.GetArgument(target, Select.FalseValue);

            return $"({condition} ? {true_value} : {false_value})";
        }

        return context.GetDefault(target);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Select target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, Select.TrueValue);
        var rhs = context.CheckArgumentType<IRType>(target, Select.FalseValue);
        return TypeInference.CommonType(lhs, rhs);
    }

    private IValue Visit(bool predicate, IValue trueValue, IValue falseValue)
    {
        return predicate ? trueValue : falseValue;
    }
}
