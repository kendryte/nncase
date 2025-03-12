// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Require"/>.
/// </summary>
[PatternMatch.PatternFunctionalGenerator]
[TypeInferGenerator]
[EvaluatorGenerator]
public partial class RequireEvaluator : IEvaluator<Require>, ITypeInferencer<Require>, IOpPrinter<Require>, ICostEvaluator<Require>, IMetricEvaluator<Require>
{
    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Require target)
    {
        var condition = context.GetArgument(target, Require.Predicate);
        var value = context.GetArgument(target, Require.Value);
        return $"Require({condition}, {value})";
    }

    public Cost Visit(ICostEvaluateContext context, Require target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Require target) => Metric.Zero;

    public IValue Visit(IEvaluateContext context, Require target)
    {
        return context.GetArgumentValue(target, Require.Value);
    }

    private IValue Visit(bool predicate, IValue value, Require target)
    {
        if (!predicate)
        {
            throw new InvalidOperationException($"The Require {target.Message} Is False!");
        }

        return value;
    }

    private IRType Visit(TensorType predicate, IRType value)
    {
        return value;
    }
}
