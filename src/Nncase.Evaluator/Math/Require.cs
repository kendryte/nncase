// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
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
public partial class RequireEvaluator : IEvaluator<Require>, ITypeInferencer<Require>, IOpPrinter<Require>, ICostEvaluator<Require>, IShapeEvaluator<Require>
{
    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Require target, bool iLmode)
    {
        var condition = context.GetArgument(target, Require.Predicate);
        var value = context.GetArgument(target, Require.Value);
        return $"IR.F.Math.Require({condition}, {value})";
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

    public Cost? Visit(ICostEvaluateContext context, Require target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Require target) =>
        context.GetArgumentShape(target, Require.Value);
}
