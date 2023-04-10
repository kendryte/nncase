// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// quantize(concat(a,b,c)) => concat(quantize(a),quantize(b),quantize(c)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineQuantizeConcat : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsQuantize(
      "quantize",
      _ => true,
      IsConcat(
        IsTuple(IsVArgsRepeat("tupleInputs", () => IsWildcard())),
        IsWildcard("axis")),
      IsWildcard("quantParam"));

    private Expr? GetReplace(Quantize quantize, IReadOnlyList<Expr> tupleInputs, Expr axis, Expr quantParam, RunPassContext options)
    {
        var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();

        // see UnitTestCombineQuantize.TestCombineQuantizeConcatNegative
        foreach (var e in tupleInputs)
        {
            if (userAnalysis[e].Count() > 1)
            {
                return null;
            }
        }

        return Concat(new IR.Tuple(tupleInputs.Select(e => IR.F.Math.Quantize(e, quantParam, quantize.TargetType)).ToArray()), axis);
    }
}

/// <summary>
/// quantize(reshape(a)) => reshape(quantize(a)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineQuantizeReshape : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsQuantize(
        "quantize",
        _ => true,
        IsReshape(
            "reshape",
            "reshapeCall",
            IsWildcard("input"),
            IsWildcard("shape")),
        IsWildcard("quantParam"));

    private Expr? GetReplace(Quantize quantize, Call reshapeCall, Expr input, Expr shape, Expr quantParam, RunPassContext options)
    {
        var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();

        if (userAnalysis[reshapeCall].Count() > 1)
        {
            return null;
        }

        var output = Reshape(Quantize(input, quantParam, quantize.TargetType), shape);
        output.InferenceType();
        return output;
    }
}

/// <summary>
/// quantize(transpose(a)) => transpose(quantize(a)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineQuantizeTranspose : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsQuantize(
        "quantize",
        _ => true,
        IsTranspose(
            "transpose",
            "transposeCall",
            IsWildcard("input"),
            IsWildcard("perm")),
        IsWildcard("quantParam"));

    private Expr? GetReplace(Quantize quantize, Call transposeCall, Expr input, Expr perm, Expr quantParam, RunPassContext options)
    {
        var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();

        if (userAnalysis[transposeCall].Count() > 1)
        {
            return null;
        }

        var output = Transpose(Quantize(input, quantParam, quantize.TargetType), perm);
        output.InferenceType();
        return output;
    }
}
