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
        "concat",
        _ => true,
        IsTuple("tuple", IsVArgsRepeat("tupleInputs", () => IsWildcard()))),
      IsWildcard("quantParam"));

    private Expr? GetReplace(Quantize quantize, IReadOnlyList<Expr> tupleInputs, IR.Tensors.Concat concat, Expr quantParam, RunPassContext options, Expr tuple)
    {
        if (options.Driver is DataflowPass)
        {
            var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();

            // see UnitTestCombineQuantize.TestCombineQuantizeConcatNegative
            foreach (var e in tupleInputs)
            {
                if (userAnalysis[e].Count() > 1)
                {
                    foreach (var user in userAnalysis[e])
                    {
                        if (user is Call { Target: Nncase.IR.Math.Quantize } userCall)
                        {
                            var quantUser = userCall.Arguments[Nncase.IR.Math.Quantize.QuantParam.Index];
                            if (quantUser != quantParam)
                            {
                                return null;
                            }
                        }
                        else
                        {
                            if (user is not Nncase.IR.Tuple)
                            {
                                return null;
                            }
                            else
                            {
                                if (user != tuple)
                                {
                                    return null;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Concat(new IR.Tuple(tupleInputs.Select(e => IR.F.Math.Quantize(e, quantParam, quantize.TargetType)).ToArray()), concat.Axis);
    }
}

/// <summary>
/// quantize(reshape(a)) => reshape(quantize(a)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineQuantizeReshape : RewriteRule<Pattern>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CombineQuantizeReshape"/> class.
    /// </summary>
    /// <param name="checkShapeSize">if true, skip pass.</param>
    public CombineQuantizeReshape(bool checkShapeSize)
    {
        Pattern = IsQuantize(
            "quantize",
            _ => true,
            IsReshape(
                "reshape",
                "reshapeCall",
                IsWildcard("input") with { TypePattern = HasShape(sp => !(checkShapeSize && sp.ToValueArray().Any(s => s >= 65536)), "CheckedShape") },
                IsWildcard("shape")),
            IsWildcard("quantParam"));
    }

    public CombineQuantizeReshape()
        : this(false)
    {
    }

    /// <inheritdoc/>
    public override Pattern Pattern { get; }

    private Expr? GetReplace(Quantize quantize, Call reshapeCall, Expr input, Expr shape, Expr quantParam, RunPassContext context)
    {
        if (context.Driver is DataflowPass)
        {
            var userAnalysis = context.GetAnalysis<IExprUserAnalysisResult>();
            if (userAnalysis[reshapeCall].Count() > 1)
            {
                return null;
            }
        }

        var output = Reshape(Quantize(input, quantParam, quantize.TargetType), shape);
        return output;
    }
}

/// <summary>
/// reshape(quantize(a)) => quantize(reshape(a)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineReshapeQuantize : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsReshape(
        "reshape",
        _ => true,
        IsQuantize(
            "quantize",
            "quantizeCall",
            _ => true,
            IsWildcard("input"),
            IsWildcard("quantParam")),
        IsWildcard("shape"));

    private Expr? GetReplace(Quantize quantize, Call quantizeCall, Expr input, Expr shape, Expr quantParam, RunPassContext options)
    {
        var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();

        if (userAnalysis[quantizeCall].Count() > 1)
        {
            return null;
        }

        var output = Quantize(Reshape(input, shape), quantParam, quantize.TargetType);
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
        try
        {
            var userAnalysis = options.GetAnalysis<IExprUserAnalysisResult>();
            if (userAnalysis[transposeCall].Count() > 1)
            {
                return null;
            }
        }
        catch (System.Exception)
        {
        }

        return Transpose(Quantize(input, quantParam, quantize.TargetType), perm);
    }
}
