// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using DryIoc.ImTools;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Combine Tuple(Reshape(Expand(Reshape(GetItem(Slice)))), ...) of GLM Model.
/// </summary>
[RuleGenerator]
public sealed partial class CombineGLMTuple : IRewriteRule
{
    private static readonly Pattern Input = IsSplit(
      "split",
      IsWildcard("input") with { TypePattern = HasRankedShape() },
      IsTensorConst("axis"),
      IsTensorConst("sections"));

    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConcat(
            "concat",
            "concatCall",
            _ => true,
            IsTuple(null, IsVArgsRepeat("tupleInputs", exprs =>
            {
                var patterns = new Pattern[exprs.Length];
                for (var i = 0; i < patterns.Length; i++)
                {
                    patterns[i] = IsReshape(
                        IsExpand(
                            IsReshape(
                                IsGetItem(Input, IsTensorConst()),
                                IsTensorConst()),
                            IsTensorConst()),
                        IsTensorConst());
                }

                return patterns;
            })));

    private Expr? GetReplace(Expr input, Call concatCall, IReadOnlyList<BaseExpr> tupleInputs, IMatchResult matchResult)
    {
        var inShape = (RankedShape)input.CheckedShape;
        var expandShape = new RankedShape([.. inShape, 2]);
        return Reshape(Expand(Unsqueeze(input, new[] { -1 }), expandShape), expandShape.SkipLast(2).Concat([expandShape[^1] * expandShape[^2]]).ToArray());
    }
}
