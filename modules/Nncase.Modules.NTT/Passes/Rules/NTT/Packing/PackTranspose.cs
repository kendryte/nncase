// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class VectorizeTransposePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsTranspose(
                "trans",
                "callee",
                IsWildcard("input"),
                IsFixedShape("perm")));

    private Expr? GetReplace(IR.Tensors.Vectorize vectorize, Call caller, Call callee, Expr input, int[] perm)
    {
        var vectorizeAxes = vectorize.Axes.Select(a => perm[a]).ToArray();
        return callee.WithArguments([
            (Transpose.Input, IR.F.Tensors.Vectorize(input, vectorize.Lanes.ToArray(), vectorizeAxes)),
        ]);
    }
}

[RuleGenerator]
public sealed partial class TransposeDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsTranspose(
            "trans",
            "caller",
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsFixedShape("perm"));

    private Expr? GetReplace(IR.Tensors.Devectorize devectorize, Call caller, Call callee, Expr input, int[] perm)
    {
        var devectorizeAxes = devectorize.Axes.Select(a => perm.IndexOf(a)).ToArray();
        return IR.F.Tensors.Devectorize(
            caller.WithArguments([
                (Transpose.Input, input),
            ]),
            devectorize.Lanes.ToArray(),
            devectorizeAxes);
    }
}
