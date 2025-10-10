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
using Nncase.IR.NTT;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class CombineVectorizedLayerNormTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        "transpose",
        _ => true,
        IsVectorizedCast(
            "cast",
            "callee",
            _ => true,
            IsWildcard("input"),
            IsWildcard("postOps")),
        IsFixedShape("perm"));

    private Expr GetReplace(VectorizedCast cast, Call callee, Expr input, Expr postOps, int[] perm)
    {
        var newAxes = cast.VectorizeAxes.Select(a => perm.IndexOf(a)).Order().ToArray();
        var newLanes = newAxes.Select(a => ((VectorType)cast.NewType).Lanes[cast.VectorizeAxes.IndexOf(perm[a])]).ToArray();
        var newType = new VectorType(((VectorType)cast.NewType).ElemType, newLanes);

        return IR.F.NTT.VectorizedCast(IR.F.Tensors.Transpose(input, perm), newType, cast.CastMode, newAxes, postOps).InheritMetaData(callee);
    }
}
