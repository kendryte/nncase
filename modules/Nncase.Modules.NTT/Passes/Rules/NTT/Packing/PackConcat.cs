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
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class PackConcatPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsConcat(
                "concat",
                "callee",
                _ => true,
                IsTuple(null, IsVArgsRepeat("tupleInputs", () => IsWildcard(null, e => e is Expr) with { TypePattern = HasRankedShape() }))));

    private Expr? GetReplace(Pack pack, Concat concat, Call caller, Call callee, IReadOnlyList<BaseExpr> tupleInputs)
    {
        var packAxisIndex = pack.Axes.IndexOf(concat.Axis);
        if (packAxisIndex != -1
            && tupleInputs.Any(input => !Dimension.TryDivExactly(input.CheckedShape[concat.Axis], pack.Lanes[packAxisIndex], out _)))
        {
            return null; // If any input is not packable, we cannot replace the concat with a pack.
        }

        var newInputs = tupleInputs.Select(input => caller.WithArguments([(Pack.Input, input)])).ToArray();
        return callee.WithArguments([(Concat.Input, new IR.Tuple(newInputs))]);
    }
}
