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
public sealed partial class PackCastPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsCast(
                "cast",
                "callee",
                _ => true,
                IsWildcard("input")));

    private Expr? GetReplace(Pack pack, Cast cast, Call caller, Call callee, Expr input)
    {
        var scale = 1f * ((VectorType)caller.CheckedDataType).ElemType.SizeInBytes / input.CheckedDataType.SizeInBytes;
        if (pack.Axes.Any(a => callee.CheckedShape[a] is { IsFixed: true, FixedValue: var d } && d / scale % 1 != 0))
        {
            return null;
        }

        var packLanes = pack.Lanes.Select(l => (int)(l * scale)).ToArray();
        var newType = new VectorType(cast.NewType, pack.Lanes);

        var ret = IR.F.Tensors.Cast(IR.F.Tensors.Pack(input, packLanes, pack.Axes.ToArray()), newType, CastMode.KDefault, pack.Axes.ToArray());
        return ret;
    }
}
