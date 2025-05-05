// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.ShapeExpr;

// shape = ShapeOf(input)
// Stack(cast(shape[0]), cast(shape[1])) -> shape
[RuleGenerator]
public partial class FoldSplitShapeOf : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsStack(
        null,
        "stack",
        IsTuple(
            "tuple",
            new VArgsPattern(
                list =>
                    Enumerable.Range(0, list.Length)
                    .Select(_ => IsAlt(IsCast(c => c.NewType == DataTypes.Int64, InputPattern), InputPattern))
                    .ToArray(),
                "args")),
        IsFixedDimension(value: 0));

    public Pattern InputPattern => IsGetItem(IsShapeOf(IsWildcard()), IsFixedDimension() | IsFixedShape());

    private BaseExpr? GetReplace(IR.Tuple tuple)
    {
        var getItemList = tuple.Fields.ToArray().OfType<Call>().Select(x =>
        {
            if (x.Target is Cast)
            {
                return x[Cast.Input];
            }

            return x;
        }).OfType<Call>().ToArray();

        var getItemIndices = new List<int>();
        foreach (var getItem in getItemList)
        {
            var index = getItem[GetItem.Index];
            if (index is DimConst dc)
            {
                getItemIndices.Add((int)dc.Value);
            }
            else if (index is RankedShape { IsFixed: true, Rank: 1 } rs)
            {
                getItemIndices.Add((int)rs[0].FixedValue);
            }
        }

        if (getItemIndices.Count == 0)
        {
            return null;
        }

        var shapeOf = getItemList[0][GetItem.Input];
        if (!shapeOf.CheckedShape[0].IsFixed)
        {
            return null;
        }

        if (getItemIndices.SequenceEqual(Enumerable.Range(0, (int)shapeOf.CheckedShape[0].FixedValue)))
        {
            return shapeOf;
        }

        return null;
    }
}
