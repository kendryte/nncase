// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.ShapeExpr;

// todo: add test
// shape = ShapeOf(input)
// Stack(cast(shape[0]), cast(shape[1])) -> shape
[RuleGenerator]
public partial class FoldSplitShapeOf : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsStack(
        null,
        "stack",
        IsTuple("tuple", IsVArgsRepeat(list =>
            Enumerable.Range(0, list.Length)
                .Select(_ => (Pattern)IsCast(null, _ => true, IsGetItem(InputPattern, IsTensorConst())))
                .ToArray())),
        IsTensorConst(tensor => tensor.Value.ToScalar<int>() == 0));

    public Pattern InputPattern => IsCast(null, _ => true, IsShapeOf((string)null, IsWildcard()));

    private Expr? GetReplace(IR.Tuple tuple)
    {
        var getItemList = tuple.Fields.ToArray().OfType<Call>().Select(c => c.Arguments[Cast.Input.Index]).OfType<Call>().ToArray();
        var getItemIndices = getItemList.Select(x => x.Arguments[GetItem.Index.Index]).OfType<TensorConst>().Select(x => x.Value.ToScalar<int>()).ToArray();
        if (getItemIndices.Length == 0)
        {
            return null;
        }

        var shapeOf = ((Call)getItemList[0].Arguments[GetItem.Input.Index]).Arguments[Cast.Input.Index];
        if (!shapeOf.CheckedShape[0].IsFixed)
        {
            return null;
        }

        if (getItemIndices.SequenceEqual(Enumerable.Range(0, shapeOf.CheckedShape[0].FixedValue)))
        {
            return shapeOf;
        }

        return null;
    }
}
