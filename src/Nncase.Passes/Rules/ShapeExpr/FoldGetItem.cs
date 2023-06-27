// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.ShapeExpr;

/// <summary>
/// (Stack(GetItem(input, i)), i) => GetItem(input, i).
/// </summary>
[RuleGenerator]
public partial class FoldStackGetItem : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsStack(
        "stack",
        IsTuple("tuple", IsVArgsRepeat(list =>
            Enumerable.Range(0, list.Length)
                    .Select(_ => (Pattern)IsGetItem(InputPattern, IsTensorConst()))
                    .ToArray())),
        IsTensorConst(tensor => tensor.Value.ToScalar<int>() == 0));

    private Pattern InputPattern => IsWildcard();

    private Expr? GetReplace(IR.Tuple tuple)
    {
        var getItems = tuple.Fields.ToArray().Select(x => (Call)x).ToArray();
        var index = getItems.Select(x => ((TensorConst)x.Arguments[GetItem.Index.Index]).Value.ToScalar<int>());
        if (!Enumerable.Range(0, getItems.Length).SequenceEqual(index))
        {
            return null;
        }

        var input = getItems[0].Arguments[GetItem.Input.Index];
        if (input.CheckedShape.Rank != getItems.Length)
        {
            return null;
        }

        return input;
    }
}
