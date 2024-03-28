// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerSqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSqueeze(
        IsWildcard("input"),
        IsTensorConst("dims"));

    private Expr? GetReplace(Expr input, int[] dims)
    {
        if (input.CheckedShape.Count < 5)
        {
            var inResO = new Var(input.CheckedType);

            var newDims = new List<int>();
            foreach (int item in dims)
            {
                if (item > 4 || item < -3)
                {
                    return null;
                }
                else
                {
                    newDims.Add(item);
                }

                if (input.CheckedShape[item] != 1)
                {
                    return null;
                }
            }

            return new Call(new Fusion("ncnn", NcnnSqueeze(inResO, newDims.ToArray()), new[] { inResO }), input);
        }

        return null;
    }
}
