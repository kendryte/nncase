// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerPermute : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsTranspose(
        IsWildcard("input"),
        IsTensorConst("perm"));

    private Expr? GetReplace(Expr input, int[] perm)
    {
        if (input.CheckedShape[0] != 1 || input.CheckedShape.Rank > 5 || perm.Length == 2)
        {
            return null;
        }

        int orderType = 0;
        if (perm.Length == 3)
        {
            if (perm.Skip(1).SequenceEqual(new[] { 1, 2 }))
            {
                orderType = 0;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 2, 1 }))
            {
                orderType = 1;
            }
            else if (perm.SequenceEqual(new[] { 1, 0, 2 }))
            {
                orderType = 0;
            }
            else if (perm.SequenceEqual(new[] { 2, 0, 1 }))
            {
                orderType = 1;
            }
            else
            {
                return null;
            }
        }
        else if (perm.Length == 4)
        {
            if (perm.Skip(1).SequenceEqual(new[] { 1, 2, 3 }))
            {
                orderType = 0;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 1, 3, 2 }))
            {
                orderType = 1;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 2, 1, 3 }))
            {
                orderType = 2;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 2, 3, 1 }))
            {
                orderType = 3;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 3, 1, 2 }))
            {
                orderType = 4;
            }
            else if (perm.Skip(1).SequenceEqual(new[] { 3, 2, 1 }))
            {
                orderType = 5;
            }
            else
            {
                return null;
            }
        }

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);

        var permute = new Call(new Fusion("ncnn", NcnnPermute(inResO, orderType, perm), inResO), inRes);
        return Unsqueeze(permute, new[] { 0 });
    }
}
