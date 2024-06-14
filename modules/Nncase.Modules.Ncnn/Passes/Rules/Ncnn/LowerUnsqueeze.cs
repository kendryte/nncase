// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerUnsqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsUnsqueeze(
        IsWildcard("input") with { TypePattern = HasFixedShape() & !IsScalar() },
        IsTensorConst("dims"));

    private Expr? GetReplace(Expr input, int[] dims)
    {
        if (input.CheckedShape.Count + dims.Length < 5)
        {
            var inResO = new Var(input.CheckedType);
            return new Call(new Fusion("ncnn", NcnnUnsqueeze(inResO, dims), new[] { inResO }), input);
        }

        return null;
    }
}
