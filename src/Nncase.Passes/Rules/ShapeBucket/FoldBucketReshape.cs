// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.ShapeBucket;

[RuleGenerator]
public sealed partial class FoldBucketPadReshape : RewriteRule<Pattern>
{
    // Reshape(Gather(Shape, 0, 0), new[] { 0 }) -> GetItem(Shape, 0)
    public override Pattern Pattern => IsReshape(
        IsBucketPad(null, "bucketPad", IsWildcard(), IsTensorConst()),
        IsTensorConst("newShape"));

    private Expr? GetReplace(Call bucketPad, Expr newShape)
    {
        return ReplaceUtility.ReplaceCallParams(bucketPad, (BucketPad.Shape.Index, newShape));
    }
}

// todo: squeeze
[RuleGenerator]
public sealed partial class FoldBucketPadUnsqueeze : RewriteRule<Pattern>
{
    // Reshape(Gather(Shape, 0, 0), new[] { 0 }) -> GetItem(Shape, 0)
    public override Pattern Pattern => IsUnsqueeze(
        null,
        "unsqueeze",
        IsBucketPad(null, "bucketPad", IsWildcard(), IsTensorConst()),
        IsTensorConst());

    private Expr? GetReplace(Call bucketPad, Call unsqueeze)
    {
        return ReplaceUtility.ReplaceCallParams(bucketPad, (BucketPad.Shape.Index, unsqueeze.CheckedShape.ToValueArray()));
    }
}
