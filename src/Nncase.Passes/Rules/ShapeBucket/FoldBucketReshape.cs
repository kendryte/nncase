using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.Utility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.IR.TypePatternUtility;
namespace Nncase.Passes.Rules.ShapeBucket;

[RuleGenerator]
public sealed partial class FoldBucketPadReshape : RewriteRule<Pattern>
{
    // Reshape(Gather(Shape, 0, 0), new[] { 0 }) -> GetItem(Shape, 0)
    public override Pattern Pattern => IsReshape(
        IsBucketPad(null, "bucketPad", IsWildcard(), IsTensorConst()),
        IsTensorConst("newShape"));

    Expr? GetReplace(Call bucketPad, Expr newShape)
    {
        return ReplaceUtility.ReplaceCallParams(bucketPad, (BucketPad.Shape.Index, newShape));
    }
}

// todo: squeeze
[RuleGenerator]
public sealed partial class FoldBucketPadUnsqueeze : RewriteRule<Pattern>
{
    // Reshape(Gather(Shape, 0, 0), new[] { 0 }) -> GetItem(Shape, 0)
    public override Pattern Pattern => IsUnsqueeze(null, "unsqueeze",
        IsBucketPad(null, "bucketPad", IsWildcard(), IsTensorConst()),
        IsTensorConst());

    Expr? GetReplace(Call bucketPad, Call unsqueeze)
    {
        return ReplaceUtility.ReplaceCallParams(bucketPad, (BucketPad.Shape.Index, unsqueeze.CheckedShape.ToValueArray()));
    }
}
