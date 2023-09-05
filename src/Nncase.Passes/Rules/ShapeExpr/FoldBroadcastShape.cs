using System.Linq;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.ShapeExpr;
using static Nncase.PatternMatch.Utility;
namespace Nncase.Passes.Rules.ShapeExpr;

[RuleGenerator]
public partial class FoldBroadcastShape : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCall(IsOp<BroadcastShape>(), IsTuple("input"));

    Expr? GetReplace(IR.Tuple input)
    {
        var broadcastShapeList = input.Fields.ToArray().Where(field => field is Call c && c.Target is BroadcastShape).ToArray();
        if (broadcastShapeList.Length > 0)
        {
            var newFields = input.Fields.ToArray().SelectMany(field =>
            {
                if (field is Call { Target: BroadcastShape } c)
                {
                    return ((Tuple)c.Arguments[0]).Fields.ToArray();
                }

                return new[] { field };
            }).ToArray();
            return IR.F.ShapeExpr.BroadcastShape(newFields);
        }

        return null;
    }
}
