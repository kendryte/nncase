using System.Linq;
using Google.OrTools.Sat;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.ShapeExpr;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.ShapeExpr;
using static Nncase.PatternMatch.Utility;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
namespace Nncase.Passes.Rules.ShapeExpr;

[RuleGenerator]
public partial class FoldBroadcastShapeConst : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCall(IsOp<BroadcastShape>(), IsTuple("input"));

    Expr? GetReplace(IR.Tuple input)
    {
        var constFields = input.Fields.ToArray().OfType<TensorConst>().ToArray();
        if (constFields.Length == 0)
        {
            return null;
        }

        if (constFields.Length == 1)
        {
            return null;
        }

        var shape = IR.F.ShapeExpr.BroadcastShape(constFields.Select(x => (Expr)x.Value).ToArray()).Evaluate().AsTensor();
        var exprFields = input.Fields.ToArray().Where(x => x is not TensorConst).ToArray();

        if(exprFields.Length == 0)
        {
            return shape;
        }

        if ((shape.Shape.Count == 0 || (shape.Shape.Count == 1 && shape.Shape[0] == 1)) && exprFields.Length != 0)
        {
            return IR.F.ShapeExpr.BroadcastShape(exprFields);
        }

        return IR.F.ShapeExpr.BroadcastShape(exprFields.Append(shape).ToArray());
    }
}

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
