using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.ShapeExpr;

public class BroadcastShape : ShapeExprOp
{
    public static readonly ParameterInfo Inputs = new(typeof(BroadcastShape), 0, "inputs");
}
