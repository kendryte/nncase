using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Utilities;

public static class ShapeExprUtility
{
    public static Expr BroadcastShape(Expr lhsShape, Expr rhsShape)
    {
        return lhsShape + (rhsShape * 0);
    }
}
