using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;
namespace Nncase.Utilities;

public static class ShapeExprUtility
{
    public static Expr BroadcastShape(Expr lhsShape, params Expr[] rhsShape)
    {
        var tmpTensor = new[] { lhsShape }
            .Concat(rhsShape)
            .Aggregate((sum, shape) => ConstantOfShape(shape, 0) * sum);
        return ShapeOf(tmpTensor);
    }

    public static Expr Positive(Expr axis, Expr inShape)
    {
        var rank = ShapeOf(inShape)[0];
        return new If(axis < 0, axis + rank, axis);
    }

    private static Expr StackOne(Expr expr) => Stack(new IR.Tuple(expr), 0);

    public static Expr Slice(Expr shape, int begin, int end)
    {
        return IR.F.Tensors.Slice(shape, new[] { begin }, new[] { end }, 1);
    }

    public static Expr Slice(Expr shape, Expr begin, Expr end)
    {
        return IR.F.Tensors.Slice(shape, StackOne(begin), StackOne(end), 1);
    }

    public static Expr Replace(Expr shapeExpr, Expr index, Expr value)
    {
        return SliceAndMerge(shapeExpr, index, value, 1);
    }

    public static Expr Insert(Expr shapeExpr, Expr index, Expr value)
    {
        return SliceAndMerge(shapeExpr, index, value, 0);
    }

    public static Expr Remove(Expr shapeExpr, Expr index)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, index + 1, int.MaxValue);
        return Concat(new IR.Tuple(front, last), 0);
    }

    private static Expr SliceAndMerge(Expr shapeExpr, Expr index, Expr value, Expr indexOffset)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, index + indexOffset, int.MaxValue);
        return Concat(new IR.Tuple(front, value, last), 0);
    }
}
