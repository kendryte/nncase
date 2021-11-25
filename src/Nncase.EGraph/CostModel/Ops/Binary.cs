using Nncase;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        private Cost VisitBinary(Binary binary)
        {
            // todo:broadcast
            var type = _context.CurrentCallResultTensorType();
            var arithm = type.Shape.Prod().FixedValue;
            // var rhsValue = _context.GetArgumentConst(binary, Binary.Rhs);
            return new Cost(
                binary.BinaryOp switch
                {
                    // BinaryOp.Add => expr,
                    // BinaryOp.Sub => expr,
                    // BinaryOp.Mul => expr,
                    // BinaryOp.Div => expr,
                    // BinaryOp.Mod => expr,
                    // BinaryOp.Min => expr,
                    // BinaryOp.Max => expr,
                    // BinaryOp.Pow => arithm * rhsValue.ToScalar<int>(), // todo when egraph how to get const?
                    // BinaryOp.BitwiseAnd => expr,
                    // BinaryOp.BitwiseOr => expr,
                    // BinaryOp.BitwiseXor => expr,
                    // BinaryOp.LogicalAnd => expr,
                    // BinaryOp.LogicalOr => expr,
                    // BinaryOp.LogicalXor => expr,
                    _ => arithm,
                });
        }
    }
}