using Nncase;
using Nncase.IR.Math;
using Nncase.IR;
using System.Linq;
using System;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        /// <summary>
        /// Compute binary cost by broadcast rule, if can broadcast the min dimension will have more memory loctiaity, so arithm should reduce.
        /// </summary>
        /// <param name="binary"></param>
        /// <returns></returns>
        private Cost VisitBinary(Binary binary)
        {
            // todo:broadcast
            var lhs = _context.GetArgumentType(binary, Binary.Lhs) as TensorType;
            var rhs = _context.GetArgumentType(binary, Binary.Rhs) as TensorType;
            var arithm = Math.Min(lhs.Shape.Prod().FixedValue,
                                  rhs.Shape.Prod().FixedValue);
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