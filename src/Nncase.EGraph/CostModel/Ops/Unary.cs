using System;
using Nncase;
using Nncase.IR.Math;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        private Cost VisitUnary(Unary unary)
        {
            var arithm = _context.CurrentCallResultTensorType().Shape.Prod().FixedValue;
            return new Cost(
                unary.UnaryOp switch
                {
                    // UnaryOp.Abs => expr,
                    // UnaryOp.Ceil => expr,
                    // UnaryOp.Cos => expr,
                    // UnaryOp.Exp => expr,
                    // UnaryOp.Floor => expr,
                    // UnaryOp.Log => expr,
                    // UnaryOp.Neg => expr,
                    // UnaryOp.Round => expr,
                    // UnaryOp.Rsqrt => expr,
                    // UnaryOp.Sin => expr,
                    // UnaryOp.Sqrt => expr,
                    // UnaryOp.Square => expr,
                    // UnaryOp.Tanh => expr,
                    // UnaryOp.BitwiseNot => expr,
                    // UnaryOp.LogicalNot => expr,
                    _ => arithm
                });
        }
    }
}