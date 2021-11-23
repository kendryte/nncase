using System;
using Nncase;
using Nncase.IR.Math;
using Nncase.IR.Tensors;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        private Cost VisitTranspose(Transpose transpose)
        {
            var arithm = _context.CurrentCallResultTensorType().Shape.Prod().FixedValue;
            return new Cost();
        }
    }
}