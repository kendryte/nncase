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
            var type = _context.CurrentCallResultTensorType();
            var arithm = type.Shape.Prod().FixedValue;
            return new Cost(arithm, arithm * DataTypes.GetLength(type.DType));
        }
    }
}