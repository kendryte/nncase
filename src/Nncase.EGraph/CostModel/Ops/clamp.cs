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
        private Cost VisitClamp(Clamp clamp)
        {
            // todo:broadcast
            var shape = _context.GetArgumentType(clamp, Clamp.Input) as TensorType;
            var arithm = (shape.Shape.Prod() * 2).FixedValue;
            // var rhsValue = _context.GetArgumentConst(binary, Binary.Rhs);
            return new Cost(arithm);
        }
    }
}