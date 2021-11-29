using System;
using Nncase.Evaluator.Ops;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator
{
    public static class Evaluator
    {
        public static torch.Tensor Eval(this Expr expr)
        {
            if (expr.CheckedType is null or InvalidType)
            {
                throw new InvalidOperationException("Expr in Evaluator need a valid type");
            }
            var evaluatorVisitor = new EvaluatorVisitor();
            var result = evaluatorVisitor.Visit(expr);
            return expr.CheckedShape.IsScalar ? result.view(new long[] { }) : result;
        }
    }
}