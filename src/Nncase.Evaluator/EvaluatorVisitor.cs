using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using TorchSharp;

namespace Nncase.Evaluator.Ops
{
    public sealed class EvaluatorContext
    {
        public Call? CurrentCall;
        
        public torch.Tensor GetParam(int index)
        {
            if (CurrentCall != null) return ToTorchTensor(CurrentCall.Parameters[index]);
            else throw new NotImplementedException();
        }
        
        private torch.Tensor ToTorchTensor(Expr expr)
        {
            if (expr is Const exprValue)
            {
                if (exprValue.ValueType.IsScalar)
                {
                    return torch.tensor(exprValue.ToScalar<float>());
                }
                else
                {
                    var shape = expr.CheckedShape.ToList().Select(x => x.FixedValue).ToList();
                    return torch.tensor(exprValue.ToTensor<float>());
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
    
    public sealed partial class EvaluatorVisitor : ExprFunctor<torch.Tensor, torch.Tensor>
    {
        private EvaluatorContext _context;

        public EvaluatorVisitor()
        {
            _context = new EvaluatorContext();
        }
        
        public override torch.Tensor Visit(Call expr)
        {
            _context.CurrentCall = expr;
            return Visit(expr.Target);
        }

        public override torch.Tensor Visit(Op expr)
        {
            return expr switch
            {
                Sigmoid => torch.nn.functional.Sigmoid(_context.GetParam(0)),
                Binary binary => VisitBinary(binary),
                _ => throw new NotImplementedException()
            };
        }
    }
}