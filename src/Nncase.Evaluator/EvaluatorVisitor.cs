using System;
using System.Collections.Generic;
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
        private readonly Dictionary<Expr, torch.Tensor> _exprMemo;

        public EvaluatorContext(Dictionary<Expr, torch.Tensor> exprMemo)
        {
            _exprMemo = exprMemo;
        }
        
        private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");

        public torch.Tensor GetArgument(Op op, ParameterInfo parameter)
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return _exprMemo[GetCurrentCall().Parameters[parameter.Index]];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
            // if (CurrentCall != null) return ToTorchTensor(CurrentCall.Parameters[index]);
            // else throw new NotImplementedException();
        }
    }
    
    public sealed partial class EvaluatorVisitor : ExprVisitor<torch.Tensor, IRType>
    {
        private EvaluatorContext _context;

        public EvaluatorVisitor()
        {
            _context = new EvaluatorContext(ExpressionMemo);
        }
        
        public override torch.Tensor VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            return expr.Target switch
            {
                Binary binary => VisitBinary(binary),
                _ => throw new NotImplementedException()
            };
        }

        public override torch.Tensor VisitLeaf(Const expr)
        {
            if (expr.ValueType.IsScalar)
            {
                return torch.tensor(expr.ToScalar<float>());
            }
            else
            {
                var shape = expr.CheckedShape.ToList().Select(x => x.FixedValue).ToList();
                return torch.tensor(expr.ToTensor<float>());
            }
        }

        public override torch.Tensor VisitLeaf(Op expr)
        {
            // todo:maybe a problem
            return torch.empty(1, 1);
        }
    }
}