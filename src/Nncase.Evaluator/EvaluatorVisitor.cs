using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
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

        private Call GetCurrentCall() =>
            CurrentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");

        public torch.Tensor GetArgument(Op op, ParameterInfo parameter)
        {
            return _exprMemo[GetArgumentExpr(op, parameter)];
        }

        public torch.Tensor GetArgument(Expr expr)
        {
            return _exprMemo[expr];
        }

        public Expr GetArgumentExpr(Op op, ParameterInfo parameter)
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return GetCurrentCall().Parameters[parameter.Index];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            if (GetArgumentExpr(op, parameter) is Const constValue)
            {
                return constValue;
            }
            else
            {
                throw new InvalidOperationException($"Op:{op} Parameter:{parameter} is not const");
            }
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
                Binary bn => VisitBinary(bn),
                Concat con => VisitConcat(con),
                ShapeOp sp => VisitShape(sp),
                Slice sl => VisitSlice(sl),
                Transpose tr => VisitTranspose(tr),
                Unary un => VisitUnary(un),
                Pad pd => VisitPad(pd),
                Stack st => VisitStack(st),
                Cast ct => VisitCast(ct),
                _ => throw new NotImplementedException()
            };
        }

        public override torch.Tensor VisitLeaf(Const expr)
        {
            return expr.ToTorchTensor();
        }

        public override torch.Tensor VisitLeaf(Op expr)
        {
            // todo:maybe a problem
            return torch.empty(1, 1);
        }

        public override torch.Tensor VisitLeaf(Function expr)
        {
            return torch.empty(1, 1);
        }

        public override torch.Tensor VisitLeaf(IR.Tuple expr)
        {
            return torch.empty(1, 1);
        }

        public override torch.Tensor VisitLeaf(Var expr) => expr.CheckedType switch
        {
            TensorType ttype =>
              ttype.IsScalar switch
              {
                  true => torch.empty(new long[] { 1 }),
                  false => torch.empty(expr.CheckedShape.Select(x => (long)x.FixedValue).ToArray())
              },
            _ => torch.empty(1, 1)
        };
    }
}