using System;
using System.Collections.Generic;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using TorchSharp;


namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor : ExprVisitor<Const, IRType>
    {
        private readonly EvaluatorContext _context;

        public EvaluatorVisitor(Dictionary<Var, Const> inputs)
        {
            _context = new EvaluatorContext(ExpressionMemo, inputs);
        }

        // when torch return a scalar, scalar's shape is {0}
        private static torch.Tensor _fixShape(Expr expr, torch.Tensor tensor) =>
            expr.CheckedShape.IsScalar ? tensor.view(new long[] { }) : tensor;

        private Type GetEvaluatorType(Expr target)
        {
            return typeof(IEvaluator<>).MakeGenericType(target.GetType());
        }

        private IEvaluator GetEvaluator(Expr target)
        {
            var t = GetEvaluatorType(target);
            return ((IEvaluator)ServiceLocator.Current.GetRequiredService(t));
        }

        public override Const VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            if (expr.Target is Function)
            {
                throw new NotImplementedException();
            }

            var target = (Op)expr.Target;
            return GetEvaluator(target).Visit(_context, target);
        }

        public override Const VisitLeaf(Const expr)
        {
            return expr;
        }

        public override Const VisitLeaf(Op expr)
        {
            // todo:maybe a problem
            return Const.FromScalar(0);
        }

        public override Const VisitLeaf(Function expr)
        {
            return Const.FromScalar(0);
        }

        public override Const VisitLeaf(IR.Tuple expr)
        {
            return Const.FromScalar(0);
        }

        public override Const VisitLeaf(Var expr)
        {
            if (!_context.Inputs.TryGetValue(expr, out var result))
            {
                throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
            }

            if (result is null)
                throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
            if (result.ValueType != expr.CheckedType)
            {
                throw new InvalidProgramException(
                  $"The Var {expr.Name} Require {expr.CheckedType} But Give {result.CheckedType}");
            }

            return result;
        }
    }
}