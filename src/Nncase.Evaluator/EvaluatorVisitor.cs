using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using Autofac;
using Autofac.Core;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;


namespace Nncase.Evaluator.Ops
{
    public sealed class EvaluatorContext
    {
        public Call? CurrentCall;
        private readonly Dictionary<Expr, Const> _exprMemo;

        public readonly Dictionary<Var, Const> Inputs;

        public EvaluatorContext(Dictionary<Expr, Const> exprMemo, Dictionary<Var, Const> inputs)
        {
            _exprMemo = exprMemo;
            Inputs = inputs;
        }

        private Call GetCurrentCall() =>
            CurrentCall ?? throw new InvalidOperationException("Current call is not set in evaluator.");

        public torch.Tensor GetTorchArgument(Op op, ParameterInfo parameter)
        {
            return GetArgumentConst(op, parameter).ToTorchTensor();
        }

        public torch.Tensor GetTorchArgument(Expr expr)
        {
            return GetArgument(expr).ToTorchTensor();
        }
        
        public Tensorflow.Tensor GetTFArgument(Op op, ParameterInfo parameter)
        {
            return GetArgumentConst(op, parameter).ToTFTensor();
        }

        public Const GetArgument(Expr expr)
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

        public T GetArgumentConstScalar<T>(Op op, ParameterInfo parameter)
            where T : unmanaged
        {
            return GetArgumentConst(op, parameter).ToScalar<T>();
        }

        public T[] GetArgumentConstArray<T>(Op op, ParameterInfo parameter)
            where T : unmanaged
        {
            return GetArgumentConst(op, parameter).ToArray<T>();
        }

        public Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            var expr = GetArgumentExpr(op, parameter);
            if (expr is Const constValue)
            {
                return constValue;
            }
            else
            {
                // maybe a valid type but not const
                return GetArgument(expr);
            }
        }

        public TensorType GetTensorType(Expr expr)
        {
            var resultType = expr.CheckedType ?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
            return resultType is TensorType resultTensorType ?
                resultTensorType :
                throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
        }

        public TensorType CurrentCallResultTensorType() => GetTensorType(CurrentCall!);

    }

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

        public override Const VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            var target = expr.Target;
            var o = Evaluator.Container.Resolve(typeof(IEvaluator<>).MakeGenericType(target.GetType()));
            var method = o.GetType().GetMethod("Visit");
            if (method == null)
            {
                throw new NotSupportedException($"Evaluator {target} not implement or no method visit");
            }
            return (Const)method.Invoke(null, new object[] {_context, target});
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