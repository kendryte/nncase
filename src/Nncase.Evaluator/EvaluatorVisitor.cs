using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using TorchSharp;
#nullable enable

namespace Nncase.Evaluator.Ops
{
    public sealed class EvaluatorContext
    {
        public Call? CurrentCall;
        private readonly Dictionary<Expr, torch.Tensor> _exprMemo;

        public readonly Dictionary<Var, torch.Tensor> Inputs;

        public EvaluatorContext(Dictionary<Expr, torch.Tensor> exprMemo, Dictionary<Var, torch.Tensor> inputs)
        {
            _exprMemo = exprMemo;
            Inputs = inputs;
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

        public TensorType GetTensorType(Expr expr)
        {
            var resultType = expr.CheckedType ?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
            return resultType is TensorType resultTensorType ?
                resultTensorType :
                throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
        }

        public TensorType CurrentCallResultTensorType() => GetTensorType(CurrentCall!);

    }

    public sealed partial class EvaluatorVisitor : ExprVisitor<torch.Tensor, IRType>
    {
        private readonly EvaluatorContext _context;

        public EvaluatorVisitor(Dictionary<Var, torch.Tensor> inputs)
        {
            _context = new EvaluatorContext(ExpressionMemo, inputs);
        }

        private torch.Tensor _fixShape(Expr expr, torch.Tensor tensor) => expr.CheckedShape.IsScalar ? tensor.view(new long[] { }) : tensor;

        public override torch.Tensor VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            return _fixShape(expr, expr.Target switch
            {
                BatchNormalization b => VisitBatchNormalization(b),
                Binary bn => VisitBinary(bn),
                Broadcast b => VisitBroadcast(b),
                Cast ct => VisitCast(ct),
                Celu c => VisitCelu(c),
                Concat con => VisitConcat(con),
                Conv2D conv => VisitConv2D(conv),
                Expand e => VisitExpand(e),
                Pad pd => VisitPad(pd),
                ReduceArg r => VisitReduceArg(r),
                Reshape rs => VisitReshape(rs),
                ShapeOp sp => VisitShape(sp),
                Slice sl => VisitSlice(sl),
                IR.Tensors.Stack st => VisitStack(st),
                Transpose tr => VisitTranspose(tr),
                Unary un => VisitUnary(un),
                _ => throw new NotImplementedException($"{expr.Target}")
            });
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

        public override torch.Tensor VisitLeaf(Var expr)
        {
            if (!_context.Inputs.TryGetValue(expr, out var result))
            {
                throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
            }
            if (result is null)
                throw new InvalidProgramException($"Must Set Input For Var {expr.Name}!");
            var target = new TensorType(result.dtype.ToDataType(), new Shape(result.shape));
            if (target != expr.CheckedType)
            {
                throw new InvalidProgramException(
                  $"The Var {expr.Name} Require {expr.CheckedType} But Give {target}");
            }
            return result;
        }
    }
}