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
            var result = expr.Target switch
            {
                // todo:reflect to visit f
                BatchNormalization b => VisitBatchNormalization(b),
                Binary bn => VisitBinary(bn),
                Broadcast b => VisitBroadcast(b),
                Cast ct => VisitCast(ct),
                Celu c => VisitCelu(c),
                Concat con => VisitConcat(con),
                Conv2D conv => VisitConv2D(conv),
                Conv2DTranspose c => VisitConv2DTranspose(c),
                Elu e => VisitElu(e),
                Expand e => VisitExpand(e),
                Flatten f => VisitFlatten(f),
                HardSwish h => VisitHardSwish(h),
                InstanceNormalization i => VisitInstanceNormalization(i),
                LeakyRelu l => VisitLeakyRelu(l),
                LogSoftMax l => VisitLogSoftMax(l),
                LRN l => VisitLRN(l),
                MatMul m => VisitMatMul(m),
                Pad pd => VisitPad(pd),
                Prod p => VisitProd(p),
                IR.Tensors.Range r => VisitRange(r),
                ReduceArg r => VisitReduceArg(r),
                ReduceWindow2D r => VisitReduceWindow2D(r),
                Relu r => VisitRelu(r),
                Reshape rs => VisitReshape(rs),
                Selu s => VisitSelu(s),
                ShapeOp sp => VisitShape(sp),
                Sigmoid s => VisitSigmoid(s),
                Size s => VisitSize(s),
                Slice sl => VisitSlice(sl),
                SoftMax s => VisitSoftMax(s),
                SoftPlus s => VisitSoftPlus(s),
                // SoftSign s => VisitSoftSign(s),
                IR.Tensors.Stack st => VisitStack(st),
                Transpose tr => VisitTranspose(tr),
                Unary un => VisitUnary(un),
                Clamp cl => VisitClamp(cl),
                _ => TFOps(expr.Target)
            };
            return _fixShape(expr, result).ToConst();
        }

        private torch.Tensor TFOps(Expr target)
        {
            var result = target switch
            {
                CumSum c => VisitCumSum(c),
                Gather g => VisitGather(g),
                GatherND g => VisitGatherND(g),
                OneHot o => VisitOneHot(o),
                Reduce r => VisitReduce(r),
                _ => throw new NotImplementedException($"{target}")
            };
            return result.ToConst().ToTorchTensor();
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
            return result;
        }
    }
}