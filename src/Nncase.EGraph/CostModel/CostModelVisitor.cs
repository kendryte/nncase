// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;

namespace Nncase.CostModel
{
    public class ExprCostModelContext
    {
        private Call? _currentCall = null;

        public virtual Call CurrentCall
        {
            get => _currentCall ?? throw new InvalidOperationException("Current call is not set.");
            set => _currentCall = value;
        }

        public Dictionary<Expr, Cost> ExpressionMemo { get; set; } = new();

        private Expr GetArgument(Op op, ParameterInfo parameter)
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return CurrentCall.Parameters[parameter.Index];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public virtual Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            if (GetArgument(op, parameter) is Const constValue)
            {
                return constValue;
            }
            else
            {
                throw new InvalidOperationException($"Op:{op} Parameter:{parameter} is not const");
            }
        }

        public virtual TensorType GetTensorType(Expr expr)
        {
            var resultType = expr.CheckedType ?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
            return resultType is TensorType resultTensorType ?
                resultTensorType :
                throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
        }

        public virtual TensorType GetArgumentType(Op op, ParameterInfo parameter) =>
            GetTensorType(GetArgument(op, parameter));

        public virtual TensorType CurrentCallResultTensorType() => GetTensorType(CurrentCall);

        public virtual Cost GetCostFromMemo(Expr expr, int i) => ExpressionMemo[expr];
    }

    public sealed partial class ExprCostModelVisitor : ExprVisitor<Cost, IRType>
    {
        private readonly ExprCostModelContext _context;

        public ExprCostModelVisitor()
        {
            _context = new ExprCostModelContext();
            _context.ExpressionMemo = ExpressionMemo;
        }

        public ExprCostModelVisitor(ExprCostModelContext context)
        {
            _context = context;
            _context.ExpressionMemo = ExpressionMemo;
        }

        public override Cost VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            return expr.Target switch
            {
                Binary bn => VisitBinary(bn),
                // Concat con => VisitConcat(con),
                Conv2D conv => VisitConv2D(conv),
                // Slice sl => VisitSlice(sl),
                // Transpose tr => VisitTranspose(tr),
                Unary un => VisitUnary(un),
                ShapeOp => throw new InvalidDataException("ShapeOp should be eliminate before CostModelVisitor"),
                _ => throw new NotImplementedException()
            };
        }

        public override Cost VisitLeaf(Const expr) =>
         new Cost(0, (ulong)DataTypes.GetLength(_context.GetTensorType(expr).DType));

        public override Cost VisitLeaf(Op expr) => new Cost();

        public override Cost VisitLeaf(Function expr) => new Cost();

        public override Cost VisitLeaf(IR.Tuple expr) => new Cost();

        public override Cost VisitLeaf(Var expr) => new Cost();

    }
}