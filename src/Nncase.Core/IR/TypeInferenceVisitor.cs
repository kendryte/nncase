// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR
{
    internal sealed class TypeInferenceContext : ITypeInferenceContext
    {
        private readonly Dictionary<Expr, IRType> _exprMemo;

        public TypeInferenceContext(Dictionary<Expr, IRType> exprMemo)
        {
            _exprMemo = exprMemo;
        }

        public Call? CurrentCall { get; set; }

        public Expr GetArgument(Op op, ParameterInfo parameter)
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

        public Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo)
        {
            return paramsInfo.Select(info => GetArgument(op, info)).ToArray();
        }
        
        public IRType GetArgumentType(Op op, ParameterInfo parameter) =>
            _exprMemo[GetArgument(op, parameter)];

        private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");
    }

    internal sealed class TypeInferenceVisitor : ExprVisitor<IRType, IRType>
    {
        private readonly TypeInferenceContext _context;

        public TypeInferenceVisitor()
        {
            _context = new TypeInferenceContext(ExpressionMemo);
        }

        /// <summary>
        /// Gets a value indicating whether is fully inferenced.
        /// </summary>
        public bool IsFullyInferenced { get; private set; } = true;

        /// <inheritdoc/>
        public override IRType VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            var type = expr.Target switch
            {
                Function func => ((CallableType)Visit(func)).ReturnType,
                Op op => op.InferInvokeResultTypeNoThrow(_context),
                _ => new InvalidType("Target of call expression should be either a function or an op."),
            };
            _context.CurrentCall = null;
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Const expr)
        {
            var type = expr.ValueType;
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Function expr)
        {
            var paramTypes = expr.Parameters.Select(Visit).ToArray();
            var type = new CallableType(expr.Body is Sequential seq ? Visit(seq.Last()) : Visit(expr.Body), ImmutableArray.Create(paramTypes));
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Op expr)
        {
            var paramTypes = expr.Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
            var type = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypes));
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Tuple expr)
        {
            var fieldTypes = expr.Fields.Select(Visit).ToArray();
            var type = new TupleType(ImmutableArray.Create(fieldTypes));
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Var expr)
        {
            var type = expr.TypeAnnotation ?? AnyType.Default;
            SetCheckedType(expr, type);
            return type;
        }

        /// <summary>
        /// Verify the expression sub field type is valid.
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="expr"></param>
        /// <param name="exprMsg"></param>
        void VerifySubField(Expr parent, Expr expr, TypePattern? pattern = null, [CallerArgumentExpression("expr")] string? exprMsg = null)
        {
            pattern ??= Utility.IsIRType();
            if (parent.CheckedType is null)
            {
                if (expr.CheckedType is InvalidType)
                {
                    SetCheckedType(parent, new InvalidType($"The {exprMsg} Is Invalid!"));
                }
                if (expr.CheckedType is AnyType any)
                {
                    SetCheckedType(parent, any);
                }
                if (!pattern.MatchLeaf(expr.CheckedType))
                {
                    SetCheckedType(parent, new InvalidType($"The {exprMsg} Require {pattern.Reason}"));
                }
            }
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(IterVar expr)
        {
            VerifySubField(expr, expr.Value);
            VerifySubField(expr, expr.Dom.Min);
            VerifySubField(expr, expr.Dom.Max);
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            var type = expr.TypeAnnotation;
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Sequential expr)
        {
            IRType type;
            foreach (var i in Enumerable.Range(0, expr.Fields.Count))
            {
                VerifySubField(expr, expr.Fields[i]);
            }
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            type = TupleType.Void;
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(For expr)
        {
            IRType type;
            VerifySubField(expr, expr.Dom.Min, Utility.IsIntegralScalar());
            VerifySubField(expr, expr.Dom.Max, Utility.IsIntegralScalar());
            VerifySubField(expr, expr.LoopVar, Utility.IsIntegralScalar());
            VerifySubField(expr, expr.Body, Utility.IsUnit());
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            type = TupleType.Void;
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(Block expr)
        {
            IRType type;
            foreach (var i in Enumerable.Range(0, expr.IterVars.Count))
            {
                VerifySubField(expr, expr.IterVars[i], Utility.IsIntegralScalar());
            }
            VerifySubField(expr, expr.InitBody, Utility.IsUnit());
            VerifySubField(expr, expr.Body, Utility.IsUnit());
            VerifySubField(expr, expr.Predicate, Utility.IsIntegralScalar());
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            type = TupleType.Void;
            SetCheckedType(expr, type);
            return type;
        }

        public override IRType VisitLeaf(BufferLoad expr)
        {
            VerifySubField(expr, expr.Buffer.Handle, Utility.IsHandle());
            foreach (var i in Enumerable.Range(0, expr.Indices.Count)) { VerifySubField(expr, expr.Indices[i], Utility.IsIntegralScalar()); }
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            var type = TensorType.Scalar(((HandleType)expr.Buffer.Handle.CheckedType!).DType);
            SetCheckedType(expr, type);
            return type;
        }

        public override IRType VisitLeaf(BufferStore expr)
        {
            VerifySubField(expr, expr.Buffer.Handle, Utility.IsHandle());
            foreach (var i in Enumerable.Range(0, expr.Indices.Count)) { VerifySubField(expr, expr.Indices[i], Utility.IsIntegralScalar()); }
            VerifySubField(expr, expr.Value, Utility.IsScalar());

            if (expr.CheckedType is not null) { return expr.CheckedType; }
            IRType type;
            if (expr.Value.CheckedDataType != expr.Buffer.Handle.CheckedDataType)
            {
                type = new InvalidType("The Value Type Is Not Equal Buffer Handle Type");
            }
            else
            {
                type = TupleType.Void;
            }
            SetCheckedType(expr, type);
            return type;
        }

        /// <inheritdoc/>
        public override IRType VisitLeaf(IfThenElse expr)
        {
            VerifySubField(expr, expr.Condition, Utility.IsIntegralScalar());
            VerifySubField(expr, expr.Then, Utility.IsUnit());
            VerifySubField(expr, expr.Else, Utility.IsUnit());
            if (expr.CheckedType is not null) { return expr.CheckedType; }
            IRType type = TupleType.Void;
            SetCheckedType(expr, type);
            return type;
        }

        /// <summary>
        /// set expr's current type
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="type"></param>
        private void SetCheckedType(Expr expr, IRType type)
        {
            expr.CheckedType = type;
            IsFullyInferenced &= type is not (AnyType or InvalidType);
        }
    }
}
