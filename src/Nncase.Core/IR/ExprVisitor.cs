// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Expression visitor.
    /// </summary>
    /// <typeparam name="TExprResult">Expression visit result type.</typeparam>
    /// <typeparam name="TTypeResult">Type visit result type.</typeparam>
    public abstract class ExprVisitor<TExprResult, TTypeResult> : ExprFunctor<TExprResult, TTypeResult>
    {
        private readonly Dictionary<Expr, TExprResult> _exprMemo = new Dictionary<Expr, TExprResult>(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<IRType, TTypeResult> _typeMemo = new Dictionary<IRType, TTypeResult>();

        /// <summary>
        /// Gets expression visit result memo.
        /// </summary>
        public Dictionary<Expr, TExprResult> ExpressionMemo => _exprMemo;

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Call expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.Target);
                foreach (var param in expr.Parameters)
                {
                    Visit(param);
                }

                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Const expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Function expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                foreach (var param in expr.Parameters)
                {
                    Visit(param);
                }

                Visit(expr.Body);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Op expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Tuple expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                foreach (var field in expr.Fields)
                {
                    Visit(field);
                }

                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(Var expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.IterVar expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.Value);
                Visit(expr.Dom.Min);
                Visit(expr.Dom.Max);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.Sequential expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                foreach (var item in expr.Fields)
                {
                    Visit(item);
                }

                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.For expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.LoopVar);
                Visit(expr.Dom.Min);
                Visit(expr.Dom.Max);
                Visit(expr.Sequence);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.Block expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.InitSequence);
                Visit(expr.Predicate);
                foreach (var iterVar in expr.IterVars) { Visit(iterVar); }
                Visit(expr.Sequence);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.BufferLoad expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                foreach (var index in expr.Indices) { Visit(index); }
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.BufferStore expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.Buffer.Handle);
                foreach (var index in expr.Indices) { Visit(index); }
                Visit(expr.Value);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TExprResult Visit(TIR.IfThenElse expr)
        {
            if (!_exprMemo.TryGetValue(expr, out var result))
            {
                Visit(expr.Condition);
                Visit(expr.Then);
                Visit(expr.Else);
                result = VisitLeaf(expr);
                _exprMemo.Add(expr, result);
            }

            return result;
        }

        /// <summary>
        /// Visit expression.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Expr expr)
        {
            return expr switch
            {
                Var var => VisitLeaf(var),
                Const con => VisitLeaf(con),
                Function func => VisitLeaf(func),
                Call call => VisitLeaf(call),
                Tuple tuple => VisitLeaf(tuple),
                Op op => VisitLeaf(op),
                TIR.Sequential seq => VisitLeaf(seq),
                TIR.For @for => VisitLeaf(@for),
                TIR.Block block => VisitLeaf(block),
                TIR.BufferLoad bufload => VisitLeaf(bufload),
                TIR.BufferStore bufstore => VisitLeaf(bufstore),
                TIR.IfThenElse ift => VisitLeaf(ift),
                _ => DefaultVisitLeaf(expr),
            };
        }

        /// <summary>
        /// Visit leaf variable expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Var expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf constant expression.
        /// </summary>
        /// <param name="expr">Constant expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Const expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf function expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Function expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf call expression.
        /// </summary>
        /// <param name="expr">Call expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Call expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf tuple expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Tuple expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf operator expression.
        /// </summary>
        /// <param name="expr">Operator expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(Op expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf IterVar expression.
        /// </summary>
        /// <param name="expr">IterVar expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.IterVar expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf sequential expression.
        /// </summary>
        /// <param name="expr">sequential expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.Sequential expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf For expression.
        /// </summary>
        /// <param name="expr">For expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.For expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf Block expression.
        /// </summary>
        /// <param name="expr">Block expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.Block expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf BufferLoad expression.
        /// </summary>
        /// <param name="expr">BufferLoad expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.BufferLoad expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf BufferRead expression.
        /// </summary>
        /// <param name="expr">BufferRead expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.BufferStore expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Visit leaf IfThenElse expression.
        /// </summary>
        /// <param name="expr">IfThenElse expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult VisitLeaf(TIR.IfThenElse expr) => DefaultVisitLeaf(expr);

        /// <summary>
        /// Default leaf visit routine.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult DefaultVisitLeaf(Expr expr)
        {
            throw new NotImplementedException($"Unhandled visit leaf routine for {expr.GetType()}.");
        }

        /// <inheritdoc/>
        public sealed override TTypeResult VisitType(AnyType type)
        {
            if (!_typeMemo.TryGetValue(type, out var result))
            {
                result = VisitTypeLeaf(type);
                _typeMemo.Add(type, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TTypeResult VisitType(CallableType type)
        {
            if (!_typeMemo.TryGetValue(type, out var result))
            {
                foreach (var param in type.Parameters)
                {
                    VisitType(param);
                }

                VisitType(type.ReturnType);
                result = VisitTypeLeaf(type);
                _typeMemo.Add(type, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TTypeResult VisitType(InvalidType type)
        {
            if (!_typeMemo.TryGetValue(type, out var result))
            {
                result = VisitTypeLeaf(type);
                _typeMemo.Add(type, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TTypeResult VisitType(TensorType type)
        {
            if (!_typeMemo.TryGetValue(type, out var result))
            {
                result = VisitTypeLeaf(type);
                _typeMemo.Add(type, result);
            }

            return result;
        }

        /// <inheritdoc/>
        public sealed override TTypeResult VisitType(TupleType type)
        {
            if (!_typeMemo.TryGetValue(type, out var result))
            {
                foreach (var field in type.Fields)
                {
                    VisitType(field);
                }

                result = VisitTypeLeaf(type);
                _typeMemo.Add(type, result);
            }

            return result;
        }

        /// <summary>
        /// Visit any type leaf.
        /// </summary>
        /// <param name="type">Any type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult VisitTypeLeaf(AnyType type) => DefaultVisitTypeLeaf(type);

        /// <summary>
        /// Visit invalid type leaf.
        /// </summary>
        /// <param name="type">Invalid type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult VisitTypeLeaf(InvalidType type) => DefaultVisitTypeLeaf(type);

        /// <summary>
        /// Visit tensor type leaf.
        /// </summary>
        /// <param name="type">Tensor type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult VisitTypeLeaf(TensorType type) => DefaultVisitTypeLeaf(type);

        /// <summary>
        /// Visit tuple type leaf.
        /// </summary>
        /// <param name="type">Tuple type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult VisitTypeLeaf(TupleType type) => DefaultVisitTypeLeaf(type);

        /// <summary>
        /// Visit tuple type leaf.
        /// </summary>
        /// <param name="type">Callable type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult VisitTypeLeaf(CallableType type) => DefaultVisitTypeLeaf(type);

        /// <summary>
        /// Default visit leaf routine.
        /// </summary>
        /// <param name="type">Type.</param>
        /// <returns>Result.</returns>
        public virtual TTypeResult DefaultVisitTypeLeaf(IRType type)
        {
            throw new NotImplementedException($"Unhandled visit leaf routine for {type.GetType()}.");
        }

        /// <summary>
        /// clear the Memo!.
        /// </summary>
        public virtual void Clear()
        {
            _exprMemo.Clear();
            _typeMemo.Clear();
        }
    }
}
