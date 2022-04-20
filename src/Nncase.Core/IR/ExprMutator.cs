// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{

    /// <summary>
    /// IMutatable Define.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMutatable<T>
    {
        /// <summary>
        /// mutate the current object.
        /// </summary>
        /// <param name="mutator">ExprMutator.</param>
        /// <returns> new instance. </returns>
        T Mutate(ExprMutator mutator);
    }

    /// <summary>
    /// Expression matutor.
    /// </summary>
    public abstract class ExprMutator : ExprVisitor<Expr, IRType>
    {
        /// <summary>
        /// for speedup the Mutator, If is Mutated we need MutateLeaf recursive.
        /// </summary>
        public bool IsMutated { get; protected set; } = false;

        /// <inheritdoc/>
        public override Expr VisitLeaf(Call expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Target = Visit(expr.Target),
                Parameters = MutateArray(expr.Parameters, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Const expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Function expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Body = Visit(expr.Body),
                Parameters = new(expr.Parameters.Select(x => (Var)Visit(x))),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.PrimFunction expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Body = (TIR.Sequential)Visit(expr.Body),
                Parameters = new(expr.Parameters.Select(x => (TIR.Buffer)Visit(x))),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Op expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Tuple expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Fields = new(expr.Fields.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Var expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(None expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.IterVar expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Dom = MutateLeaf(expr.Dom),
                Value = (Var)Visit(expr.Value),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Sequential expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Fields = MutateArray(expr.Fields, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.For expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                LoopVar = (Var)Visit(expr.LoopVar),
                Dom = MutateLeaf(expr.Dom),
                Body = (TIR.Sequential)Visit(expr.Body),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.IfThenElse expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Condition = Visit(expr.Condition),
                Then = (TIR.Sequential)Visit(expr.Then),
                Else = (TIR.Sequential)Visit(expr.Else),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Block expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                // the block realize 
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                Predicate = Visit(expr.Predicate),
                IterVars = MutateArray(expr.IterVars, x => (TIR.IterVar)Visit(x)),

                // the block internal.
                Body = (TIR.Sequential)Visit(expr.Body),
                Reads = MutateArray(expr.Reads, b => (TIR.BufferRegion)Visit(b)),
                Writes = MutateArray(expr.Writes, b => (TIR.BufferRegion)Visit(b)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferStore expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Value = Visit(expr.Value),
                Indices = MutateArray(expr.Indices, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferLoad expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Indices = MutateArray(expr.Indices, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Let expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }

            return expr with
            {
                Var = (Var)Visit(expr.Var),
                Expression = Visit(expr.Expression),
                Body = (TIR.Sequential)Visit(expr.Body),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Buffer expr)
        {

            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferRegion expr)
        {

            var nexpr = MutateLeaf(expr);
            if (!expr.Equals(nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated)
            {
                return expr;
            }
            return expr with
            {
                Buffer = (TIR.Buffer)Visit(expr.Buffer),
                Region = MutateArray(expr.Region, MutateLeaf)
            };
        }

        /// <summary>
        /// defulat mutate leaf is not mutate.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr DefaultMutateLeaf(Expr expr) => expr;

        /// <summary>
        /// mutate the call.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Call expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the const.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Const expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the function.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Function expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the prim function.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.PrimFunction expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the op.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Op expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the tuple.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Tuple expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the var.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Var expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the var.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(None expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the itervar.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.IterVar expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the sequential.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.Sequential expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the for.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.For expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the for.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.IfThenElse expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the block.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.Block expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the bufferstore.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.BufferStore expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the buffer load.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.BufferLoad expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the let
        /// </summary>
        /// <param name="expr">let expr.</param>
        /// <returns>new expr.</returns>
        public virtual Expr MutateLeaf(TIR.Let expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the memref
        /// </summary>
        /// <param name="expr">new memref.</param>
        /// <returns>new expr.</returns>
        public virtual Expr MutateLeaf(TIR.Buffer expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the buffer region
        /// </summary>
        /// <param name="expr">new memref.</param>
        /// <returns>new expr.</returns>
        public virtual Expr MutateLeaf(TIR.BufferRegion expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate irarray list.
        /// </summary>
        /// <typeparam name="TInput"></typeparam>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="arrayList"></param>
        /// <param name="visitor"></param>
        /// <returns></returns>
        public virtual IRArrayList<TResult> MutateArray<TInput, TResult>(IRArrayList<TInput> arrayList, Func<TInput, TResult> visitor)
        {
            return new(arrayList.Select(visitor));
        }

        /// <summary>
        /// Mutate IRArray.
        /// </summary>
        /// <typeparam name="TInput"></typeparam>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="array"></param>
        /// <param name="visitor"></param>
        /// <returns></returns>
        public virtual IRArray<TResult> MutateArray<TInput, TResult>(IRArray<TInput> array, Func<TInput, TResult> visitor)
        {
            return new(array.Select(visitor));
        }

        /// <summary>
        /// mutate range.
        /// </summary>
        /// <param name="range"></param>
        /// <returns></returns>
        public virtual TIR.Range MutateLeaf(TIR.Range range)
        {
            return range with
            {
                Start = Visit(range.Start),
                Stop = Visit(range.Stop),
                Step = Visit(range.Step),
            };
        }
    }
}
