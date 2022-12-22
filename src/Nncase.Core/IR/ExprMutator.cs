// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// IVisitable interface for the custom class visit leaf.
/// </summary>
public interface IVisitable
{
    /// <summary>
    /// accept the visit.
    /// </summary>
    /// <typeparam name="TExprResult"></typeparam>
    /// <typeparam name="TTypeResult"></typeparam>
    /// <param name="functor"></param>
    /// <returns></returns>
    object Visit<TExprResult, TTypeResult>(ExprFunctor<TExprResult, TTypeResult> functor);
}

/// <summary>
/// IMutatable Define.
/// </summary>
public interface IMutatable : IVisitable
{
    /// <summary>
    /// mutate the current object.
    /// NOTE In order to ensure the consistency of coding, please return a new object.
    /// </summary>
    /// <param name="mutator">ExprMutator.</param>
    /// <returns> new instance. </returns>
    // object MutateLeaf(ExprMutator mutator);

    /// <summary>
    /// recursive build new object
    /// </summary>
    /// <param name="mutator">ExprMutator.</param>
    /// <returns></returns>
    object WithNew(ExprVisitor<Expr, IRType> mutator);
}

/// <summary>
/// Deep Expression matutor.
/// </summary>
public abstract class DeepExprMutator : ExprVisitor<Expr, IRType>
{
    /// <summary>
    /// The Struct Equal Memo folding the const/op.
    /// </summary>
    private readonly Dictionary<Expr, Expr> _exprSEqualMemo = new();

    /// <summary>
    /// Gets the Struct Equal Memo.
    /// </summary>
    public Dictionary<Expr, Expr> ExpressionStructMemo => _exprSEqualMemo;

    /// <summary>
    /// Gets or sets a value indicating whether for speedup the Mutator, If is Mutated we need MutateLeaf recursive.
    /// </summary>
    public bool IsMutated { get; set; }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Call expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return StructEqualFolding(expr);
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Function expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
    public override Expr VisitLeaf(Fusion expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
    public override Expr VisitLeaf(PrimFunctionWrapper expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            Target = (TIR.PrimFunction)Visit((IR.BaseFunction)expr.Target),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TIR.PrimFunction expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            Body = (TIR.Sequential)Visit(expr.Body),
            Parameters = new(expr.Parameters.Select(x => (TIR.PhysicalBuffer)Visit(x))),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Op expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
    public override Expr VisitLeaf(Var expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr;
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(Marker expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            Target = Visit(expr.Target),
            Attribute = Visit(expr.Attribute),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TIR.IterVar expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            Dom = (TIR.Range)Visit(expr.Dom),
            Value = (Var)Visit(expr.Value),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TIR.Sequential expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            LoopVar = (Var)Visit(expr.LoopVar),
            Domain = (TIR.Range)Visit(expr.Domain),
            Body = (TIR.Sequential)Visit(expr.Body),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TIR.IfThenElse expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            // the block realize
            InitBody = expr.InitBody.Fields.IsDefaultOrEmpty ? expr.InitBody : (TIR.Sequential)Visit(expr.InitBody),
            Predicate = Visit(expr.Predicate),
            IterVars = expr.IterVars.IsDefaultOrEmpty ? expr.IterVars : MutateArray(expr.IterVars, x => (TIR.IterVar)Visit(x)),

            // the block internal.
            Body = expr.Body.Fields.IsDefaultOrEmpty ? expr.Body : (TIR.Sequential)Visit(expr.Body),
            Reads = expr.Reads.IsDefaultOrEmpty ? expr.Reads : MutateArray(expr.Reads, b => (TIR.BufferRegion)Visit(b)),
            Writes = expr.Writes.IsDefaultOrEmpty ? expr.Writes : MutateArray(expr.Writes, b => (TIR.BufferRegion)Visit(b)),
            AllocBuffers = expr.AllocBuffers.IsDefaultOrEmpty ? expr.AllocBuffers : MutateArray(expr.AllocBuffers, b => (TIR.Buffer)Visit(b)),
        };
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TIR.BufferStore expr)
    {
        var nexpr = MutateLeaf(expr);
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

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
        if (!object.ReferenceEquals(expr, nexpr))
        {
            IsMutated = true;
            return nexpr;
        }

        if (!IsMutated)
        {
            return expr;
        }

        return expr with
        {
            Buffer = (TIR.Buffer)Visit(expr.Buffer),
            Region = MutateArray(expr.Region, rg => (TIR.Range)Visit(rg)),
        };
    }

    /// <inheritdoc/>
    public override object VisitLeaf(IVisitable visitable)
    {
        if (visitable is IMutatable mutatable)
        {
            var nexpr = MutateLeaf(mutatable);
            if (!object.ReferenceEquals(mutatable, nexpr))
            {
                IsMutated = true;
                return nexpr;
            }

            if (!IsMutated)
            {
                return mutatable;
            }

            return mutatable.WithNew(this);
        }

        throw new NotSupportedException($"IVisitable {visitable.GetType().Name} Is Not IMutatable!");
    }

    /// <summary>
    /// defulat mutate leaf is not mutate.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public virtual Expr DefaultMutateLeaf(Expr expr) => expr;

    /// <summary>
    /// default mutate leaf is not mutate.
    /// </summary>
    /// <param name="mutatable"></param>
    /// <returns></returns>
    public virtual IMutatable DefaultMutateLeaf(IMutatable mutatable) => mutatable;

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
    /// mutate the fusion.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public virtual Expr MutateLeaf(Fusion expr) => DefaultMutateLeaf(expr);

    /// <summary>
    /// mutate the prim function wrapper.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public virtual Expr MutateLeaf(PrimFunctionWrapper expr) => DefaultMutateLeaf(expr);

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
    /// mutate the marker.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public virtual Expr MutateLeaf(Marker expr) => DefaultMutateLeaf(expr);

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
    /// mutate the let.
    /// </summary>
    /// <param name="expr">let expr.</param>
    /// <returns>new expr.</returns>
    public virtual Expr MutateLeaf(TIR.Let expr) => DefaultMutateLeaf(expr);

    /// <summary>
    /// mutate the memref.
    /// </summary>
    /// <param name="expr">new memref.</param>
    /// <returns>new expr.</returns>
    public virtual Expr MutateLeaf(TIR.Buffer expr) => DefaultMutateLeaf(expr);

    /// <summary>
    /// mutate the buffer region.
    /// </summary>
    /// <param name="expr">new memref.</param>
    /// <returns>new expr.</returns>
    public virtual Expr MutateLeaf(TIR.BufferRegion expr) => DefaultMutateLeaf(expr);

    /// <summary>
    /// mutate the imutatable.
    /// </summary>
    /// <param name="mutatable">IMutatable instance.</param>
    /// <returns>new expr.</returns>
    public virtual IMutatable MutateLeaf(IMutatable mutatable) => DefaultMutateLeaf(mutatable);

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
    /// fold the expr by struct comparer.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    public virtual Expr StructEqualFolding(Expr expr)
    {
        if (!_exprSEqualMemo.TryGetValue(expr, out var folded))
        {
            folded = expr;
            _exprSEqualMemo.Add(expr, folded);
        }

        return folded;
    }
}

/// <summary>
/// NOTE the mutator only visit the only one basefunction and skip other basefunction.
/// </summary>
public abstract class ExprMutator : DeepExprMutator
{
    private BaseFunction? _entryBaseFunc;

    /// <inheritdoc/>
    public override Expr Visit(BaseFunction baseFunction)
    {
        if (_entryBaseFunc is null)
        {
            _entryBaseFunc = baseFunction;
        }

        return base.Visit(baseFunction);
    }

    /// <inheritdoc/>
    public override Expr Visit(Fusion expr)
    {
        if (_entryBaseFunc is null)
        {
            _entryBaseFunc = expr;
            return base.Visit(expr);
        }

        return object.ReferenceEquals(_entryBaseFunc, expr) ? base.Visit(expr) : expr;
    }

    /// <inheritdoc/>
    public override Expr Visit(Function expr)
    {
        if (_entryBaseFunc is null)
        {
            _entryBaseFunc = expr;
            return base.Visit(expr);
        }

        return object.ReferenceEquals(_entryBaseFunc, expr) ? base.Visit(expr) : expr;
    }

    /// <inheritdoc/>
    public override Expr Visit(PrimFunctionWrapper expr)
    {
        if (_entryBaseFunc is null)
        {
            _entryBaseFunc = expr;
            return base.Visit(expr);
        }

        return object.ReferenceEquals(_entryBaseFunc, expr) ? base.Visit(expr) : expr;
    }

    /// <inheritdoc/>
    public override Expr Visit(TIR.PrimFunction expr)
    {
        if (_entryBaseFunc is null)
        {
            _entryBaseFunc = expr;
            return base.Visit(expr);
        }

        return object.ReferenceEquals(_entryBaseFunc, expr) ? base.Visit(expr) : expr;
    }
}
