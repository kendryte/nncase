// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// unroll loop and sequential.
/// </summary>
public sealed class UnRollLoopSequential : ExprRewriter
{
    private readonly Dictionary<Type, Evaluator.IEvaluator> _evaluator_cache = new();

    /// <inheritdoc/>
    protected internal override Expr VisitFor(For expr, Unit context)
    {
        var replace = TryUnroll(expr);
        if (!ReferenceEquals(expr, replace))
        {
            // only unroll one for loop
            return replace;
        }
        else
        {
            Visit(expr.LoopVar, context);
            Visit(expr.Domain, context);
            Visit(expr.Body, context);
            return VisitLeafFor(expr, context);
        }
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafSequential(TIR.Sequential expr)
    {
        if (expr.Fields.AsValueEnumerable().Any(f => f is Sequential))
        {
            return Sequential.Flatten(expr.Fields);
        }

        return expr;
    }

    /// <summary>
    /// convert the loop var to tensor const.
    /// </summary>
    private static IEnumerable<TensorConst> MakeGrid(TIR.For loop)
    {
        int start = ((TensorConst)loop.Domain.Start).Value.ToScalar<int>();
        int stop = ((TensorConst)loop.Domain.Stop).Value.ToScalar<int>();
        int step = ((TensorConst)loop.Domain.Step).Value.ToScalar<int>();

        for (int i = start; i < stop; i += step)
        {
            yield return i;
        }
    }

    private bool IsCanUnroll(TIR.For for_loop)
    {
        return for_loop.Domain.Start is TensorConst && for_loop.Domain.Stop is TensorConst && for_loop.Domain.Step is TensorConst && for_loop.Mode == LoopMode.Unrolled;
    }

    private Expr TryUnroll(TIR.For expr)
    {
        // try collect the all loops
        var nested_loops = new List<TIR.For>();

        var outter_for = expr;
        if (IsCanUnroll(outter_for))
        {
            nested_loops.Add(outter_for);
        }
        else
        {
            return outter_for;
        }

        while (outter_for.Body.Count == 1 && outter_for.Body[0] is TIR.For inner_for)
        {
            if (IsCanUnroll(inner_for))
            {
                nested_loops.Add(inner_for);
            }
            else
            {
                break;
            }

            outter_for = inner_for;
        }

        var vmaps = (from grid in LinqExtensions.CartesianProduct(from loop in nested_loops select MakeGrid(loop))
                     select grid.ToArray()).
          Select(grid =>
            {
                var vmap = new Dictionary<Var, TensorConst>();
                for (int i = 0; i < grid.Length; i++)
                {
                    vmap.Add(nested_loops[i].LoopVar, grid[i]);
                }

                return vmap;
            });

        // warming up
        var unrolled_first = new LoopBodyCloner(vmaps.First(), _evaluator_cache).Clone(nested_loops[^1].Body, default);
        var unrolled = new Expr[] { unrolled_first }.
            Concat(vmaps.
                Skip(1).

                // AsParallel().
                // AsOrdered().
                Select(vmap => new LoopBodyCloner(vmap, _evaluator_cache).Clone(nested_loops[^1].Body, default)));

        return Sequential.Flatten(unrolled.ToArray());
    }

    /// <summary>
    /// clone loop body and fold the math call.
    /// </summary>
    private sealed class LoopBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Var, TensorConst> _vmap;
        private readonly Dictionary<Var, IValue> _cmap;
        private readonly Dictionary<Type, Evaluator.IEvaluator> _evaluator_cache;

        public LoopBodyCloner(IReadOnlyDictionary<Var, TensorConst> vmap, Dictionary<Type, Evaluator.IEvaluator> evaluator_cache)
        {
            _vmap = vmap;
            _cmap = new(ReferenceEqualityComparer.Instance);
            _evaluator_cache = evaluator_cache;
            foreach (var p in vmap)
            {
                _cmap.Add(p.Key, Value.FromConst(p.Value));
            }
        }

        protected override Expr VisitLeafPhysicalBuffer(PhysicalBuffer expr, Unit context) => expr;

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            if (_vmap.TryGetValue(expr, out var result))
            {
                return result;
            }

            return expr;
        }

        protected override Expr VisitLeafCall(Call expr, Unit context)
        {
            var target = Clone(expr.Target, context);
            var arguments = CloneArray(expr.Arguments, context);
            if (target is Op op && op.CanFoldConstCall && arguments.AsValueEnumerable().All(e => e is Const))
            {
                return Const.FromValue(CompilerServices.Evaluate(expr.With(target, arguments), _cmap, _evaluator_cache));
            }

            if (target is Function fn)
            {
                var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance);
                foreach (var (v, arg) in fn.Parameters.ToArray().Zip(arguments.ToArray()))
                {
                    if (arg is not Const constArg)
                    {
                        return expr.With(target, arguments);
                    }

                    feedDict.Add(v, Value.FromConst(constArg));
                }

                return Const.FromValue(CompilerServices.Evaluate(fn.Body, feedDict, _evaluator_cache));
            }

            return expr.With(target, arguments);
        }
    }
}
