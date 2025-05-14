﻿// Copyright (c) Canaan Inc. All rights reserved.
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
    private readonly Dictionary<BaseExpr, BaseExpr> _cseMemo = new(ReferenceEqualityComparer.Instance);

    /// <inheritdoc/>
    protected internal override BaseExpr VisitFor(For expr, Unit context)
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
    private static IEnumerable<long> MakeGrid(TIR.For loop)
    {
        long start = loop.Domain.Start.FixedValue;
        long stop = loop.Domain.Stop.FixedValue;
        long step = loop.Domain.Step.FixedValue;

        for (long i = start; i < stop; i += step)
        {
            yield return i;
        }
    }

    private bool IsCanUnroll(TIR.For for_loop)
    {
        return for_loop.Domain.Start.IsFixed && for_loop.Domain.Stop.IsFixed && for_loop.Domain.Step.IsFixed && for_loop.Mode == LoopMode.Unrolled;
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
                var vmap = new Dictionary<IVar, long>();
                for (int i = 0; i < grid.Length; i++)
                {
                    vmap.Add(nested_loops[i].LoopVar, grid[i]);
                }

                return vmap;
            });

        // warming up
        var unrolled_first = new LoopBodyCloner(vmaps.First(), _evaluator_cache, _cseMemo).Clone(nested_loops[^1].Body, default);
        var unrolled = new Expr[] { unrolled_first }.
            Concat(vmaps.
                Skip(1).
                Select(vmap => new LoopBodyCloner(vmap, _evaluator_cache, _cseMemo).Clone(nested_loops[^1].Body, default)));

        return Sequential.Flatten(unrolled.ToArray());
    }
}

/// <summary>
/// clone loop body and fold the math call.
/// </summary>
internal sealed class LoopBodyCloner : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<IVar, long> _vmap;
    private readonly Dictionary<IVar, IValue> _cmap;
    private readonly Dictionary<Type, Evaluator.IEvaluator> _evaluator_cache;
    private readonly IDictionary<BaseExpr, BaseExpr> _cseMemo;

    public LoopBodyCloner(IReadOnlyDictionary<IVar, long> vmap, Dictionary<Type, Evaluator.IEvaluator> evaluator_cache, IDictionary<BaseExpr, BaseExpr> cseMemo)
    {
        _vmap = vmap;
        _cmap = new(ReferenceEqualityComparer.Instance);
        _evaluator_cache = evaluator_cache;
        _cseMemo = cseMemo;
        foreach (var p in vmap)
        {
            _cmap.Add(p.Key, Value.FromConst(p.Value));
        }
    }

    protected override BaseExpr VisitLeafMemSpan(MemSpan expr, Unit context)
    {
        return expr.With(Clone(expr.Start, context), Clone(expr.Size, context));
    }

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        if (_vmap.TryGetValue(expr, out var result))
        {
            return result;
        }

        return expr;
    }

    protected override Dimension VisitLeafDimVar(DimVar expr, Unit context)
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
        if (target is Op op && op.CanFoldConstCall && arguments.AsValueEnumerable().All(e => e is Const or DimConst or RankedShape { IsFixed: true }))
        {
            return CSE(Const.FromValue(CompilerServices.Evaluate(expr.With(target, arguments), _cmap, _evaluator_cache)));
        }

        if (target is Function fn)
        {
            var feedDict = new Dictionary<IVar, IValue>(ReferenceEqualityComparer.Instance);
            foreach (var (v, arg) in fn.Parameters.ToArray().Zip(arguments.ToArray()))
            {
                if (arg is not Const constArg)
                {
                    return expr.With(target, arguments);
                }

                feedDict.Add(v, Value.FromConst(constArg));
            }

            return CSE(Const.FromValue(CompilerServices.Evaluate(fn.Body, feedDict, _evaluator_cache)));
        }

        return expr.With(target, arguments);
    }

    protected override Expr VisitLeafRange(TIR.Range expr, Unit context)
    {
        return CSE(expr.With(start: Clone(expr.Start, context), stop: Clone(expr.Stop, context), step: Clone(expr.Step, context)));
    }

    protected override Expr VisitLeafBuffer(TIR.Buffer expr, Unit context)
    {
        return expr.With(
            memSpan: Clone<MemSpan>(expr.MemSpan, context),
            dimensions: CloneArray(expr.Dimensions, context).Select(e => CSE(e)).ToArray(),
            strides: CloneArray(expr.Strides, context));
    }

    private T CSE<T>(T c)
        where T : BaseExpr
    {
        if (!_cseMemo.TryGetValue(c, out var result))
        {
            result = c;
            _cseMemo.Add(c, result);
        }

        return (T)result;
    }
}
