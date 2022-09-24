// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;

/// <summary>
/// unroll loop
/// </summary>
internal sealed class UnRollLoop : ExprMutator
{

    private readonly Dictionary<Type, Evaluator.IEvaluator> _evaluator_cache = new();

    /// <inheritdoc/>
    public override Expr Visit(TIR.For expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            result = TryUnroll(expr);
            if (!object.ReferenceEquals(result, expr))
            {
                IsMutated = true;
                ExpressionMemo[expr] = result;
                // only unroll one for loop
                return result;
            }
            return base.Visit(expr);
        }
        return result;
    }

    bool IsCanUnroll(TIR.For for_loop)
    {
        return (for_loop.Domain.Start is TensorConst && for_loop.Domain.Stop is TensorConst && for_loop.Domain.Step is TensorConst && for_loop.Mode == LoopMode.Unrolled);
    }

    /// <summary>
    /// convert the loop var to tensor const.
    /// </summary>
    /// <param name="loop"></param>
    /// <returns></returns>
    static IEnumerable<TensorConst> MakeGrid(TIR.For loop)
    {
        int start = ((TensorConst)loop.Domain.Start).Value.ToScalar<int>();
        int stop = ((TensorConst)loop.Domain.Stop).Value.ToScalar<int>();
        int step = ((TensorConst)loop.Domain.Step).Value.ToScalar<int>();

        for (int i = start; i < stop; i += step)
        {
            yield return i;
        }
    }

    Expr TryUnroll(TIR.For expr)
    {
        // try collect the all loops
        var nested_loops = new List<TIR.For>();

        var outter_for = expr;
        if (IsCanUnroll(outter_for))
            nested_loops.Add(outter_for);
        else
            return outter_for;

        while (outter_for.Body.Count == 1 && outter_for.Body[0] is TIR.For inner_for)
        {
            if (IsCanUnroll(inner_for))
                nested_loops.Add(inner_for);
            else
                break;
            outter_for = inner_for;
        }

        var vmaps = (from grid in LinqExtensions.CartesianProduct(from loop in nested_loops select MakeGrid(loop))
                     select grid.ToArray()).
          Select(grid =>
            {
                var vmap = new Dictionary<Var, TensorConst>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < grid.Length; i++)
                {
                    vmap.Add(nested_loops[i].LoopVar, grid[i]);
                }

                return vmap;
            });

        /// warming up
        var unrolled_first = new OptimizedSubstitutor(vmaps.First(), _evaluator_cache).Visit(nested_loops[^1].Body);
        var unrolled = new[] { unrolled_first }.
            Concat(vmaps.
                Skip(1).
                AsParallel().
                Select(vmap => new OptimizedSubstitutor(vmap, _evaluator_cache).Visit(nested_loops[^1].Body))).
            ToImmutableArray();

        return new Sequential(new IRArray<Expr>(unrolled));
    }

    /// <summary>
    /// fold the math operations, avoid too much call.
    /// </summary>
    private sealed class OptimizedSubstitutor : ExprMutator
    {
        private readonly IReadOnlyDictionary<Var, TensorConst> _vmap;
        private readonly Dictionary<Var, IValue> _cmap;
        private readonly Dictionary<Type, Evaluator.IEvaluator> _evaluator_cache;

        public OptimizedSubstitutor(IReadOnlyDictionary<Var, TensorConst> vmap, Dictionary<Type, Evaluator.IEvaluator> evaluator_cache)
        {
            _vmap = vmap;
            _cmap = new(ReferenceEqualityComparer.Instance);
            _evaluator_cache = evaluator_cache;
            foreach (var p in vmap)
            {
                _cmap.Add(p.Key, Value.FromConst(p.Value));
            }
        }

        public override Expr MutateLeaf(IR.Var expr)
        {
            if (_vmap.TryGetValue(expr, out var result))
            {
                return result;
            }
            return expr;
        }

        /// <inheritdoc/>
        public override Expr MutateLeaf(Call expr)
        {
            if (expr.Target is Op op && op.GetType().Namespace is string @namespace
              && (@namespace.StartsWith("Nncase.IR.Math") || @namespace.StartsWith("Nncase.IR.Tensors"))
              && expr.Parameters.Select(Visit).All(e => e is Const))
            {
                return StructEqualFolding(Const.FromValue(CompilerServices.Evaluate(expr, _cmap, _evaluator_cache)));
            }
            if (expr.Target is Function fn)
            {
                var arg_map = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance);
                foreach (var (v, arg) in fn.Parameters.Zip(expr.Parameters))
                {
                    if (Visit(arg) is not Const const_arg)
                        return expr;
                    arg_map.Add(v, Value.FromConst(const_arg));
                }
                return StructEqualFolding(Const.FromValue(CompilerServices.Evaluate(fn.Body, arg_map, _evaluator_cache)));
            }
            return expr;
        }
    }
}
