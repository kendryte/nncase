// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Expression visitor.
/// </summary>
/// <typeparam name="TExprPatternResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
public abstract class PatternVisitor<TExprPatternResult, TTypeResult> : PatternFunctor<TExprPatternResult, TTypeResult>
{
    private readonly Dictionary<IPattern, TExprPatternResult> _patternMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<TypePattern, TTypeResult> _typeMemo = new();

    /// <summary>
    /// Gets pattern visit result memo.
    /// </summary>
    public Dictionary<IPattern, TExprPatternResult> PatternMemo => _patternMemo;

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(CallPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            Visit(pattern.Target);
            foreach (var param in pattern.Arguments)
            {
                Visit(param);
            }

            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(ConstPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(TensorConstPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(TupleConstPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(FunctionPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(IOpPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(TuplePattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            foreach (var field in pattern.Fields)
            {
                Visit(field);
            }

            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(ExprPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <inheritdoc/>
    public sealed override TExprPatternResult Visit(VarPattern pattern)
    {
        if (!_patternMemo.TryGetValue(pattern, out var result))
        {
            result = VisitLeaf(pattern);
            _patternMemo.Add(pattern, result);
        }

        return result;
    }

    /// <summary>
    /// Visit pattern.
    /// </summary>
    /// <param name="pattern">Expression.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(IPattern pattern)
    {
        return pattern switch
        {
            VarPattern var => VisitLeaf(var),
            ConstPattern con => VisitLeaf(con),
            TensorConstPattern con => VisitLeaf(con),
            TupleConstPattern con => VisitLeaf(con),
            FunctionPattern func => VisitLeaf(func),
            CallPattern call => VisitLeaf(call),
            TuplePattern tuple => VisitLeaf(tuple),
            IOpPattern op => VisitLeaf(op),
            ExprPattern wildcard => VisitLeaf(wildcard),
            _ => DefaultVisitLeaf(pattern),
        };
    }

    /// <summary>
    /// Visit expression pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(ExprPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf variable pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(VarPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf constant pattern.
    /// </summary>
    /// <param name="pattern">Constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(ConstPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf tensor constant pattern.
    /// </summary>
    /// <param name="pattern">Constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(TensorConstPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf tuple constant pattern.
    /// </summary>
    /// <param name="pattern">Constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(TupleConstPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf function pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(FunctionPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf call pattern.
    /// </summary>
    /// <param name="pattern">Call pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(CallPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf tuple pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(TuplePattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Visit leaf operator pattern.
    /// </summary>
    /// <param name="pattern">Operator pattern.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult VisitLeaf(IOpPattern pattern) => DefaultVisitLeaf(pattern);

    /// <summary>
    /// Default leaf visit routine.
    /// </summary>
    /// <param name="pattern">Expression.</param>
    /// <returns>Result.</returns>
    public virtual TExprPatternResult DefaultVisitLeaf(IPattern pattern)
    {
        throw new NotImplementedException($"Unhandled visit leaf routine for {pattern.GetType()}.");
    }

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(TypePattern pattern)
    {
        if (!_typeMemo.TryGetValue(pattern, out var result))
        {
            result = VisitTypeLeaf(pattern);
            _typeMemo.Add(pattern, result);
        }

        return result;
    }

    /// <summary>
    /// Visit leaf type pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitTypeLeaf(TypePattern pattern) => DefaultVisitTypeLeaf(pattern);

    /// <summary>
    /// Default visit leaf routine.
    /// </summary>
    /// <param name="pattern">Type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult DefaultVisitTypeLeaf(TypePattern pattern)
    {
        throw new NotImplementedException($"Unhandled visit leaf routine for {pattern.GetType()}.");
    }

    /// <summary>
    /// clear the Memo!.
    /// </summary>
    public virtual void Clear()
    {
        _patternMemo.Clear();
        _typeMemo.Clear();
    }
}
