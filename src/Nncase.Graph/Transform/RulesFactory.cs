// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

/// <summary>
/// Rules Factory.
/// </summary>
public static class RulesFactory
{
    /// <summary>
    /// create the rewrite patternrule calss.
    /// </summary>
    /// <param name="lhs">lhs pattern.</param>
    /// <param name="rhs">rhs pattern expression.</param>
    /// <param name="predicate"> predicate pattern expression. </param>
    /// <returns> PatternRule. </returns>
    public static IRewriteRule Rewrite(Pattern lhs, Pattern rhs, Pattern? predicate = null)
      => new TemplateRule(lhs, rhs, predicate);
}

/// <summary>
/// a template rule.
/// </summary>
public class TemplateRule : IRewriteRule
{
    /// <summary>
    /// after expr.
    /// </summary>
    private readonly Pattern _rhs;

    /// <summary>
    /// predicate will be eval to bool.
    /// </summary>
    private readonly Pattern? _predicate;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemplateRule"/> class.
    /// <see cref="RulesFactory.Rewrite(Pattern, Pattern, Pattern?)"/>.
    /// </summary>
    public TemplateRule(Pattern lhs, Pattern rhs, Pattern? predicate = null)
    {
        Pattern = lhs;
        _rhs = rhs;
        _predicate = predicate;
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    /// <inheritdoc/>
    public Expr? GetReplace(IMatchResult result, RunPassContext options)
    {
        var converter = new ExprGeneratorVisitor(result);
        if (_predicate is null || (_predicate is not null && converter.Visit(_predicate).Evaluate().AsTensor().ToScalar<bool>()))
        {
            return converter.Visit(_rhs);
        }

        return null;
    }
}

/// <inheritdoc/>
internal sealed class ExprGeneratorVisitor : PatternVisitor<Expr, IRType>
{
    private readonly IMatchResult _result;

    public ExprGeneratorVisitor(IMatchResult result)
    {
        _result = result;
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(CallPattern pattern)
    {
        return new Call(PatternMemo[pattern.Target], pattern.Arguments.Select(p => PatternMemo[p]).ToArray());
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(ConstPattern pattern)
    {
        return _result.Get(pattern);
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TensorConstPattern pattern)
    {
        return _result.Get(pattern);
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TupleConstPattern pattern)
    {
        return _result.Get(pattern);
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(FunctionPattern pattern)
    {
        return new Function(PatternMemo[pattern.Body], pattern.Parameters.Select(p => (Var)PatternMemo[p]).ToArray());
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(IOpPattern pattern)
    {
        return (Expr)_result[pattern];
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(TuplePattern pattern)
    {
        return new IR.Tuple(pattern.Fields.Select(f => PatternMemo[f]).ToArray());
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(VarPattern pattern)
    {
        return _result.Get(pattern);
    }

    /// <inheritdoc/>
    public override Expr VisitLeaf(ExprPattern pattern)
    {
        return _result.Get(pattern);
    }
}
