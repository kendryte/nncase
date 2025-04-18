// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class InlineFunction : RewriteRule<Pattern>
{
    public InlineFunction(int maxInlineSize)
    {
        MaxInlineSize = maxInlineSize;
    }

    public override Pattern Pattern => IsWildcard("expr", expr => expr is Call call && call.Target is Function);

    public int MaxInlineSize { get; }

    private Expr? GetReplace(Call expr)
    {
        var target = (Function)expr.Target;
        if (target.ModuleKind == Callable.StackVMModuleKind)
        {
            var count = ExprCollector.Collect(target.Body).Count;
            if (count <= MaxInlineSize)
            {
                var mapper = target.Parameters.ToArray().Zip(expr.Arguments.ToArray(), (p, a) => (p, a)).ToDictionary(x => x.p, x => x.a);
                var cloner = new FunctionBodyCloner(mapper);
                return cloner.Visit(target.Body, Unit.Default);
            }
        }

        return null;
    }
}

internal sealed class FunctionBodyCloner : ExprCloner<Unit>
{
    private readonly Dictionary<IVar, Expr> _mapper;

    public FunctionBodyCloner(Dictionary<IVar, Expr> mapper)
    {
        _mapper = mapper;
    }

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        return _mapper[expr];
    }

    protected override Expr VisitDimVar(DimVar expr, Unit context)
    {
        return _mapper[expr];
    }
}
