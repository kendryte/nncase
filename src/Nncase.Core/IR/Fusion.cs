﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Fusion expression.
/// </summary>
public class Fusion : BaseFunction
{
    private static int _globalFusionIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string name, string moduleKind, BaseExpr body, ReadOnlySpan<IVar> parameters)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<IVar, BaseExpr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string moduleKind, BaseExpr body, ReadOnlySpan<IVar> parameters)
        : this($"func_{_globalFusionIndex++}", moduleKind, body, parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string name, string moduleKind, BaseExpr body, params IVar[] parameters)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<IVar, BaseExpr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string moduleKind, BaseExpr body, params IVar[] parameters)
        : this($"func_{_globalFusionIndex++}", moduleKind, body, parameters)
    {
    }

    public BaseExpr Body => Operands[0];

    public ReadOnlySpan<IVar> Parameters => SpanUtility.UnsafeCast<BaseExpr, IVar>(Operands[1..]);

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((Expr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFusion(this, context);

    public Fusion With(string? name = null, string? moduleKind = null, BaseExpr? body = null, IVar[]? parameters = null)
        => new Fusion(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters);
}
