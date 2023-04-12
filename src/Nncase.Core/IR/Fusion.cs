// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Fusion expression.
/// </summary>
public sealed class Fusion : BaseFunction
{
    private static int _globalFusionIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string name, string moduleKind, Expr body, ReadOnlySpan<Var> parameters)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<Var, Expr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string moduleKind, Expr body, ReadOnlySpan<Var> parameters)
        : this($"func_{_globalFusionIndex++}", moduleKind, body, parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string name, string moduleKind, Expr body, params Var[] parameters)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<Var, Expr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// build function.
    /// </summary>
    public Fusion(string moduleKind, Expr body, params Var[] parameters)
        : this($"func_{_globalFusionIndex++}", moduleKind, body, parameters)
    {
    }

    public Expr Body => Operands[0];

    public ReadOnlySpan<Var> Parameters => SpanUtility.UnsafeCast<Expr, Var>(Operands[1..]);

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType?> ParameterTypes => Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFusion(this, context);

    public Fusion With(string? name = null, string? moduleKind = null, Expr? body = null, Var[]? parameters = null)
        => new Fusion(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters);
}
