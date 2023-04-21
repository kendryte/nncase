// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Function expression.
/// </summary>
public sealed class Function : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(string name, Expr body, ReadOnlySpan<Var> parameters, Dictionary<Var, Expr[]> varMap = null)
        : base(name, StackVMModuleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<Var, Expr>(parameters)))
    {
        VarMap = varMap;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(Expr body, ReadOnlySpan<Var> parameters, Dictionary<Var, Expr[]> varMap = null)
        : this($"func_{_globalFuncIndex++}", body, parameters)
    {
        VarMap = varMap;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(string name, Expr body, params Var[] parameters)
        : this(name, body, parameters.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(Expr body, params Var[] parameters)
        : this(body, parameters.AsSpan())
    {
    }

    public Expr Body => Operands[0];

    public ReadOnlySpan<Var> Parameters => SpanUtility.UnsafeCast<Expr, Var>(Operands[1..]);

    public Dictionary<Var, Expr[]> VarMap { get; }
    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType?> ParameterTypes => Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFunction(this, context);

    public Function With(string? name = null, Expr? body = null, Var[]? parameters = null)
        => new Function(name ?? Name, body ?? Body, parameters ?? Parameters);
}
