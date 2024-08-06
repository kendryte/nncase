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
    /// used for save expr in VarMap.
    /// </summary>
    private readonly ExprPinner? _pinner;

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(string name, Expr body, ReadOnlySpan<Var> parameters, string? moduleKind = null)
        : this(name, body, parameters, new Dictionary<Var, Expr[]>(), moduleKind)
    {
    }

    public Function(string name, string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Dictionary<Var, Expr[]>? varMap)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<Var, Expr>(parameters)))
    {
        VarMap = varMap ?? new();
        var dynamicDims = VarMap.Values.SelectMany(x => x).ToArray();
        _pinner = new ExprPinner(dynamicDims);
    }

    public Function(string name, Expr body, ReadOnlySpan<Var> parameters, Dictionary<Var, Expr[]>? varMap)
        : this(name, StackVMModuleKind, body, parameters, varMap)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(Expr body, ReadOnlySpan<Var> parameters, string? moduleKind = null)
        : this($"func_{_globalFuncIndex++}", body, parameters, new Dictionary<Var, Expr[]>(), moduleKind)
    {
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

    public Dictionary<Var, Expr[]>? VarMap { get; }

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType?> ParameterTypes => Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFunction(this, context);

    public Function With(string? name = null, string? moduleKind = null, Expr? body = null, Var[]? parameters = null)
        => new Function(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters, VarMap);
}
