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
    public Function(string name, Expr body, ReadOnlySpan<IVar> parameters)
        : this(name, body, parameters, new Dictionary<Var, Dimension[]>())
    {
    }

    public Function(string name, string moduleKind, Expr body, ReadOnlySpan<IVar> parameters, Dictionary<Var, Dimension[]>? varMap = null)
        : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<IVar, Expr>(parameters)))
    {
        VarMap = varMap ?? new();
        var dynamicDims = VarMap.Values.SelectMany(x => x).ToArray();
        _pinner = new ExprPinner(dynamicDims);
    }

    public Function(string name, Expr body, ReadOnlySpan<IVar> parameters, Dictionary<Var, Dimension[]>? varMap)
        : this(name, StackVMModuleKind, body, parameters, varMap)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(Expr body, ReadOnlySpan<IVar> parameters)
        : this($"func_{_globalFuncIndex++}", body, parameters, new Dictionary<Var, Dimension[]>())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(string name, Expr body, params IVar[] parameters)
        : this(name, body, parameters.AsReadOnlySpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(Expr body, params IVar[] parameters)
        : this(body, parameters.AsReadOnlySpan())
    {
    }

    public Expr Body => Operands[0];

    public ReadOnlySpan<IVar> Parameters => SpanUtility.UnsafeCast<Expr, IVar>(Operands[1..]);

    public Dictionary<Var, Dimension[]>? VarMap { get; }

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType?> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((Expr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFunction(this, context);

    public Function With(string? name = null, string? moduleKind = null, Expr? body = null, IVar[]? parameters = null)
        => new Function(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters, VarMap);
}
