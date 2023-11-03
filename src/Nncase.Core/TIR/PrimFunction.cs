// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// PrimFunction expression.
/// </summary>
public sealed class PrimFunction : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="parameters">Arguments.</param>
    /// <param name="body">Body.</param>
    w public PrimFunction(string name, string moduleKind, Sequential body, ReadOnlySpan<Buffer> parameters)
            : base(name, moduleKind, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<Buffer, Expr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="parameters">Arguments.</param>
    /// <param name="body">Body.</param>
    public PrimFunction(string moduleKind, Sequential body, ReadOnlySpan<Buffer> parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// build function.
    /// </summary>
    public PrimFunction(string moduleKind, Sequential body, params Buffer[] parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, new(parameters))
    {
    }

    /// <summary>
    /// Gets body.
    /// </summary>
    public Sequential Body => (Sequential)Operands[0];

    public ReadOnlySpan<Buffer> Parameters => SpanUtility.UnsafeCast<Expr, Buffer>(Operands.Slice(1));

    public override IEnumerable<IRType?> ParameterTypes => Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPrimFunction(this, context);

    public PrimFunction With(string? name = null, string? moduleKind = null, Sequential? body = null, Buffer[]? parameters = null, Schedule.SchedFunctionResult? sched = null)
        => new PrimFunction(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters)
        {
            // note maybe add SchedResult into ctor.
            SchedResult = sched ?? SchedResult,
        };
}
