// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class Grid : Expr
{
    private readonly int _bodyParametersCount;
    private readonly int _accessMapsCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="Grid"/> class.
    /// </summary>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="domainParameter">the grid domain parameter. </param>
    /// <param name="bodyParameters">Body parameters.</param>
    /// <param name="accessMaps">Access maps.</param>
    /// <param name="buffers">output buffers.</param>
    /// <param name="reads">Reads.</param>
    /// <param name="body">The body sequence.</param>
    public Grid(string moduleKind, Var domainParameter, ReadOnlySpan<Var> bodyParameters, ReadOnlySpan<AffineMap> accessMaps, ReadOnlySpan<Expr> buffers, ReadOnlySpan<Expr> reads, Sequential body)
        : base(new Expr[] { domainParameter }.Concat(bodyParameters.ToArray()).Concat(accessMaps.ToArray()).Concat(buffers.ToArray()).Concat(reads.ToArray()).Append(body))
    {
        ModuleKind = moduleKind;
        _bodyParametersCount = bodyParameters.Length;
        _accessMapsCount = accessMaps.Length;

        if (buffers.Length != _accessMapsCount
            || buffers.Length != bodyParameters.Length)
        {
            throw new ArgumentException("Invalid buffers count.");
        }

        if (reads.Length != _accessMapsCount - 1)
        {
            throw new ArgumentException("Invalid reads count.");
        }
    }

    public string ModuleKind { get; }

    public Var DomainParameter => (Var)Operands[0];

    public ReadOnlySpan<Var> BodyParameters => SpanUtility.UnsafeCast<Expr, Var>(Operands.Slice(1, _bodyParametersCount));

    public ReadOnlySpan<AffineMap> AccessMaps => SpanUtility.UnsafeCast<Expr, AffineMap>(Operands.Slice(1 + _bodyParametersCount, _accessMapsCount));

    public ReadOnlySpan<Expr> Buffers => Operands.Slice(1 + _bodyParametersCount + _accessMapsCount, _accessMapsCount);

    public ReadOnlySpan<Expr> Reads => Operands.Slice(1 + _bodyParametersCount + (_accessMapsCount * 2), _accessMapsCount - 1);

    public Sequential Body => (Sequential)Operands[1 + _bodyParametersCount + (_accessMapsCount * 3) - 1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitGrid(this, context);

    public Grid With(string? moduleKind = null, Var? domainParameter = null, Var[]? bodyParameters = null, AffineMap[]? accessMaps = null, Expr[]? buffers = null, Expr[]? reads = null, Sequential? body = null)
        => new Grid(moduleKind ?? ModuleKind, domainParameter ?? DomainParameter, bodyParameters ?? BodyParameters, accessMaps ?? AccessMaps, buffers ?? Buffers, reads ?? Reads, body ?? Body);
}
