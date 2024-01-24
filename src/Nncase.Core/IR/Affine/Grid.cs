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
    /// <param name="bodyParameters">Body parameters.</param>
    /// <param name="accessMaps">Access maps.</param>
    /// <param name="buffers">Buffers.</param>
    /// <param name="reads">Reads.</param>
    /// <param name="body">The body sequence.</param>
    public Grid(ReadOnlySpan<Var> bodyParameters, ReadOnlySpan<AffineMap> accessMaps, ReadOnlySpan<Expr> buffers, ReadOnlySpan<Expr> reads, Sequential body)
        : base(bodyParameters.ToArray().AsEnumerable<Expr>().Concat(accessMaps.ToArray()).Concat(buffers.ToArray()).Concat(reads.ToArray()).Append(body))
    {
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

    public ReadOnlySpan<Var> BodyParameters => SpanUtility.UnsafeCast<Expr, Var>(Operands.Slice(0, _bodyParametersCount));

    public ReadOnlySpan<AffineMap> AccessMaps => SpanUtility.UnsafeCast<Expr, AffineMap>(Operands.Slice(_bodyParametersCount, _accessMapsCount));

    public ReadOnlySpan<Expr> Buffers => Operands.Slice(_bodyParametersCount + _accessMapsCount, _accessMapsCount);

    public ReadOnlySpan<Expr> Reads => Operands.Slice(_bodyParametersCount + (_accessMapsCount * 2), _accessMapsCount - 1);

    public Sequential Body => (Sequential)Operands[_bodyParametersCount + (_accessMapsCount * 3) - 1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitGrid(this, context);

    public Grid With(Var[]? loopVars = null, AffineMap[]? accessMaps = null, TIR.Buffer[]? buffers = null, Expr[]? reads = null, Sequential? body = null)
        => new Grid(loopVars ?? BodyParameters, accessMaps ?? AccessMaps, buffers ?? Buffers, reads ?? Reads, body ?? Body);
}
