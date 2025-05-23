// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class UnknownDim : Dimension, IEquatable<UnknownDim?>
{
    public static readonly UnknownDim Default = new();

    public UnknownDim()
        : base(Array.Empty<Expr>())
    {
    }

    public override DimensionKind Kind => DimensionKind.Unknown;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitUnknownDim(this, context);

    public UnknownDim With() => Default;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as UnknownDim);

    /// <inheritdoc/>
    public bool Equals(UnknownDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null;
    }

    public override string ToString() => "?";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => 0;
}
