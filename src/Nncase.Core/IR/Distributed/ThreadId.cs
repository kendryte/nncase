// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Distributed;

public sealed class ThreadIdDim : Dimension, IEquatable<ThreadIdDim?>
{
    public static readonly ThreadIdDim Default = new();

    public ThreadIdDim()
        : base(Array.Empty<Expr>())
    {
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitThreadIdDim(this, context);

    public ThreadIdDim With() => Default;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ThreadIdDim);

    /// <inheritdoc/>
    public bool Equals(ThreadIdDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null;
    }

    public override string ToString() => "tid";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => 0;
}
