// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class AsDim : Dimension, IEquatable<AsDim?>
{
    public AsDim(Expr dim)
        : base([dim])
    {
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets dim.
    /// </summary>
    public Expr Dim => (Expr)Operands[0];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitAsDim(this, context);

    public AsDim With(Expr? dim = null) => new AsDim(dim ?? Dim);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as AsDim);

    /// <inheritdoc/>
    public bool Equals(AsDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Dim.Equals(other.Dim);
    }

    public override Dimension Simplify()
    {
        if (Dim is TensorConst tc)
        {
            return new DimConst(tc.Value.ToScalar<long>());
        }

        return this;
    }

    public override string ToString() => $"as({Dim})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => Dim.GetHashCode();
}
