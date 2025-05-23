// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class DimConst : Dimension, IEquatable<DimConst?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimConst"/> class.
    /// </summary>
    /// <param name="value">Value.</param>
    public DimConst(long value)
        : base(Array.Empty<Expr>())
    {
        Value = value;
        Metadata.Range = new ValueRange<double>(value, value);
    }

    public override DimensionKind Kind => DimensionKind.Fixed;

    public override long FixedValue => Value;

    /// <summary>
    /// Gets value.
    /// </summary>
    public long Value { get; }

    public static implicit operator DimConst(long value) => new DimConst(value);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimConst(this, context);

    public DimConst With(long? value = null) => new DimConst(value ?? Value);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimConst);

    /// <inheritdoc/>
    public bool Equals(DimConst? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Value == other.Value;
    }

    public override string ToString() => Value.ToString();

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value);
}
