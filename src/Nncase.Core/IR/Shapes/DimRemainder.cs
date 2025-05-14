// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class DimRemainder : Dimension, IEquatable<DimRemainder?>
{
    public DimRemainder(Dimension numerator, Dimension denominator)
        : base([numerator, denominator])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets numerator.
    /// </summary>
    public Dimension Numerator => (Dimension)Operands[0];

    /// <summary>
    /// Gets denominator.
    /// </summary>
    public Dimension Denominator => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimRemainder(this, context);

    public Dimension With(Dimension? numerator = null, Dimension? denominator = null)
    {
        if (numerator is null && denominator is null)
        {
            return new DimRemainder(Numerator, Denominator);
        }

        return Simplify(numerator ?? Numerator, denominator ?? Denominator);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimRemainder);

    /// <inheritdoc/>
    public bool Equals(DimRemainder? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Numerator.Equals(other.Numerator) && Denominator.Equals(other.Denominator);
    }

    public override string ToString() => $"({Numerator} % {Denominator})";

    public override Dimension Simplify() => Simplify(Numerator, Denominator);

    internal static Dimension Simplify(Dimension numerator, Dimension denominator)
    {
        return (numerator, denominator) switch
        {
            (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value % rhsConst.Value,
            (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
            (_, DimConst dimConst) when dimConst.Value == 1 => DimConst.Zero,
            (_, _) when numerator.IsUnknown || denominator.IsUnknown => Unknown,
            (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => DimConst.Zero,
            (_, _) => new DimRemainder(numerator, denominator),
        };
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Numerator, Denominator);

    private ValueRange<double> InferRange()
    {
        if (Numerator.Metadata.Range is ValueRange<double> numRange && Denominator.Metadata.Range is ValueRange<double> denRange)
        {
            var ranges = new[] {
                (long)numRange.Min % (long)denRange.Min,
                (long)numRange.Min % (long)denRange.Max,
                (long)numRange.Max % (long)denRange.Min,
                (long)numRange.Max % (long)denRange.Max,
            };
            return new(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}
