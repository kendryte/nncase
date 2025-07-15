// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Nncase.IR.Shapes;

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
            (DimProduct dimProductA, DimProduct dimProductB) => SimplifyProductRemainder(dimProductA, dimProductB),
            (OpaqueDim opaqueDim, DimProduct dimProduct) => SimplifyProductRemainder(new DimProduct([opaqueDim]), dimProduct),
            (DimProduct dimProduct, OpaqueDim opaqueDim) => SimplifyProductRemainder(dimProduct, new DimProduct([opaqueDim])),
            (_, _) when numerator == denominator => DimConst.Zero,
            (_, _) => new DimRemainder(numerator, denominator),
        };
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Numerator, Denominator);

    private static Dimension SimplifyProductRemainder(DimProduct numerator, DimProduct denominator)
    {
        bool simplified = false;
        var (numScale, numPows) = DimHelpers.GetScaleAndPows(numerator);
        var (denScale, denPows) = DimHelpers.GetScaleAndPows(denominator);

        if (numScale % denScale == 0)
        {
            // If the scale of the numerator is divisible by the scale of the denominator,
            // we can simplify the scale.
            numScale /= denScale;
            denScale = 1;
            simplified = true;
        }

        var newDenPows = new Dictionary<Dimension, int>(ReferenceEqualityComparer.Instance);
        foreach (var (dim, pow) in denPows)
        {
            ref var numPow = ref CollectionsMarshal.GetValueRefOrNullRef(numPows, dim);
            if (Unsafe.IsNullRef(ref numPow))
            {
                // If the dimension is not in the numerator, we need to add it to the denominator.
                newDenPows.Add(dim, pow);
            }
            else
            {
                // If the dimension is in both, we subtract the powers.
                if (numPow >= pow)
                {
                    numPows.Remove(dim);
                    if (numPow > pow)
                    {
                        newDenPows.Add(dim, numPow - pow);
                    }
                }
                else
                {
                    numPows[dim] = numPow - pow;
                }

                simplified = true;
            }
        }

        var newNumerator = DimHelpers.Simplify(numScale, numPows);
        var newDenominator = DimHelpers.Simplify(denScale, newDenPows);
        return simplified ? newNumerator % newDenominator : new DimRemainder(newNumerator, newDenominator);
    }

    private ValueRange<double> InferRange()
    {
        if (Numerator.Metadata.Range is ValueRange<double> numRange && Denominator.Metadata.Range is ValueRange<double> denRange)
        {
            var ranges = new[] {
                (long)(numRange.Min % denRange.Min),
                (long)(numRange.Min % denRange.Max),
                (long)(numRange.Max % denRange.Min),
                (long)(numRange.Max % denRange.Max),
            };
            return new(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}
