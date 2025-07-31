// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Nncase.IR.Shapes;
using Nncase.Utilities;

namespace Nncase.IR;

public enum DimDivideMode
{
    FloorDiv,
    CeilDiv,
}

public sealed class DimFraction : Dimension, IEquatable<DimFraction?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimFraction"/> class.
    /// </summary>
    /// <param name="divMode">Division mode.</param>
    /// <param name="numerator">Numerator.</param>
    /// <param name="denominator">Denominator.</param>
    public DimFraction(DimDivideMode divMode, Dimension numerator, Dimension denominator)
        : base([numerator, denominator])
    {
        DivMode = divMode;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    public DimDivideMode DivMode { get; }

    /// <summary>
    /// Gets numerator.
    /// </summary>
    public Dimension Numerator => (Dimension)Operands[0];

    /// <summary>
    /// Gets denominator.
    /// </summary>
    public Dimension Denominator => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimFraction(this, context);

    public Dimension With(DimDivideMode? divMode = null, Dimension? numerator = null, Dimension? denominator = null)
    {
        if (divMode is null && numerator is null && denominator is null)
        {
            return new DimFraction(DivMode, Numerator, Denominator);
        }

        return Simplify(divMode ?? DivMode, numerator ?? Numerator, denominator ?? Denominator);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimFraction);

    /// <inheritdoc/>
    public bool Equals(DimFraction? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Numerator.Equals(other.Numerator) && Denominator.Equals(other.Denominator);
    }

    public override string ToString() => DivMode switch
    {
        DimDivideMode.FloorDiv => $"floor({Numerator} / {Denominator})",
        DimDivideMode.CeilDiv => $"ceil({Numerator} / {Denominator})",
        _ => throw new NotSupportedException($"Unsupported division mode {DivMode}"),
    };

    public override Dimension Simplify() => Simplify(DivMode, Numerator, Denominator);

    internal static Dimension Simplify(DimDivideMode divMode, Dimension numerator, Dimension denominator)
    {
        if (divMode == DimDivideMode.FloorDiv)
        {
            return (numerator, denominator) switch
            {
                (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value / rhsConst.Value,
                (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
                (_, DimConst dimConst) when dimConst.Value == 1 => numerator,
                (_, _) when numerator.IsUnknown || denominator.IsUnknown => Unknown,
                (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => dimProduct.With(scale: dimProduct.Scale / dimConst.Value),
                _ => CreateWithSimplify(divMode, numerator, denominator),
            };
        }
        else if (divMode == DimDivideMode.CeilDiv)
        {
            return (numerator, denominator) switch
            {
                (DimConst lhsConst, DimConst rhsConst) => MathUtility.CeilDiv(lhsConst.Value, rhsConst.Value),
                (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
                (_, DimConst dimConst) when dimConst.Value == 1 => numerator,
                (_, _) when numerator.IsUnknown || denominator.IsUnknown => Unknown,
                (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => dimProduct.With(scale: dimProduct.Scale / dimConst.Value),
                (_, _) => CreateWithSimplify(DimDivideMode.CeilDiv, numerator, denominator),
            };
        }
        else
        {
            throw new NotSupportedException($"Unsupported division mode {divMode}");
        }
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Numerator, Denominator);

    private static Dimension CreateWithSimplify(DimDivideMode divMode, Dimension numerator, Dimension denominator)
    {
        var (numeratorScale, numeratorPows) = DimHelpers.GetScaleAndPows(numerator);
        var (denominatorScale, denominatorPows) = DimHelpers.GetScaleAndPows(denominator);
        if (numeratorScale % denominatorScale == 0)
        {
            numeratorScale /= denominatorScale;
            denominatorScale = 1;
        }

        foreach (var (denominatorElem, denominatorElemPow) in denominatorPows.ToArray())
        {
            ref var numeratorPow = ref CollectionsMarshal.GetValueRefOrNullRef(numeratorPows, denominatorElem);
            if (!Unsafe.IsNullRef(ref numeratorPow))
            {
                if (numeratorPow > denominatorElemPow)
                {
                    denominatorPows.Remove(denominatorElem);
                    numeratorPow -= denominatorElemPow;
                }
                else if (numeratorPow == denominatorElemPow)
                {
                    numeratorPows.Remove(denominatorElem);
                    denominatorPows.Remove(denominatorElem);
                }
                else
                {
                    numeratorPows.Remove(denominatorElem);
                    denominatorPows[denominatorElem] -= numeratorPow;
                }
            }
        }

        var newNumerator = DimHelpers.Simplify(numeratorScale, numeratorPows);
        var newDenominator = DimHelpers.Simplify(denominatorScale, denominatorPows);
        return (newNumerator, newDenominator) switch
        {
            (DimConst numConst, DimConst denConst) => divMode == DimDivideMode.FloorDiv
                ? numConst.Value / denConst.Value
                : MathUtility.CeilDiv(numConst.Value, denConst.Value),
            (DimConst numConst, _) when numConst.Value == 0 => Zero,
            (_, DimConst denConst) when denConst.Value == 1 => newNumerator,
            _ => new DimFraction(divMode, newNumerator, newDenominator),
        };
    }

    private ValueRange<double> InferRange()
    {
        if (Numerator.Metadata.Range is ValueRange<double> numRange && Denominator.Metadata.Range is ValueRange<double> denRange)
        {
            double[] ranges = DivMode == DimDivideMode.FloorDiv ? [
                System.Math.Floor(numRange.Min / denRange.Min),
                System.Math.Floor(numRange.Min / denRange.Max),
                System.Math.Floor(numRange.Max / denRange.Min),
                System.Math.Floor(numRange.Max / denRange.Max),
            ] : [
                System.Math.Ceiling(numRange.Min / denRange.Min),
                System.Math.Ceiling(numRange.Min / denRange.Max),
                System.Math.Ceiling(numRange.Max / denRange.Min),
                System.Math.Ceiling(numRange.Max / denRange.Max),
            ];
            return new(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}
