// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR;

public sealed class DimPower : Dimension, IEquatable<DimPower?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimPower"/> class.
    /// </summary>
    /// <param name="dim">Dim.</param>
    /// <param name="power">Power.</param>
    public DimPower(OpaqueDim dim, int power)
        : base([dim])
    {
        Power = power;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets dim.
    /// </summary>
    public OpaqueDim Dim => (OpaqueDim)Operands[0];

    public int Power { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimPower(this, context);

    public DimPower With(OpaqueDim? dim = null, int? power = null) => new DimPower(dim ?? Dim, power ?? Power);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimPower);

    /// <inheritdoc/>
    public bool Equals(DimPower? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Dim.Equals(other.Dim) && Power == other.Power;
    }

    /// <inheritdoc/>
    public override string ToString() =>
        Power switch
        {
            1 => Dim.ToString(),
            _ => $"{Dim}^{Power}",
        };

    public override Dimension Simplify() =>
        Power switch
        {
            0 => DimConst.One,
            1 => Dim,
            _ => this,
        };

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Dim, Power);

    private ValueRange<double> InferRange()
    {
        if (Dim.Metadata.Range is ValueRange<double> dimRange)
        {
            var ranges = new[] {
                System.Math.Pow(dimRange.Min, Power),
                System.Math.Pow(dimRange.Max, Power),
            };
            return new(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}
