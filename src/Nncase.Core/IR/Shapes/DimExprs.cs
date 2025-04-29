// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.IR.Shapes;
using Nncase.Utilities;

namespace Nncase.IR;

public enum DimDivideMode
{
    FloorDiv,
    CeilDiv,
}

public abstract class OpaqueDim : Dimension
{
    protected OpaqueDim(BaseExpr[] operands)
        : base(operands)
    {
    }
}

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

    public override string ToString() => $"as({Dim})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => Dim.GetHashCode();
}

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

public sealed class DimVar : OpaqueDim, IVar, IEquatable<DimVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="DimVar"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public DimVar(string name)
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = name;
        Metadata.Range = ValueRange<double>.Full;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DimVar"/> class.
    /// </summary>
    public DimVar()
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = $"dimVar_{GlobalVarIndex}";
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Create a dim var.
    /// </summary>
    public static implicit operator DimVar(string name) => new DimVar(name);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimVar(this, context);

    public DimVar With(string? name = null) => new DimVar(name ?? Name)
    {
        Metadata =
        {
            Range = Metadata.Range,
        },
    };

    IVar IVar.With(string? name) => With(name);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimVar);

    /// <inheritdoc/>
    public bool Equals(DimVar? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    bool IEquatable<IVar?>.Equals(IVar? other) => Equals(other as DimVar);

    public override string ToString() => $"{Name}";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }
}

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

    public DimFraction With(DimDivideMode? divMode = null, Dimension? numerator = null, Dimension? denominator = null) =>
        new DimFraction(divMode ?? DivMode, numerator ?? Numerator, denominator ?? Denominator);

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

    public override string ToString() => $"({Numerator} / {Denominator})";

    public override Dimension Simplify()
    {
        var (numeratorScale, numeratorPows) = DimHelpers.GetScaleAndPows(Numerator);
        var (denominatorScale, denominatorPows) = DimHelpers.GetScaleAndPows(Denominator);
        if (numeratorScale % denominatorScale == 0)
        {
            numeratorScale /= denominatorScale;
            denominatorScale = 1;
        }

        foreach (var (denominator, denominatorPow) in denominatorPows.ToArray())
        {
            ref var numeratorPow = ref CollectionsMarshal.GetValueRefOrNullRef(numeratorPows, denominator);
            if (!Unsafe.IsNullRef(ref numeratorPow))
            {
                if (numeratorPow > denominatorPow)
                {
                    denominatorPows.Remove(denominator);
                    numeratorPow -= denominatorPow;
                }
                else if (numeratorPow == denominatorPow)
                {
                    numeratorPows.Remove(denominator);
                    denominatorPows.Remove(denominator);
                }
                else
                {
                    numeratorPows.Remove(denominator);
                    denominatorPows[denominator] -= numeratorPow;
                }
            }
        }

        var newNumerator = DimHelpers.Simplify(numeratorScale, numeratorPows);
        var newDenominator = DimHelpers.Simplify(denominatorScale, denominatorPows);
        return (newNumerator, newDenominator) switch
        {
            (DimConst numConst, DimConst denConst) => DivMode == DimDivideMode.FloorDiv
                ? numConst.Value / denConst.Value
                : MathUtility.CeilDiv(numConst.Value, denConst.Value),
            (DimConst numConst, _) when numConst.Value == 0 => Zero,
            (_, DimConst denConst) when denConst.Value == 1 => newNumerator,
            _ => new DimFraction(DivMode, newNumerator, newDenominator),
        };
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Numerator, Denominator);

    private ValueRange<double> InferRange()
    {
        if (Numerator.Metadata.Range is ValueRange<double> numRange && Denominator.Metadata.Range is ValueRange<double> denRange)
        {
            long[] ranges = DivMode == DimDivideMode.FloorDiv ? [
                (long)numRange.Min / (long)denRange.Min,
                (long)numRange.Min / (long)denRange.Max,
                (long)numRange.Max / (long)denRange.Min,
                (long)numRange.Max / (long)denRange.Max,
            ] : [
                MathUtility.CeilDiv((long)numRange.Min, (long)denRange.Min),
                MathUtility.CeilDiv((long)numRange.Min, (long)denRange.Max),
                MathUtility.CeilDiv((long)numRange.Max, (long)denRange.Min),
                MathUtility.CeilDiv((long)numRange.Max, (long)denRange.Max),
            ];
            return new(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}

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

    public DimRemainder With(Dimension? numerator = null, Dimension? denominator = null) =>
        new DimRemainder(numerator ?? Numerator, denominator ?? Denominator);

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

public sealed class DimProduct : Dimension, IEquatable<DimProduct?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimProduct"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <param name="scale">Scale.</param>
    public DimProduct(Dimension[] operands, long scale = 1)
        : base(operands)
    {
        Scale = scale;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    public long Scale { get; }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public int Count => Operands.Length;

    public Dimension this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimProduct(this, context);

    public DimProduct With(Dimension[]? operands = null, long? scale = null) => new DimProduct(operands ?? Operands.ToArray(), scale ?? Scale);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimProduct);

    /// <inheritdoc/>
    public bool Equals(DimProduct? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Scale == other.Scale && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString()
    {
        var scale = Scale == 1 ? string.Empty : $"{Scale} * ";
        return $"({scale}{StringUtility.Join(" * ", Operands)})";
    }

    public override Dimension Simplify()
    {
        (var scale, var pows) = DimHelpers.GetScaleAndPows(this);
        return DimHelpers.Simplify(scale, pows);
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        hash.Add(Scale);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var operands = Operands.ToArray().Append(Scale);
        var allMinMax = from lhs in operands
                        from rhs in operands
                        where !ReferenceEquals(lhs, rhs)
                        let ranges = new[] {
                         lhs.Metadata.Range!.Value.Min * rhs.Metadata.Range!.Value.Min,
                         lhs.Metadata.Range!.Value.Min * rhs.Metadata.Range!.Value.Max,
                         lhs.Metadata.Range!.Value.Max * rhs.Metadata.Range!.Value.Min,
                         lhs.Metadata.Range!.Value.Max * rhs.Metadata.Range!.Value.Max,
                     }
                        select new ValueRange<double>(ranges.Min(), ranges.Max());
        var min = allMinMax.MinBy(x => x.Min).Min;
        var max = allMinMax.MaxBy(x => x.Max).Max;
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimSum : Dimension, IEquatable<DimSum?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimSum"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <param name="bias">Bias.</param>
    public DimSum(Dimension[] operands, long bias = 0)
        : base(operands)
    {
        Bias = bias;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public long Bias { get; }

    public int Count => Operands.Length;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimSum(this, context);

    public DimSum With(Dimension[]? operands = null, long? bias = null) => new DimSum(operands ?? Operands.ToArray(), bias ?? Bias);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimSum);

    /// <inheritdoc/>
    public bool Equals(DimSum? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString()
    {
        var bias = Bias == 1 ? string.Empty : $"{Bias} * ";
        return $"({bias}{StringUtility.Join(" + ", Operands)})";
    }

    public override Dimension Simplify()
    {
        long bias = Bias;
        var scales = new Dictionary<Dimension, long>(ReferenceEqualityComparer.Instance);

        foreach (var operand in Operands)
        {
            if (operand is DimConst dimConst)
            {
                bias += dimConst.Value;
            }
            else
            {
                ref var value = ref CollectionsMarshal.GetValueRefOrAddDefault(scales, operand, out _);
                value += 1;
            }
        }

        var newOperands = scales.Select(kvp => kvp.Key * kvp.Value).ToArray();
        return (bias, newOperands.Length) switch
        {
            (_, 0) => new DimConst(bias),
            (0, 1) => newOperands[0],
            _ => new DimSum(newOperands, bias),
        };
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var min = (double)Bias;
        var max = (double)Bias;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range!.Value;
            min = System.Math.Min(min, range.Min);
            max = System.Math.Max(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimAbs : Dimension, IEquatable<DimAbs?>
{
    public DimAbs(Dimension operand)
        : base([operand])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimAbs(this, context);

    public DimAbs With(Dimension? operand = null) => new DimAbs(operand ?? Operand);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimAbs);

    /// <inheritdoc/>
    public bool Equals(DimAbs? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand);
    }

    public override string ToString() => $"|{Operand}|";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => Operand.GetHashCode();

    private ValueRange<double> InferRange()
    {
        if (Operand.Metadata.Range is ValueRange<double> operandRange)
        {
            var ranges = new[] {
                System.Math.Abs(operandRange.Min),
                System.Math.Abs(operandRange.Max),
            };
            return new ValueRange<double>(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}

public sealed class DimClamp : OpaqueDim, IEquatable<DimClamp?>
{
    public DimClamp(Dimension operand, Dimension minValue, Dimension maxValue)
        : base([operand, minValue, maxValue])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    /// <summary>
    /// Gets min.
    /// </summary>
    public Dimension MinValue => (Dimension)Operands[1];

    /// <summary>
    /// Gets max.
    /// </summary>
    public Dimension MaxValue => (Dimension)Operands[2];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimClamp(this, context);

    public DimClamp With(Dimension? operand = null, Dimension? minValue = null, Dimension? maxValue = null) =>
        new DimClamp(operand ?? Operand, minValue ?? MinValue, maxValue ?? MaxValue);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimClamp);

    /// <inheritdoc/>
    public bool Equals(DimClamp? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand) && MinValue.Equals(other.MinValue) && MaxValue.Equals(other.MaxValue);
    }

    public override string ToString() => $"clamp({Operand}, {MinValue}, {MaxValue})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Operand, MinValue, MaxValue);

    private ValueRange<double> InferRange()
    {
        var operandRange = Operand.Metadata.Range!.Value;
        var min = System.Math.Max(operandRange.Min, MinValue.Metadata.Range!.Value.Min);
        var max = System.Math.Min(operandRange.Max, MaxValue.Metadata.Range!.Value.Max);
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimCompareAndSelect : OpaqueDim, IEquatable<DimCompareAndSelect?>
{
    public DimCompareAndSelect(Dimension value, Dimension expected, Dimension trueValue, Dimension falseValue)
        : base([value, expected, trueValue, falseValue])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets value.
    /// </summary>
    public Dimension Value => (Dimension)Operands[0];

    /// <summary>
    /// Gets expected.
    /// </summary>
    public Dimension Expected => (Dimension)Operands[1];

    /// <summary>
    /// Gets true value.
    /// </summary>
    public Dimension TrueValue => (Dimension)Operands[2];

    /// <summary>
    /// Gets false value.
    /// </summary>
    public Dimension FalseValue => (Dimension)Operands[3];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimCompareAndSelect(this, context);

    public DimCompareAndSelect With(Dimension? value = null, Dimension? expected = null, Dimension? trueValue = null, Dimension? falseValue = null) =>
        new DimCompareAndSelect(value ?? Value, expected ?? Expected, trueValue ?? TrueValue, falseValue ?? FalseValue);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimCompareAndSelect);

    /// <inheritdoc/>
    public bool Equals(DimCompareAndSelect? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Value.Equals(other.Value) && Expected.Equals(other.Expected) && TrueValue.Equals(other.TrueValue) && FalseValue.Equals(other.FalseValue);
    }

    public override string ToString() => $"({Value} == {Expected} ? {TrueValue} : {FalseValue})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value, Expected, TrueValue, FalseValue);

    private ValueRange<double> InferRange()
    {
        var trueValueRange = TrueValue.Metadata.Range!.Value;
        var falseValueRange = FalseValue.Metadata.Range!.Value;
        var min = System.Math.Min(trueValueRange.Min, falseValueRange.Min);
        var max = System.Math.Max(trueValueRange.Max, falseValueRange.Max);
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimMin : OpaqueDim, IEquatable<DimMin?>
{
    public DimMin(params Dimension[] operands)
        : base(operands)
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimMin(this, context);

    public DimMin With(Dimension[]? operands = null) => new DimMin(operands ?? Operands.ToArray());

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimMin);

    /// <inheritdoc/>
    public bool Equals(DimMin? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString() => $"min({StringUtility.Join(", ", Operands)})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var min = double.MaxValue;
        var max = double.MinValue;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range!.Value;
            min = System.Math.Min(min, range.Min);
            max = System.Math.Min(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimMax : OpaqueDim, IEquatable<DimMax?>
{
    public DimMax(params Dimension[] operands)
        : base(operands)
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimMax(this, context);

    public DimMax With(Dimension[]? operands = null) => new DimMax(operands ?? Operands.ToArray());

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimMax);

    /// <inheritdoc/>
    public bool Equals(DimMax? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString() => $"max({StringUtility.Join(", ", Operands)})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var min = double.MaxValue;
        var max = double.MinValue;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range!.Value;
            min = System.Math.Max(min, range.Min);
            max = System.Math.Max(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimPositive : OpaqueDim, IEquatable<DimPositive?>
{
    public DimPositive(Dimension operand, Dimension extent)
        : base([operand, extent])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    /// <summary>
    /// Gets extent.
    /// </summary>
    public Dimension Extent => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimPositive(this, context);

    public DimPositive With(Dimension? operand = null, Dimension? extent = null) => new DimPositive(operand ?? Operand, extent ?? Extent);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimPositive);

    /// <inheritdoc/>
    public bool Equals(DimPositive? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand) && Extent.Equals(other.Extent);
    }

    public override string ToString() => $"positive({Operand}, {Extent})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Operand, Extent);

    private ValueRange<double> InferRange()
    {
        if (Operand.Metadata.Range is ValueRange<double> operandRange && Extent.Metadata.Range is ValueRange<double> extentRange)
        {
            var min = operandRange.Min < 0 ? 0 : operandRange.Min;
            var max = operandRange.Max >= 0 ? System.Math.Min(operandRange.Max, extentRange.Max) : extentRange.Max;
            return new ValueRange<double>(min, max);
        }

        return ValueRange<double>.Full;
    }
}

public sealed class DimAt : OpaqueDim, IEquatable<DimAt?>
{
    public DimAt(Shape shape, Dimension index)
        : base([shape, index])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets shape.
    /// </summary>
    public Shape Shape => (Shape)Operands[0];

    /// <summary>
    /// Gets index.
    /// </summary>
    public Dimension Index => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimAt(this, context);

    public DimAt With(Shape? shape = null, Dimension? index = null) => new DimAt(shape ?? Shape, index ?? Index);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimAt);

    /// <inheritdoc/>
    public bool Equals(DimAt? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Shape.Equals(other.Shape) && Index.Equals(other.Index);
    }

    public override string ToString() => $"at({Shape}, {Index})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Shape, Index);

    private ValueRange<double> InferRange()
    {
        if (Shape.Metadata.Range is ValueRange<double> shapeRange)
        {
            return shapeRange;
        }

        return ValueRange<double>.Full;
    }
}
