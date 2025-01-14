// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Passes;
using Nncase.Passes.Mutators;

namespace Nncase.IR
{
    /// <summary>
    /// Dimension kind.
    /// </summary>
    public enum DimensionKind : byte
    {
        /// <summary>
        /// Unknown dimension.
        /// </summary>
        Unknown,

        /// <summary>
        /// Fixed dimesnion.
        /// </summary>
        Fixed,

        /// <summary>
        /// Used for shape pattern.
        /// </summary>
        Any,
    }

    /// <summary>
    /// Shape dimension.
    /// </summary>
    public sealed class Dimension : IEquatable<Dimension>
    {
        public static readonly Dimension Any = new Dimension();

        private readonly long? _fixedValue;
        private readonly Expr? _exprValue;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dimension"/> class.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public Dimension(long value)
        {
            Kind = DimensionKind.Fixed;
            _fixedValue = value;
        }

        public Dimension(Expr value)
        {
            value = CompilerServices.SimplifyForDimension(value);
            if (value is TensorConst tc)
            {
                Kind = DimensionKind.Fixed;
                _fixedValue = tc.Value.ToScalar<int>();
            }
            else
            {
                Kind = DimensionKind.Unknown;
                _exprValue = value;
            }
        }

        private Dimension()
        {
            Kind = DimensionKind.Any;
            _exprValue = new Var("Any", DataTypes.Int64);
        }

        /// <summary>
        /// Gets kind.
        /// </summary>
        public DimensionKind Kind { get; }

        /// <summary>
        /// Gets value.
        /// </summary>
        public Expr Value => _exprValue ?? _fixedValue!.Value;

        /// <summary>
        /// Gets FixedValue.
        /// </summary>
        public long FixedValue
        {
            get => _fixedValue ??
               throw new InvalidOperationException("Only Can Get It When Shape Is Fixed !");
        }

        /// <summary>
        /// Gets a value indicating whether unknown.
        /// </summary>
        public bool IsUnknown => Kind is DimensionKind.Unknown or DimensionKind.Any;

        /// <summary>
        /// Gets a value indicating whether fixed.
        /// </summary>
        public bool IsFixed => Kind == DimensionKind.Fixed;

        public bool IsAny => Kind == DimensionKind.Any;

        /// <summary>
        /// Convert <see cref="long"/> to a fixed <see cref="Dimension"/>.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public static implicit operator Dimension(long value) => new(value);

        /// <summary>
        /// Convert <see cref="Expr"/> to a <see cref="Dimension"/> expression.
        /// </summary>
        /// <param name="value">Dimension value.</param>
        public static implicit operator Dimension(Expr value) => new(value);

        public static bool operator ==(Dimension left, Dimension right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Dimension left, Dimension right)
        {
            return !(left == right);
        }

        public static Dimension operator +(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue + rhs.FixedValue,
            (_, _) => new Dimension(lhs.Value + rhs.Value),
        };

        public static Dimension operator +(Dimension lhs, int rhs) => lhs.IsFixed ? lhs.FixedValue + rhs : new Dimension(lhs.Value + rhs);

        public static Dimension operator -(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue - rhs.FixedValue,
            (_, _) => new Dimension(lhs.Value - rhs.Value),
        };

        public static Dimension operator *(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue * rhs.FixedValue,
            (_, _) => new Dimension(lhs.Value * rhs.Value),
        };

        public static Dimension operator /(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
        {
            (true, true) => lhs.FixedValue / rhs.FixedValue,
            (_, _) => new Dimension(lhs.Value / rhs.Value),
        };

        public static Dimension Abs(Dimension value)
        {
            if (value.IsFixed)
            {
                return System.Math.Abs(value.FixedValue);
            }

            return IR.F.Math.Abs(value.Value);
        }

        public static Dimension Clamp(Dimension value, Dimension min, Dimension max)
        {
            if (value.IsFixed && min.IsFixed && max.IsFixed)
            {
                return System.Math.Clamp(value.FixedValue, min.FixedValue, max.FixedValue);
            }

            return IR.F.Math.Clamp(value.Value, min.Value, max.Value);
        }

        public static Dimension CeilDiv(Dimension lhs, Dimension rhs)
        {
            if (lhs.IsFixed && rhs.IsFixed)
            {
                return (lhs.FixedValue + rhs.FixedValue - 1) / rhs.FixedValue;
            }

            return IR.F.Math.CeilDiv(lhs.Value, rhs.Value);
        }

        // public static Dimension Unknown(string? name = null) => new Dimension(name is null ? new Var(DataTypes.Int64) : new Var(name, DataTypes.Int64));

        public static Dimension Unknown(string? name = null) => Any;

        /// <inheritdoc/>
        public override string ToString()
        {
            return Value?.ToString() ?? "?";
        }

        /// <inheritdoc/>
        public override bool Equals(object? obj)
        {
            return obj is Dimension dimension && Equals(dimension);
        }

        /// <inheritdoc/>
        public bool Equals(Dimension? other)
        {
            return other is not null && (Kind, other.Kind) switch
            {
                (DimensionKind.Any, DimensionKind.Any) => true,
                (DimensionKind.Unknown, DimensionKind.Unknown) => Value == other.Value,
                (DimensionKind.Fixed, DimensionKind.Fixed) => FixedValue == other.FixedValue,
                (_, _) => false,
            };
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return IsFixed ? HashCode.Combine(Kind, FixedValue) : HashCode.Combine(Kind, Value);
        }

        public bool HasFixedValue(Predicate<long> predicate)
        {
            return IsFixed && predicate(FixedValue);
        }

        public bool IsAssignableFrom(Dimension dimension)
        {
            if (IsUnknown)
            {
                return true;
            }

            return dimension.Kind == DimensionKind.Fixed && Value == dimension.Value;
        }
    }
}
