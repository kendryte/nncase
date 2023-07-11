// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using static NetFabric.Hyperlinq.ArrayExtensions;

namespace Nncase.Evaluator;

public static class MetricFactorNames
{
    public static readonly string FLOPs = "FLOPs";

    public static readonly string OnChipMemoryTraffic = "OnChipMemoryTraffic";

    public static readonly string OffChipMemoryTraffic = "OffChipMemoryTraffic";

    public static readonly string Parallel = "Parallel";
}

/// <summary>
/// Metric.
/// </summary>
public sealed record Metric : IEquatable<Metric>
{
    /// <summary>
    /// Zero Metric.
    /// </summary>
    public static readonly Metric Zero = new();

    /// <summary>
    /// Gets or sets factors.
    /// </summary>
    public Dictionary<string, UInt128> Factors { get; set; } = new();

    public UInt128 this[string name]
    {
        get => Factors[name];
        set => Factors[name] = value;
    }

    /// <summary>
    /// Add two Metric.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="rhs">Rhs.</param>
    /// <returns>Added result.</returns>
    public static Metric operator +(Metric lhs, Metric rhs)
    {
        var newMetric = new Metric() with { Factors = new(lhs.Factors) };
        foreach (var factor in rhs.Factors)
        {
            if (newMetric.Factors.TryGetValue(factor.Key, out var oldValue))
            {
                newMetric.Factors[factor.Key] = oldValue + factor.Value;
            }
            else
            {
                newMetric.Factors.Add(factor.Key, factor.Value);
            }
        }

        return newMetric;
    }

    /// <inheritdoc/>
    public bool Equals(Metric? other)
    {
        if (other == null)
        {
            return false;
        }

        if (Factors.Count != other.Factors.Count)
        {
            return false;
        }

        foreach (var factor in Factors)
        {
            if (other.Factors.TryGetValue(factor.Key, out var otherValue))
            {
                if (factor.Value != otherValue)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    public override int GetHashCode()
    {
        return Factors.GetHashCode();
    }

    public override string ToString()
    {
        if (Equals(Metric.Zero))
        {
            return "Zero";
        }

        return $"{{ {string.Join(", ", Factors.Select(kv => $"{kv.Key}: {kv.Value}"))}}}";
    }
}

/// <summary>
/// Metric extensions.
/// </summary>
public static class MetricExtensions
{
    /// <summary>
    /// Sum all Metrics.
    /// </summary>
    /// <param name="metrics">Source.</param>
    /// <returns>Result.</returns>
    public static Metric Sum(this IEnumerable<Metric> metrics)
    {
        return metrics.Aggregate(Metric.Zero, (x, y) => x + y)!;
    }

    /// <summary>
    /// Sum all Metrics.
    /// </summary>
    /// <param name="metrics">Source.</param>
    /// <returns>Result.</returns>
    public static Metric Sum<TSource, TSelector>(this in SpanSelectEnumerable<TSource, Metric, TSelector> metrics)
            where TSelector : struct, IFunction<TSource, Metric>
    {
        var sum = Metric.Zero;
        foreach (var metric in metrics)
        {
            sum += metric;
        }

        return sum;
    }
}

public static class MetricUtility
{
    public static UInt128 ATanhFLOPs => LogFLOPs + (2 * AddFLOPs) + DivFLOPs + MulFLOPs;

    public static UInt128 AddFLOPs => 1;

    public static UInt128 SubFLOPs => 1;

    public static UInt128 CmpFLOPs => 1;

    public static UInt128 MulFLOPs => 1;

    public static UInt128 DivFLOPs => 4;

    /// <summary>
    /// Gets ref from https://github.com/reyoung/avx_mathfun.
    /// </summary>
    public static UInt128 ExpFLOPs => 32;

    public static UInt128 PowFLOPs => ExpFLOPs;

    public static UInt128 LogFLOPs => 43;

    public static UInt128 SinFLOPs => 39;

    public static UInt128 CosFLOPs => 39;

    public static UInt128 SqrtFLOPs => 24;

    public static UInt128 SigmoidFLOPs => DivFLOPs + 2 + ExpFLOPs;

    public static UInt128 TanhFLOPs => ExpFLOPs + (2 * AddFLOPs) + DivFLOPs;

    public static UInt128 ATanFLOPs => SqrtFLOPs + (3 * MulFLOPs) + MulFLOPs + DivFLOPs;

    public static UInt128 ResizeLinearFLOPs => 4;

    public static UInt128 ResizeCubicFLOPs => 8;

    public static UInt128 GetFLOPs(IRType type, int scale = 1)
    {
        return type switch
        {
            TensorType t => (UInt128)t.Shape.Aggregate(scale, (acc, x) => acc * (x.IsFixed ? x.FixedValue : 1)),
            TupleType t => t.Fields.Sum(f => GetFLOPs(f, scale)),
            _ => 0,
        };
    }

    public static UInt128 GetUnaryFLOPs(UnaryOp op) => op switch
    {
        UnaryOp.Abs => 1,
        UnaryOp.Acosh or UnaryOp.Acos or UnaryOp.Cos or
        UnaryOp.Cosh => CosFLOPs,
        UnaryOp.Asin or UnaryOp.Asinh or UnaryOp.Sin or UnaryOp.Sinh => SinFLOPs,
        UnaryOp.Sign or UnaryOp.Round or UnaryOp.Neg or UnaryOp.Floor or UnaryOp.Ceil => 1,
        UnaryOp.Exp => ExpFLOPs,
        UnaryOp.Log => LogFLOPs,
        UnaryOp.Rsqrt or UnaryOp.Sqrt => SqrtFLOPs,
        UnaryOp.Square => 2,
        UnaryOp.Tanh => TanhFLOPs,
        UnaryOp.BitwiseNot => 1,
        UnaryOp.LogicalNot => 1,
        _ => 1,
    };

    public static UInt128 GetBinaryFLOPs(BinaryOp op) => op switch
    {
        BinaryOp.Add => AddFLOPs,
        BinaryOp.Sub => SubFLOPs,
        BinaryOp.Mul => MulFLOPs,
        BinaryOp.Div => DivFLOPs,
        BinaryOp.Mod => 1,
        BinaryOp.Min or BinaryOp.Max => CmpFLOPs,
        BinaryOp.Pow => 1,
        BinaryOp.BitwiseAnd => 1,
        BinaryOp.BitwiseOr => 1,
        BinaryOp.BitwiseXor => 1,
        BinaryOp.LogicalAnd => 1,
        BinaryOp.LogicalOr => 1,
        BinaryOp.LogicalXor => 1,
        BinaryOp.LeftShift => 1,
        BinaryOp.RightShift => 1,
        _ => 1,
    };

    public static UInt128 GetMatMulFLOPs(UInt128 m, UInt128 n, UInt128 k) => m * n * ((2 * k) - 1);
}
