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

    public static readonly string OffMemoryTraffic = "OffChipMemoryTraffic";
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
}
