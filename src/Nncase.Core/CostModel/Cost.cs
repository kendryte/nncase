// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.CostModel;

/// <summary>
/// Cost.
/// </summary>
/// <param name="Arith">Arithmetic cost.</param>
/// <param name="Memory">Memory cost.</param>
public sealed record Cost(double Arith = 0, double Memory = 0) : IComparable<Cost>
{
    /// <summary>
    /// Zero cost.
    /// </summary>
    public static readonly Cost Zero = new(0, 0);

    /// <summary>
    /// Gets score.
    /// </summary>
    public double Score => (Arith * 2) + Memory;

    /// <summary>
    /// Whether lhs is greater than rhs.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="rhs">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator >(Cost lhs, Cost rhs) => lhs.Score > rhs.Score;

    /// <summary>
    /// Whether lhs is less than rhs.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="rhs">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator <(Cost lhs, Cost rhs) => lhs.Score < rhs.Score;

    /// <summary>
    /// Add two cost.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="rhs">Rhs.</param>
    /// <returns>Added result.</returns>
    public static Cost operator +(Cost lhs, Cost rhs) => new(lhs.Arith + rhs.Arith, lhs.Memory + rhs.Memory);

    /// <summary>
    /// Multiply cost with a scale.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="scale">Scale.</param>
    /// <returns>Added result.</returns>
    public static Cost operator *(Cost lhs, double scale) => new(lhs.Arith * scale, lhs.Memory * scale);

    /// <inheritdoc/>
    public int CompareTo(Cost? other)
    {
        return (int)(Score - other?.Score ?? 0);
    }
}

/// <summary>
/// Cost extensions.
/// </summary>
public static class CostExtensions
{
    /// <summary>
    /// Sum all costs.
    /// </summary>
    /// <param name="costs">Source.</param>
    /// <returns>Result.</returns>
    public static Cost Sum(this IEnumerable<Cost> costs)
    {
        return costs.Aggregate(Cost.Zero, (x, y) => x + y);
    }
}
