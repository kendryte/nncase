// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using NetFabric.Hyperlinq;
using Nncase.IR;
using static NetFabric.Hyperlinq.ArrayExtensions;

namespace Nncase.CostModel;

public static class CostFactorNames
{
    public static readonly string MemoryLoad = "MemoryLoad";

    public static readonly string MemoryStore = "MemoryStore";

    public static readonly string CPUCycles = "CPUCycles";
}

/// <summary>
/// Cost.
/// </summary>
public sealed record Cost(bool Freezed = false) : IComparable<Cost>, IEquatable<Cost>
{
    /// <summary>
    /// Zero cost.
    /// </summary>
    public static readonly Cost Zero = new(true);

    /// <summary>
    /// Gets or sets factors.
    /// </summary>
    public Dictionary<string, UInt128> Factors { get; set; } = new();

    /// <summary>
    /// Gets score.
    /// </summary>
    public UInt128 Score => Factors.Sum(x => x.Value);

    public UInt128 this[string name]
    {
        get => Factors[name];
        set
        {
            if (Freezed)
            {
                throw new InvalidOperationException("Can't modify freezed cost!");
            }

            Factors[name] = value;
        }
    }

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
    public static Cost operator +(Cost lhs, Cost rhs)
    {
        var newCost = new Cost() with { Factors = new(lhs.Factors) };
        foreach (var factor in rhs.Factors)
        {
            if (newCost.Factors.TryGetValue(factor.Key, out var oldValue))
            {
                newCost.Factors[factor.Key] = oldValue + factor.Value;
            }
            else
            {
                newCost.Factors.Add(factor.Key, factor.Value);
            }
        }

        return newCost;
    }

    /// <summary>
    /// Multiply cost with a scale.
    /// </summary>
    /// <param name="lhs">Lhs.</param>
    /// <param name="scale">Scale.</param>
    /// <returns>Added result.</returns>
    public static Cost operator *(Cost lhs, UInt128 scale)
    {
        var newCost = new Cost();
        foreach (var factor in lhs.Factors)
        {
            newCost.Factors.Add(factor.Key, factor.Value * scale);
        }

        return newCost;
    }

    public static bool operator <=(Cost left, Cost right)
    {
        return left is null || left.CompareTo(right) <= 0;
    }

    public static bool operator >=(Cost left, Cost right)
    {
        return left is null ? right is null : left.CompareTo(right) >= 0;
    }

    /// <inheritdoc/>
    public int CompareTo(Cost? other)
    {
        return Comparer<UInt128>.Default.Compare(Score, other?.Score ?? 0UL);
    }

    /// <inheritdoc/>
    public bool Equals(Cost? other)
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
        if (Equals(Cost.Zero))
        {
            return "Zero";
        }

        return $"{{ {string.Join(", ", Factors.Select(kv => $"{kv.Key}: {kv.Value}"))}, Score:{Score} }}";
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
        return costs.Aggregate(Cost.Zero, (x, y) => x + y)!;
    }

    /// <summary>
    /// Sum all costs.
    /// </summary>
    /// <param name="costs">Source.</param>
    /// <returns>Result.</returns>
    public static Cost Sum<TSource, TSelector>(this in SpanSelectEnumerable<TSource, Cost, TSelector> costs)
            where TSelector : struct, IFunction<TSource, Cost>
    {
        var sum = Cost.Zero;
        foreach (var cost in costs)
        {
            sum += cost;
        }

        return sum;
    }
}

public static class CostUtility
{
    public static UInt128 GetMemoryAccess(IRType type)
    {
        return type switch
        {
            TensorType t => (UInt128)(t.Shape.Aggregate(1D, (acc, x) => acc * (x.IsFixed ? x.FixedValue : 1)) * t.DType.SizeInBytes),
            TupleType t => t.Fields.Sum(GetMemoryAccess),
            _ => 0,
        };
    }

    public static UInt128 GetMemoryAccess(params IRType[] types)
    {
        return types.Aggregate((UInt128)0, (sum, type) => sum + GetMemoryAccess(type));
    }

    public static UInt128 GetFakeMemoryAccess(IRType type, uint bits)
    {
        return type switch
        {
            TensorType t => (UInt128)Math.Ceiling((float)t.Shape.Aggregate(1D, (acc, x) => acc * (x.IsFixed ? x.FixedValue : 1)) * t.DType.SizeInBytes * bits / 8),
            TupleType t => t.Fields.Sum(x => GetFakeMemoryAccess(x, bits)),
            _ => 0,
        };
    }

    public static UInt128 GetCPUCycles(IRType type, double cyclesPerElement = 1)
    {
        return type switch
        {
            TensorType t => (UInt128)(t.Shape.Aggregate(1D, (acc, x) => acc * (x.IsFixed ? x.FixedValue : 1)) * cyclesPerElement),
            TupleType t => t.Fields.Sum(GetMemoryAccess),
            _ => 0,
        };
    }

    public static uint GetCPUCyclesOfUnary(UnaryOp unaryOp)
    {
        // TODO: Arch dependent
        return unaryOp switch
        {
            UnaryOp.Abs => 1,
            UnaryOp.Acos => 8,
            UnaryOp.Acosh => 8,
            UnaryOp.Asin => 8,
            UnaryOp.Asinh => 8,
            UnaryOp.Ceil => 1,
            UnaryOp.Cos => 8,
            UnaryOp.Cosh => 8,
            UnaryOp.Exp => 8,
            UnaryOp.Floor => 1,
            UnaryOp.Log => 8,
            UnaryOp.Neg => 1,
            UnaryOp.Round => 1,
            UnaryOp.Rsqrt => 4,
            UnaryOp.Sin => 8,
            UnaryOp.Sinh => 8,
            UnaryOp.Sign => 1,
            UnaryOp.Sqrt => 8,
            UnaryOp.Square => 2,
            UnaryOp.Tanh => 8,
            UnaryOp.BitwiseNot => 1,
            UnaryOp.LogicalNot => 1,
            _ => 1,
        };
    }

    public static uint GetCPUCyclesOfBinary(BinaryOp binaryOp)
    {
        // TODO: Arch dependent
        return binaryOp switch
        {
            BinaryOp.Add => 1,
            BinaryOp.Sub => 1,
            BinaryOp.Mul => 2,
            BinaryOp.Div => 8,
            BinaryOp.Mod => 8,
            BinaryOp.Min => 1,
            BinaryOp.Max => 1,
            BinaryOp.Pow => 8,
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
    }

    // todo:GetCPUCyclesOfMath
    public static uint GetCPUCyclesOfMax()
    {
        return 1;
    }

    public static uint GetCPUCyclesOfCompare()
    {
        return 1;
    }

    // cost for op similar to reshape, e.g. squeeze
    public static Cost GetReshapeCost()
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public static Cost GetShapeExprCost()
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public static Cost GetActivationCost(TensorType ret, uint macPerElement)
    {
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    // cost for op similar to broadcast
    public static Cost GetBroadcastCost(TensorType input, TensorType ret)
    {
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = 1,
        };
    }
}
