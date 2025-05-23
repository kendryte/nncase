// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR.Shapes;

internal static class DimHelpers
{
    // Cache for factorial calculations to avoid redundant computations
    private static readonly Dictionary<int, long> _factorialCache = new Dictionary<int, long> { { 0, 1 }, { 1, 1 } };

    public static (long Scale, Dictionary<Dimension, int> Pows) GetScaleAndPows(Dimension dimension)
    {
        long scale = dimension switch
        {
            DimConst dimConst => dimConst.Value,
            DimProduct dimProduct => dimProduct.Scale,
            _ => 1,
        };
        var operands = dimension switch
        {
            DimConst => Array.Empty<Dimension>(),
            DimProduct dimProduct => dimProduct.Operands,
            _ => new[] { dimension },
        };

        var pows = new Dictionary<Dimension, int>(ReferenceEqualityComparer.Instance);
        foreach (var operand in operands)
        {
            if (operand is DimConst dimConst)
            {
                scale *= dimConst.Value;
            }
            else if (operand is DimPower powerOfDim)
            {
                ref var value = ref CollectionsMarshal.GetValueRefOrAddDefault(pows, powerOfDim.Dim, out _);
                value += powerOfDim.Power;
            }
            else
            {
                ref var value = ref CollectionsMarshal.GetValueRefOrAddDefault(pows, operand, out _);
                value += 1;
            }
        }

        return (scale, pows);
    }

    public static Dimension Simplify(long scale, IReadOnlyDictionary<Dimension, int> pows)
    {
        var newOperands = pows.Select(kvp => Dimension.Pow(kvp.Key, kvp.Value)).ToArray();
        return (scale, newOperands.Length) switch
        {
            (_, 0) => new DimConst(scale),
            (0, 1) => newOperands[0],
            _ => new DimProduct(newOperands, scale),
        };
    }

    public static Dimension Pow(DimSum source, int power)
    {
        ReadOnlySpan<Dimension> operands = [source.Bias, .. source.Operands];
        var combinations = GenerateCombinationsIterative(power, operands.Length);

        long resultBias = 0;
        var result = new List<Dimension>(combinations.Count);
        foreach (var comb in combinations)
        {
            Dimension term = MultinomialCoefficient(power, comb);
            for (int i = 0; i < comb.Count; i++)
            {
                if (comb[i] > 0)
                {
                    term *= Dimension.Pow(operands[i], comb[i]);
                }
            }

            if (term is DimConst dimConst)
            {
                resultBias += dimConst.Value;
            }
            else
            {
                result.Add(term);
            }
        }

        return result.Count switch
        {
            0 => new DimConst(resultBias),
            _ => new DimSum(result.ToArray(), resultBias),
        };
    }

    // Calculate factorial with memoization
    private static long Factorial(int n)
    {
        if (_factorialCache.TryGetValue(n, out long result))
        {
            return result;
        }

        result = n * Factorial(n - 1);
        _factorialCache[n] = result;
        return result;
    }

    // Calculate multinomial coefficient: n! / (k1! * k2! * ... * km!)
    private static long MultinomialCoefficient(int n, List<int> k)
    {
        long result = Factorial(n);
        foreach (int ki in k)
        {
            if (ki > 1)
            {
                // Skip unnecessary divisions
                result /= Factorial(ki);
            }
        }

        return result;
    }

    // Generate all combinations of exponents where k1 + k2 + ... + km = n
    // Using iterative approach to avoid stack overflow
    private static List<List<int>> GenerateCombinationsIterative(int n, int m)
    {
        var result = new List<List<int>>();

        // Initial state: all variables have exponent 0, except the last one which has n
        var current = new List<int>(m);
        for (int i = 0; i < m - 1; i++)
        {
            current.Add(0);
        }

        current.Add(n);

        result.Add([.. current]);

        // Continue generating all other combinations
        bool hasNext = true;
        while (hasNext)
        {
            hasNext = false;

            // Find the rightmost non-zero term that can be decremented
            int i;
            for (i = m - 1; i >= 0; i--)
            {
                if (current[i] > 0)
                {
                    hasNext = true;
                    break;
                }
            }

            if (hasNext)
            {
                // Decrement the found term
                current[i]--;

                // Add the sum of all terms to the right to the next position
                int sum = 1; // 1 is what we just subtracted from current[i]
                for (int j = i + 1; j < m; j++)
                {
                    sum += current[j];
                    current[j] = 0;
                }

                current[i + 1] = sum;

                result.Add([.. current]);
            }
        }

        return result;
    }
}
