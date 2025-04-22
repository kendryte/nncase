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
}
