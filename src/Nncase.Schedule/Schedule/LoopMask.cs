// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Schedule;

/// <summary>
/// loops access mask for buffer.
/// <example>
/// For example, buffer a[m,k] is accessed by loop `m,n,k`, the mask is 101(knm) (binary) = 13 (decimal).
/// </example>
/// </summary>
public struct LoopMask
{
    private uint _mask;

    public LoopMask(uint mask)
    {
        _mask = mask;
    }

    public int Ones => BitOperations.PopCount(_mask);

    public void SetRelated(int loop) => _mask |= 1U << loop;

    public bool IsRelated(int loop) => (_mask & (1 << loop)) != 0;

    public int LastRelated(int rank)
    {
        for (int i = rank - 1; i >= 0; i--)
        {
            if (IsRelated(i))
            {
                return i;
            }
        }

        return -1;
    }

    public bool IsRelated(IR.Affine.AffineDim dim) => (_mask & (1 << dim.Position)) != 0;

    public override string ToString() => Convert.ToString(_mask, 2);
}
