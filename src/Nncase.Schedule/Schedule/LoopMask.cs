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

public struct LoopMask
{
    private readonly uint _mask;

    public LoopMask(uint mask)
    {
        _mask = mask;
    }

    public int Ones => BitOperations.PopCount(_mask);

    public static LoopMask operator &(LoopMask left, LoopMask right) => new LoopMask(left._mask & right._mask);

    public bool IsRelated(int loop) => (_mask & (1 << loop)) != 0;

    public bool IsRelated(IR.Affine.AffineDim dim) => (_mask & (1 << dim.Position)) != 0;

    public override string ToString() => Convert.ToString(_mask, 2);
}

public record LoopMasks(LoopMask[] Masks) : IReadOnlyList<LoopMask>
{
    public int Count => Masks.Length;

    public LoopMask this[int index] => Masks[index];

    public IEnumerator<LoopMask> GetEnumerator() => ((IEnumerable<LoopMask>)Masks).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => Masks.GetEnumerator();

    public bool IsRelated(int dim) => Masks.Any(m => m.IsRelated(dim));

    public bool IsRelated(IR.Affine.AffineDim dim) => Masks.Any(m => m.IsRelated(dim));

    public int IndexOf(IR.Affine.AffineDim dim)
    {
        for (int j = 0; j < Masks.Length; j++)
        {
            if (Masks[j].IsRelated(dim))
            {
                return j;
            }
        }

        return -1;
    }
}
