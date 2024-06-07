// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
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
}
