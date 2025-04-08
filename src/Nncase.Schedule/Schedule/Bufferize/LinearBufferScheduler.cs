// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.Bufferize;

public sealed class LinearBufferScheduler : BufferScheduler
{
    public LinearBufferScheduler(MemoryLocation memoryLocation)
        : base(memoryLocation)
    {
    }

    protected override bool TryScheduleCore(IEnumerable<BufferLifetime> lifetimes, long maxMemoryPoolSize, out long memoryPoolSize)
    {
        memoryPoolSize = 0;
        foreach (var lifetime in lifetimes)
        {
            var alignment = lifetime.Buffer.ElemType.SizeInBytes;
            var start = MathUtility.AlignUp(memoryPoolSize, alignment);
            var size = lifetime.Memory.Size;
            lifetime.Memory.Start = start;
            memoryPoolSize = lifetime.Memory.Stop = start + size;
        }

        return true;
    }
}
