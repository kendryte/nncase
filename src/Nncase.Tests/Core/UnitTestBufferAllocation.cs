// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Nncase;
using Nncase.Converters;
using Nncase.Runtime;
using Nncase.Schedule;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestBufferAllocation
{
    [Fact]
    public void TestBufferAllocation()
    {
        var bufferAllocation = new BufferAllocation(default, DataTypes.Int8, 1UL, 1UL, 1UL, new[] { 1, 3, 16, 16 }, new[] { 1, 1, 1, 1 }, new[] { 1 });
        Assert.Equal(default, bufferAllocation.MemoryLocate);
        Assert.Equal(1UL, bufferAllocation.SharedModule);
        Assert.Equal(1UL, bufferAllocation.Start);
        Assert.Equal(DataTypes.Int8, bufferAllocation.DType);
        Assert.Equal(1UL, bufferAllocation.Size);
        Assert.Equal(new[] { 1, 3, 16, 16 }, bufferAllocation.Shape);
        Assert.Equal(new[] { 1, 1, 1, 1 }, bufferAllocation.Strides);
        Assert.Equal(new[] { 1 }, bufferAllocation.StridesShape);
        Assert.Equal(2UL, bufferAllocation.LinearEnd());
        Assert.True(bufferAllocation.Overlap(bufferAllocation));
        var bufferAllocationMemoryRange = bufferAllocation.MemoryRange;
        var memoryRange = new MemoryRange(bufferAllocation.MemoryLocate, PrimTypeCodes.ToTypeCode(DataTypes.Int8), (ushort)bufferAllocation.SharedModule, (uint)bufferAllocation.Start, (uint)bufferAllocation.Size);
        Assert.Equal(memoryRange, bufferAllocationMemoryRange);

        Assert.Throws<KeyNotFoundException>(() => new BufferAllocation(default, DataTypes.Utf8Char, 1UL, 1UL, 1UL, new[] { 1, 3, 16, 16 }, new[] { 1, 1, 1, 1 }, new[] { 1 }).MemoryRange);
    }
}
