// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Immutable;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace Nncase.Tests.CoreTest;

public static class TestExtensions
{
    public static ArrayExtensions.SpanWhereEnumerable<TIR.PhysicalBuffer, FunctionWrapper<TIR.PhysicalBuffer, bool>> InputOf(this ReadOnlySpan<TIR.PhysicalBuffer> arr) => arr.AsValueEnumerable().Where(b => b.MemLocation == Schedule.MemoryLocation.Input);

    public static ArrayExtensions.SpanWhereEnumerable<TIR.PhysicalBuffer, FunctionWrapper<TIR.PhysicalBuffer, bool>> OutputOf(this ReadOnlySpan<TIR.PhysicalBuffer> arr) => arr.AsValueEnumerable().Where(b => b.MemLocation == Schedule.MemoryLocation.Output);
}

public sealed class UnitTestStringUtility
{
    private readonly TIR.PrimFunction _entry = new("test_module", new Sequential(1), new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0), new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0));

    [Fact]
    public void TestJoin()
    {
        var result = StringUtility.Join(",", _entry.Parameters.InputOf().Select(b => b));
        Assert.Equal("PhysicalBuffer(testInput, f32, MemLocation),PhysicalBuffer(testInput, f32, MemLocation)", result);

        var result1 = StringUtility.Join(",", _entry.Parameters.OutputOf().Select(b => b));
        Assert.Equal(string.Empty, result1);
    }
}
