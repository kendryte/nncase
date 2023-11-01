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
    public static ArrayExtensions.SpanWhereEnumerable<TIR.Buffer, FunctionWrapper<TIR.Buffer, bool>> InputOf(this ReadOnlySpan<TIR.Buffer> arr) => arr.AsValueEnumerable().Where(b => b.MemSpan.Location == MemoryLocation.Input);

    public static ArrayExtensions.SpanWhereEnumerable<TIR.Buffer, FunctionWrapper<TIR.Buffer, bool>> OutputOf(this ReadOnlySpan<TIR.Buffer> arr) => arr.AsValueEnumerable().Where(b => b.MemSpan.Location == MemoryLocation.Output);
}

public sealed class UnitTestStringUtility
{
    private readonly TIR.PrimFunction _entry = new("test_module", new Sequential(1), new TIR.Buffer("testInput", DataTypes.Float32, new MemSpan(0, 123, MemoryLocation.Input), new Expr[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new Expr[] { 1, 16, 64, 400 })), new TIR.Buffer("testInput", DataTypes.Float32, new MemSpan(0, 123, MemoryLocation.Output), new Expr[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new Expr[] { 1, 16, 64, 400 })));

    [Fact]
    public void TestJoin()
    {
        var result = StringUtility.Join(",", _entry.Parameters.InputOf().Select(b => b));
        Assert.Equal("Nncase.TIR.Buffer", result);

        var result1 = StringUtility.Join(",", _entry.Parameters.OutputOf().Select(b => b));
        Assert.Equal("Nncase.TIR.Buffer", result1);
    }
}
