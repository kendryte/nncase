// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;
using OrtKISharp;
using Xunit;
using Buffer = Nncase.TIR.Buffer;
using Range = Nncase.TIR.Range;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTIR
{
    [Fact]
    public void TestLogicalBuffer()
    {
        var logicalBuffer1 = new LogicalBuffer("logicalBuffer", default, new TensorConst(new Tensor<int>(new[] { 1 })));
        var logicalBuffer2 = new LogicalBuffer("logicalBuffer", DataTypes.Int32, default, new[] { (Expr)new Tensor<int>(new[] { 1 }) });
        Assert.Equal(logicalBuffer2.Length.ToString(), logicalBuffer1.Length.ToString());
        Assert.Equal("LogicalBuffer(logicalBuffer, i32, MemLocation)", logicalBuffer1.ToString());
    }

    [Fact]
    public void TestScheduler()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 0.8F;
        var original = IR.F.NN.Celu(input.ToTensor(), alpha);
        var function = new Function(original);
        var scheduler = new Scheduler(new Function("Celu", original));
        Assert.Equal(function, scheduler.Entry);
        Assert.Throws<InvalidOperationException>(() => scheduler.GetBlock("test"));
        scheduler.GetLoops(new Block("block"));
        scheduler.Split(new For("loopVar", new Range(0, 0, 0), LoopMode.Parallel, new Sequential()), new Expr[] { original });
    }

    [Fact]
    public void TestBufferStore()
    {
        Assert.Throws<InvalidOperationException>(() => T.Store(null!, null!));

        var variable = new Var("x", DataTypes.Int32);
        int index = 0;
        Expr loadOp = T.Load(variable, index);
        Expr value = 42;
        _ = T.Store(loadOp, value);

        var physicalBuffer = new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0);
        var indices = new Expr[] { 0, 1 };
        Expr storeOp = T.Store(new BufferLoad(physicalBuffer, indices), value);
        var store = (BufferStore)storeOp;
        Assert.Equal(physicalBuffer, store.Buffer);
        Assert.Equal(value, store.Value);
        _ = store.Indices;
    }
}
