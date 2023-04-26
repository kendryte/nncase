// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.TIR.Builders;
using OrtKISharp;
using Tensorflow;
using Xunit;
using Buffer = Nncase.TIR.Buffer;
using Function = Nncase.IR.Function;
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

    [Fact]
    public void TestIterVar()
    {
        var dom = new Range(-1f, 1f, 1);
        var mode = IterationMode.Opaque;
        var value = new Var("test", DataTypes.Float32);
        var iterVar = new IterVar(dom, mode, value);
        Assert.Equal(dom, iterVar.Dom);
        Assert.Equal(mode, iterVar.Mode);
        Assert.Equal(value, iterVar.Value);
    }

    [Fact]
    public void TestSizeVar()
    {
        var name = "test";
        var actual = T.SizeVar(name);
        var expected = Var.SizeVar(name);
        Assert.True(actual.ToString().Equals(expected.ToString(), StringComparison.Ordinal));
    }

    [Fact]
    public void TestSerial()
    {
        var domain = new Range(-1f, 1f, 1);
        var actual = T.Serial(out _, domain);
        var expect = T.ForLoop(out _, domain, LoopMode.Serial, "v");
        Assert.Equal(expect.ToString(), actual.ToString());
    }

    [Fact]
    public void TestSequential()
    {
        var expect1 = new SequentialBuilder<Sequential>(body => body);
        var actual1 = T.Sequential();
        Assert.Equal(expect1.ToString(), actual1.ToString());

        var expect2 = TIR.Sequential.Flatten(Array.Empty<object>());
        var actual2 = T.Sequential(Array.Empty<object>());
        Assert.Equal(expect2, actual2);
    }

    [Fact]
    public void TestBuffer()
    {
        var buffer = T.Buffer(DataTypes.Float32, MemoryLocation.Input, new Expr[] { 1, 16, 64, 400 }, out _);
        Assert.Equal(DataTypes.Float32, buffer.ElemType);
        var expect = new LogicalBuffer("_", DataTypes.Float32, MemoryLocation.Input, new Expr[] { 1, 16, 64, 400 });
        Assert.Equal(expect, buffer);
    }

    [Fact]
    public void TestForSegment()
    {
        var count = IR.F.Tensors.Cast(2 / IR.F.Tensors.Cast(2, DataTypes.Float32), DataTypes.Int32);
        var expect = T.Serial(out var i, (0, count));
        var actual = T.ForSegment(out var seg, 1, 2, 3);
        Assert.Equal(expect.ToString(), actual.ToString());
    }

    [Fact]
    public void TestGrid()
    {
        var grid1 = T.Grid(out _, LoopMode.Serial, new Range(-1f, 1f, 1));
        var grid2 = T.Grid(out _, out _, new(1, 1));
        Assert.Equal(grid1.GetDataType(), grid2.GetDataType());
    }

    [Fact]
    public void TestEmit()
    {
        int result;
        T.Emit(out result, () => 5);
        Assert.Equal(5, result);
    }

    [Fact]
    public void TestBufferRegion()
    {
        var buffer = T.Buffer(DataTypes.Float32, MemoryLocation.Input, new Expr[] { 1, 16, 64, 400 }, out _);
        var region = new Range[] { new Range(1, 2, 2), new Range(-1, 3, 2) };
        var bufferRegion = new BufferRegion(buffer, region);

        var newRegion = bufferRegion[new Range(0, 1, 2), new Range(-3, 3, 2)];
        Assert.Equal(buffer, newRegion.Buffer);
        Assert.Equal(new Range(0, 1, 2), newRegion.Region[0]);
        Assert.Equal(new Range(-3, 3, 2), newRegion.Region[1]);
    }

    [Fact]
    public void TestNop()
    {
        var nop = new Nop();
        Assert.False(nop.CanFoldConstCall);
    }

    [Fact]
    public void TestPrimFunction()
    {
        var primFunc = new PrimFunction("test_module", new Sequential(new Expr[] { 1 }), new[]
        {
            new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0),
            new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0),
        });

        var primFuncParameters = primFunc.Parameters;
        var primFuncParameterTypes = primFunc.ParameterTypes;
        var expect = primFuncParameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        Assert.Equal(expect, primFuncParameterTypes);

        var newModuleKind = "new_module";
        var newBody = new Sequential(new Expr[] { 3 });
        var newParams = new[]
        {
            new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0),
            new TIR.PhysicalBuffer("testInput", DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 16, 64, 400 }, TensorUtilities.GetStrides(new[] { 1, 16, 64, 400 }), 0, 0),
        };

        var newPrimFunc = primFunc.With(moduleKind: newModuleKind, body: newBody, parameters: newParams);

        Assert.NotSame(primFunc, newPrimFunc);
        Assert.Equal(newModuleKind, newPrimFunc.ModuleKind);
        Assert.Equal(newBody, newPrimFunc.Body);
        Assert.Equal(newParams, newPrimFunc.Parameters.ToArray());
        Assert.Equal(primFunc.Name, newPrimFunc.Name); // should not change the name

        Assert.NotNull(new PrimFunction("test_module", new Sequential(new Expr[] { 1 }), default(ReadOnlySpan<PhysicalBuffer>)));
    }
}
