// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.InteropServices;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Runtime.Interop;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.SimulatorTest;

[AutoSetupTestMethod(InitSession = false)]
public class UnitTestInterop : TestClassBase
{
    private readonly byte[] _kmodel;

    public UnitTestInterop()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget(CPUTarget.Kind);
        var modelBuilder = new ModelBuilder(target, CompileOptions);
        var linkedModel = modelBuilder.Build(module);
        using var output = new MemoryStream();
        linkedModel.Serialize(output);
        _kmodel = output.ToArray();
    }

    [Fact]
    public void TestCreateRTInterpreter()
    {
        var interp = RTInterpreter.Create();
        Assert.NotNull(interp);
    }

    [Fact]
    public void TestGetHostBufferAllocator()
    {
        var allocator = RTBufferAllocator.Host;
        Assert.NotNull(allocator);
    }

    [Fact]
    public void TestAllocateHostBuffer()
    {
        var allocator = RTBufferAllocator.Host;
        var buffer = allocator.Allocate(256);
        Assert.NotNull(buffer.AsHost());
    }

    [Fact]
    public void TestMapHostBuffer()
    {
        var allocator = RTBufferAllocator.Host;
        var buffer = allocator.Allocate(256).AsHost();
        Assert.NotNull(buffer);
        using (var mmOwner = buffer!.Map(RTMapAccess.Write))
        {
            mmOwner.Memory.Span.Fill(1);
        }

        using (var mmOwner = buffer.Map(RTMapAccess.Read))
        {
            Assert.All(mmOwner.Memory.Span.ToArray(), x => Assert.Equal(1, x));
        }
    }

    [Fact]
    public void TestDataTypeCreatePrim()
    {
        var dtype = RTDataType.FromTypeCode(Runtime.TypeCode.Float32);
        Assert.NotNull(dtype);
    }

    [Fact]
    public void TestCreateTensor()
    {
        var allocator = RTBufferAllocator.Host;
        var buffer = allocator.Allocate(256);
        var dtype = RTDataType.FromTypeCode(Runtime.TypeCode.Float32);
        var dims = new uint[] { 1, 64 };
        var strides = new uint[] { 1, 1 };
        var bufferSlice = new RTBufferSlice { Buffer = buffer, Start = 0, SizeBytes = 256 };
        var tensor = RTTensor.Create(dtype, dims, strides, bufferSlice);
        Assert.NotNull(tensor);
        Assert.Equal(dtype, tensor.ElementType);
        Assert.Equal(bufferSlice, tensor.Buffer);
        Assert.Equal(dims, tensor.Dimensions.ToArray());
        Assert.Equal(strides, tensor.Strides.ToArray());
    }

    [Fact]
    public void TestCreateTensorFromTensor()
    {
        var tensor = (Tensor)new float[] { 1.0f, 2.0f };
        var rtTensor = RTTensor.FromTensor(tensor);
        var dtype = RTDataType.FromTypeCode(Runtime.TypeCode.Float32);
        Assert.NotNull(rtTensor);
        Assert.Equal(dtype, rtTensor.ElementType);
        Assert.Equal(MemoryMarshal.Cast<int, uint>(tensor.Dimensions).ToArray(), rtTensor.Dimensions.ToArray());
        Assert.Equal(MemoryMarshal.Cast<int, uint>(tensor.Strides).ToArray(), rtTensor.Strides.ToArray());

        var buffer = rtTensor.Buffer.Buffer.AsHost()!;
        using (var mmOwner = buffer.Map(RTMapAccess.Read))
        {
            Assert.Equal(mmOwner.Memory.Span.ToArray(), tensor.BytesBuffer.ToArray());
        }
    }

    [Fact]
    public void TestRTInterpreterLoadModel()
    {
        var interp = RTInterpreter.Create();
        interp.LoadModel(_kmodel);
        var entry = interp.Entry;
        Assert.NotNull(entry);
        Assert.Equal(1u, entry!.ParamsCount);
    }

    [Fact]
    public void TestRTInterpreterRunModel()
    {
        var interp = RTInterpreter.Create();
        interp.LoadModel(_kmodel);
        var entry = interp.Entry;
        Assert.NotNull(entry);

        var input = RTTensor.FromTensor(new[] { 2.0f });
        var result = (RTTensor)entry!.Invoke(input);
        var buffer = result.Buffer.Buffer.AsHost()!;
        using (var mmOwner = buffer.Map(RTMapAccess.Read))
        {
            Assert.Equal(new[] { 3.0f }, MemoryMarshal.Cast<byte, float>(mmOwner.Memory.Span).ToArray());
        }
    }

    [Fact]
    public void TestRTDatatype()
    {
        var dt1 = RTDataType.FromTypeCode(Runtime.TypeCode.Int16);
        Assert.False(dt1.IsInvalid);
    }

    [Fact]
    public void TestRTExtensions()
    {
        Assert.Equal(DataTypes.Boolean, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Boolean)));
        Assert.Equal(DataTypes.Int8, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Int8)));
        Assert.Equal(DataTypes.Int16, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Int16)));
        Assert.Equal(DataTypes.Int32, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Int32)));
        Assert.Equal(DataTypes.Int64, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Int64)));
        Assert.Equal(DataTypes.UInt8, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.UInt8)));
        Assert.Equal(DataTypes.UInt16, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.UInt16)));
        Assert.Equal(DataTypes.UInt32, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.UInt32)));
        Assert.Equal(DataTypes.UInt64, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.UInt64)));
        Assert.Equal(DataTypes.Float16, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Float16)));
        Assert.Equal(DataTypes.Float32, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Float32)));
        Assert.Equal(DataTypes.Float64, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.Float64)));
        Assert.Equal(DataTypes.BFloat16, RTExtensions.ToPrimType(RTDataType.FromTypeCode(Runtime.TypeCode.BFloat16)));
    }

    [Fact]
    public void TestRTTuple()
    {
        Assert.Throws<InvalidOperationException>(() => RTTuple.FromTuple(new TupleValue(ReadOnlySpan<IValue>.Empty)));

        var intVal = Value.FromConst(42);
        var floatVal = Value.FromConst(3.14f);
        var tupleValue = new TupleValue(new[] { intVal, floatVal });
        var tuple = RTTuple.FromTuple(tupleValue);
        var fields = tuple.Fields;
        Assert.Equal(2, fields.Length);
        Assert.Equal(intVal, fields[0].ToValue());
        Assert.Equal(floatVal, fields[1].ToValue());
    }
}
