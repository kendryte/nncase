﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Runtime.Interop;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.SimulatorTest;

[AutoSetupTestMethod(InitSession = false)]
public class UnitTestInteropIntegrated : TestClassBase
{
    private readonly byte[] _kmodel;

    public UnitTestInteropIntegrated()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1 }));
        var y = x + 1.0f;
        var main = new Function("main", y, new[] { x });
        var module = new IRModule(main);
        var target = CompilerServices.GetTarget(CPUTarget.Kind);
        var stackVMModuleBuilder = new StackVMModuleBuilder();
        var modelBuilder = new ModelBuilder(target, CompileOptions, stackVMModuleBuilder);
        var linkedModel = modelBuilder.Build(module);
        using var output = new MemoryStream();
        linkedModel.Serialize(output);
        _kmodel = output.ToArray();
    }

    [Fact]
    public void TestRTInterpreterLoadModel()
    {
        var interp = RTInterpreter.Create();
        interp.LoadModel(_kmodel, true);
        var entry = interp.Entry;
        Assert.NotNull(entry);
        Assert.Equal(1u, entry!.ParamsCount);
    }

    [Fact]
    public void TestRTInterpreterRunModel()
    {
        var interp = RTInterpreter.Create();
        interp.LoadModel(_kmodel, true);
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
}

public class UnitTestInterop
{
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
        Assert.Equal(MemoryMarshal.Cast<int, uint>(tensor.Dimensions.ToInts()).ToArray(), rtTensor.Dimensions.ToArray());
        Assert.Equal(MemoryMarshal.Cast<int, uint>(tensor.Strides.ToInts()).ToArray(), rtTensor.Strides.ToArray());

        var buffer = rtTensor.Buffer.Buffer.AsHost()!;
        using (var mmOwner = buffer.Map(RTMapAccess.Read))
        {
            Assert.Equal(mmOwner.Memory.Span.ToArray(), tensor.BytesBuffer.ToArray());
        }
    }

    [Fact]
    public void TestRTDatatype()
    {
        {
            var dt1 = RTDataType.FromTypeCode(Runtime.TypeCode.Int16);
            Assert.False(dt1.IsInvalid);
        }

        {
            var dt = new IR.NN.PagedAttentionKVCacheType();
            var rdt = RTDataType.From(dt);
            Assert.IsType<RTValueType>(rdt);
            var rvt = rdt as RTValueType;
            var bytes = dt.Uuid.ToByteArray();
            var uuid = new System.Guid(bytes);
            Assert.Equal(dt.Uuid, uuid);
            Assert.Equal(dt.Uuid, rvt!.Uuid);
        }

        {
            var dtt = new IR.NN.PagedAttentionKVCacheType();
            var dt = new ReferenceType(dtt);
            var rdt = RTDataType.From(dt);
            Assert.IsType<RTReferenceType>(rdt);
            var rrt = rdt as RTReferenceType;
            var rvt = rrt!.ElemType;
            Assert.IsType<RTValueType>(rvt);
            var rvvt = rvt as RTValueType;
            Assert.Equal(dtt.Uuid, rvvt!.Uuid);
        }
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

    [Fact]
    public void TestRTAttentionConfig()
    {
        var a = new IR.NN.AttentionConfig(1, 2, 3, DataTypes.Float32);
        var r_a = RTAttentionConfig.FromConfig(a);
        Assert.Equal(a.NumLayers, r_a.NumLayers);
        Assert.Equal(a.NumKVHeads, r_a.NumKVHeads);
        Assert.Equal(a.HeadDim, r_a.HeadDim);
        r_a.NumLayers = 3;
        r_a.NumKVHeads = 2;
        r_a.HeadDim = 1;
        Assert.Equal(3, r_a.NumLayers);
        Assert.Equal(2, r_a.NumKVHeads);
        Assert.Equal(1, r_a.HeadDim);
        {
            var b = new IR.NN.PagedAttentionConfig(1, 2, 3, DataTypes.Float16, 4, new[] { IR.NN.PagedKVCacheDimKind.BlockSize, IR.NN.PagedKVCacheDimKind.HeadDim, IR.NN.PagedKVCacheDimKind.KV, IR.NN.PagedKVCacheDimKind.NumBlocks, IR.NN.PagedKVCacheDimKind.NumKVHeads, IR.NN.PagedKVCacheDimKind.NumLayers }, [], [], [], []);
            var r_ = RTAttentionConfig.FromConfig(b);
            Assert.IsType<RTPagedAttentionConfig>(r_);
            var r_b = (RTPagedAttentionConfig)r_;
            Assert.Equal(b.NumLayers, r_b.NumLayers);
            Assert.Equal(b.NumKVHeads, r_b.NumKVHeads);
            Assert.Equal(b.HeadDim, r_b.HeadDim);
            Assert.Equal(b.BlockSize, r_b.BlockSize);
            r_b.NumLayers = 3;
            r_b.NumKVHeads = 2;
            r_b.HeadDim = 1;
            r_b.BlockSize = 0;
            Assert.Equal(3, r_b.NumLayers);
            Assert.Equal(2, r_b.NumKVHeads);
            Assert.Equal(1, r_b.HeadDim);
            Assert.Equal(0, r_b.BlockSize);

            Assert.Empty(r_b.PackedAxes);
            Assert.Empty(r_b.Lanes);
            r_b.PackedAxes = new[] { IR.NN.PagedKVCacheDimKind.HeadDim };
            r_b.Lanes = new[] { 64 };
            Assert.True(r_b.PackedAxes.SequenceEqual(new[] { IR.NN.PagedKVCacheDimKind.HeadDim }));
            Assert.True(r_b.Lanes.SequenceEqual(new[] { 64 }));

            Assert.Throws<InvalidOperationException>(() =>
            {
                r_b.Lanes = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            });
        }

        {
            var config = new IR.NN.PagedAttentionConfig(1, 2, 3, DataTypes.Float16, 4, new[] { IR.NN.PagedKVCacheDimKind.BlockSize, IR.NN.PagedKVCacheDimKind.HeadDim, IR.NN.PagedKVCacheDimKind.KV, IR.NN.PagedKVCacheDimKind.NumBlocks, IR.NN.PagedKVCacheDimKind.NumKVHeads, IR.NN.PagedKVCacheDimKind.NumLayers }, new[] { IR.NN.PagedKVCacheDimKind.HeadDim }, new[] { 32 }, new[] { IR.NN.PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(1, 2) });
            var rtConfig = RTAttentionConfig.FromConfig(config);
            Assert.IsType<RTPagedAttentionConfig>(rtConfig);
            var rtPagedConfig = (RTPagedAttentionConfig)rtConfig;
            Assert.True(rtPagedConfig.AxisPolicies.SequenceEqual([SBP.S(1, 2)]));
        }
    }

    [Fact]
    public void TestRTAttentionKVCache()
    {
        var pagedConfig = new IR.NN.PagedAttentionConfig(
            NumLayers: 2,
            NumKVHeads: 4,
            HeadDim: 32,
            KVType: DataTypes.Float32,
            BlockSize: 16,
            CacheLayout: new[]
            {
                IR.NN.PagedKVCacheDimKind.NumBlocks,
                IR.NN.PagedKVCacheDimKind.NumLayers,
                IR.NN.PagedKVCacheDimKind.KV,
                IR.NN.PagedKVCacheDimKind.BlockSize,
                IR.NN.PagedKVCacheDimKind.NumKVHeads,
                IR.NN.PagedKVCacheDimKind.HeadDim,
            },
            PackedAxes: new[] { IR.NN.PagedKVCacheDimKind.HeadDim },
            Lanes: new[] { 32 },
            new[] { IR.NN.PagedKVCacheDimKind.NumBlocks },
            new[] { SBP.S(0) });

        var contextLens = Tensor.From(new[] { 64L });
        var seqLens = Tensor.From(new[] { 128L });
        var blockTable = Tensor.From(new long[] { 0, -1, -1, -1 }); // Example block table
        var slotMapping = Tensor.From(new long[] { 5, 4, 3, 2, 1 }); // Example slot mapping

        var rtPagedConfig = RTPagedAttentionConfig.FromConfig(pagedConfig);

        var rtContextLens = RTTensor.FromTensor(contextLens);
        var rtSeqLens = RTTensor.FromTensor(seqLens);
        var rtBlockTable = RTTensor.FromTensor(blockTable);
        var rtSlotMapping = RTTensor.FromTensor(slotMapping);
        var rtpagedAttn = RTPagedAttentionKVCache.Create(
            rtPagedConfig, 1, 64, rtContextLens, rtSeqLens, rtBlockTable, rtSlotMapping, 15, [1]);

        Assert.Equal(15, rtpagedAttn.NumBlocks);

        var totensor = Tensor.FromScalar(new Reference<IR.NN.IPagedAttentionKVCache>(rtpagedAttn));
        RTTensor.FromTensor(totensor);
    }

#if false
    [Fact]
    public void TestRTPagedAttentionScheduler()
    {
        var cfg = new IR.NN.PagedAttentionConfig(1, 2, 3, 4, [IR.NN.PagedKVCacheDimKind.BlockSize, IR.NN.PagedKVCacheDimKind.HeadDim, IR.NN.PagedKVCacheDimKind.KV, IR.NN.PagedKVCacheDimKind.NumBlocks, IR.NN.PagedKVCacheDimKind.NumKVHeads, IR.NN.PagedKVCacheDimKind.NumLayers], [], [], DataTypes.Float32);
        var s = RTPagedAttentionScheduler.Create(cfg, 128, 1238);

        var sessionIds = Tensor.From([1L]);
        var tokenCounts = Tensor.From([128L]);
        var cache = s.Schedule(sessionIds, tokenCounts);
        Assert.Equal(1, cache.NumSeqs);
        Assert.Equal(128, cache.SeqLen(0));
        Assert.Equal(0, cache.ContextLen(0));
    }
#endif
}
