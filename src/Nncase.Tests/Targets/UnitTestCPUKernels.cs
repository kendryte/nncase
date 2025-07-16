// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.Evaluator.OrtKIExtensions;

namespace Nncase.Tests.TargetTest;

public class CpuKernelCase
{
    public CpuKernelCase(string name, Fusion fusion, IVar[] vars, Tensor[] inputs, Tensor[] rtinputs)
    {
        Name = name;
        Fusion = fusion;
        Vars = vars;
        Inputs = inputs;
        RTInputs = rtinputs;
    }

    public string Name { get; }

    public Fusion Fusion { get; }

    public IReadOnlyList<IVar> Vars { get; }

    public IReadOnlyList<Tensor> Inputs { get; }

    public IReadOnlyList<Tensor> RTInputs { get; set; }
}

public sealed class TestUpdatePagedAttentionCase : TheoryData<TestFixture.PagedAttentionKVCacheTestFixture, int[], int>
{
    private static readonly (string Name, long[] QueryLens, long[] SeqLens)[] TestScenarios =
    [
        ("prefill", [4L], [4L]),

        // ("prefill*2", [12L, 15L], [12L, 15L]),
        // ("extend", [4L], [8L]),
        // ("prefill+extend", [4L, 4L], [4L, 8L]),
        // ("prefill+decode", [4L, 1L], [4L, 9L]),
    ];

    private static readonly Runtime.TypeCode[] TypeConfigs = [
        Runtime.TypeCode.Float32,
        Runtime.TypeCode.Float16,
    ];

    private static readonly (int NumQ, int NumKV, int Dim)[] HeadConfigs =
    [
        (2, 2, 64),

        // (1, 1, 64),
        // (4, 4, 128),
    ];

    private static readonly (int Layer, int BlockSize, int NumBlocks)[] CacheConfigs = [
        (1, 4, 8),

        // (1, 16, 8),
        // (1, 32, 16),
    ];

    private static readonly (PagedKVCacheDimKind[] Cache, PagedKVCacheDimKind[] Packed)[] LayoutConfigs =
    [
        (new[] {
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.HeadDim,
            PagedKVCacheDimKind.BlockSize,
         },
         new[] { PagedKVCacheDimKind.HeadDim }),
    ];

    private static readonly (PagedKVCacheDimKind[] Sharding, SBPSplit[] Policies, int[] Hierarchy)[] ShardingConfigs =
    [
        (new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, [1]),
    ];

    private static readonly (AttentionDimKind[] QLayout, AttentionDimKind[] KLayout)[] QKLayoutConfigs =
    [
        ([AttentionDimKind.Seq, AttentionDimKind.Dim, AttentionDimKind.Head],
         [AttentionDimKind.Seq, AttentionDimKind.Dim, AttentionDimKind.Head]),
    ];

    public TestUpdatePagedAttentionCase()
    {
        int count = 0;
        foreach (var (name, queryLens, seqLens) in TestScenarios)
        {
            foreach (var (numQHeads, numKVHeads, headDim) in HeadConfigs)
            {
                foreach (var (numLayer, blockSize, numBlocks) in CacheConfigs)
                {
                    foreach (var typeCode in TypeConfigs)
                    {
                        foreach (var (cacheLayout, packedAxes) in LayoutConfigs)
                        {
                            foreach (var (shardingAxes, axisPolicies, hierarchy) in ShardingConfigs)
                            {
                                foreach (var (qlayout, klayout) in QKLayoutConfigs)
                                {
                                    Add(new TestFixture.PagedAttentionKVCacheTestFixture(queryLens, seqLens, numQHeads, numKVHeads, headDim, blockSize, numBlocks, typeCode, numLayer, cacheLayout, packedAxes, shardingAxes, axisPolicies, qlayout, klayout), hierarchy, count++);
                                }
                            }
                        }
                    }
                }
            }
        }

        Add(new TestFixture.PagedAttentionKVCacheTestFixture([256], [256], 14, 2, 64, 256, 16, Runtime.TypeCode.Float32, 1, [PagedKVCacheDimKind.NumBlocks, PagedKVCacheDimKind.NumLayers, PagedKVCacheDimKind.KV, PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.HeadDim, PagedKVCacheDimKind.BlockSize], [PagedKVCacheDimKind.HeadDim], [PagedKVCacheDimKind.NumBlocks], [SBP.S(0)], [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq], [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]), [1], count++);
    }
}

public sealed class TestPagedAttentionCase : TheoryData<TestFixture.PagedAttentionKVCacheTestFixture, int[], int>
{
    private static readonly (string Name, long[] QueryLens, long[] SeqLens)[] TestScenarios =
    [
        ("prefill", [4L], [4L]),

        // ("prefill*2", [12L, 15L], [12L, 15L]),
        // ("extend", [4L], [8L]),
        // ("prefill+extend", [4L, 4L], [4L, 8L]),
        // ("prefill+decode", [4L, 1L], [4L, 9L]),
    ];

    private static readonly Runtime.TypeCode[] TypeConfigs = [
        Runtime.TypeCode.Float32,

        // Runtime.TypeCode.Float16,
    ];

    private static readonly (int NumQ, int NumKV, int Dim)[] HeadConfigs =
    [
        (1, 1, 64),
        (4, 2, 32),

        // (1, 1, 64),
        // (4, 4, 128),
    ];

    private static readonly (int Layer, int BlockSize, int NumBlocks)[] CacheConfigs = [
        (1, 4, 8),

        // (1, 16, 8),
        // (1, 32, 16),
    ];

    private static readonly (PagedKVCacheDimKind[] Cache, PagedKVCacheDimKind[] Packed)[] LayoutConfigs =
    [
        (new[] {
            PagedKVCacheDimKind.NumBlocks,
            PagedKVCacheDimKind.NumLayers,
            PagedKVCacheDimKind.KV,
            PagedKVCacheDimKind.NumKVHeads,
            PagedKVCacheDimKind.HeadDim,
            PagedKVCacheDimKind.BlockSize,
         },
         new[] { PagedKVCacheDimKind.HeadDim }),
    ];

    private static readonly (PagedKVCacheDimKind[] Sharding, SBPSplit[] Policies, int[] Hierarchy)[] ShardingConfigs =
    [
        (new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, [1]),
    ];

    private static readonly (AttentionDimKind[] QLayout, AttentionDimKind[] KLayout)[] QKLayoutConfigs =
    [
        ([AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
         [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]),
    ];

    public TestPagedAttentionCase()
    {
        int count = 0;
        foreach (var (name, queryLens, seqLens) in TestScenarios)
        {
            foreach (var (numQHeads, numKVHeads, headDim) in HeadConfigs)
            {
                foreach (var (numLayer, blockSize, numBlocks) in CacheConfigs)
                {
                    foreach (var typeCode in TypeConfigs)
                    {
                        foreach (var (cacheLayout, packedAxes) in LayoutConfigs)
                        {
                            foreach (var (shardingAxes, axisPolicies, hierarchy) in ShardingConfigs)
                            {
                                foreach (var (qlayout, klayout) in QKLayoutConfigs)
                                {
                                    Add(new TestFixture.PagedAttentionKVCacheTestFixture(queryLens, seqLens, numQHeads, numKVHeads, headDim, blockSize, numBlocks, typeCode, numLayer, cacheLayout, packedAxes, shardingAxes, axisPolicies, qlayout, klayout), hierarchy, count++);
                                }
                            }
                        }
                    }
                }
            }
        }

        Add(new TestFixture.PagedAttentionKVCacheTestFixture([4], [4], 14, 2, 64, 256, 16, Runtime.TypeCode.Float32, 1, [PagedKVCacheDimKind.NumBlocks, PagedKVCacheDimKind.NumLayers, PagedKVCacheDimKind.KV, PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.HeadDim, PagedKVCacheDimKind.BlockSize], [PagedKVCacheDimKind.HeadDim], [PagedKVCacheDimKind.NumBlocks], [SBP.S(0)], [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq], [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]), [1], count++);
    }
}

[CollectionDefinition(nameof(NotThreadSafeResourceCollection), DisableParallelization = true)]
public class NotThreadSafeResourceCollection
{
}

[Collection(nameof(NotThreadSafeResourceCollection))]
[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestCPUKernels : TestClassBase
{
    public UnitTestCPUKernels()
    {
        DefaultTargetName = CPUTarget.Kind;
        CompileOptions.TargetOptions = new NTTTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Compile | Diagnostics.DumpFlags.Schedule | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static Placement DefaultPlacement => new Placement(new[] { 1 }, "t");

    public static int Lane => Vector256.IsHardwareAccelerated ? 32 : 16;

    public static int Rank => 1;

    public static TheoryData<long[], int[], List<int[][]>, int> TestReshardData { get; } = new()
    {
        { [1, 77, 768], [2, 32, 4], new() { new int[][] { [-1, 1], [-1, 1], [0, 2] }, new int[][] { [-1, 2], [-1, 2], [0, 1] } }, 0 },
    };

    public static TheoryData<BinaryOp, long[], long[], int[], int[][], int> TestPackBinaryData { get; } = new()
    {
        { BinaryOp.Add, [8, 2], [8, 2], [1], [], 0 },
        { BinaryOp.Mul, [1, 8, 64, 2 * 8], [1, 1, 64, 2 * 8], [1], [], 1 },
        { BinaryOp.Add, [8, 16], [16], [1], [], 2 },
        { BinaryOp.Mul, [1, 8, 64, 2 * 8], [1, 1, 64, 2 * 8], [4], [[-1], [-1], [0], [-1]], 3 },
    };

    public static TheoryData<ReduceOp, long[], int[], float, bool, int[], int[][], int> TestPackReduceData { get; } = new()
    {
        { ReduceOp.Sum, new long[] { 1, 64, 384, 128 }, new[] { 3 }, 0, true, new[] { 1 }, [], 0 },
        { ReduceOp.Mean, new long[] { 1, 384, 128 }, new[] { 2 }, 0, true, new[] { 1 }, [], 1 },
        { ReduceOp.Mean, new long[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, [[-1], [0], [-1]], 2 },
        { ReduceOp.Max, new long[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, [[-1], [0], [-1]], 3 },
        { ReduceOp.Min, new long[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, [[-1], [0], [-1]], 4 },
        { ReduceOp.Sum, new long[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, [[-1], [0], [-1]], 5 },
        { ReduceOp.Mean, new long[] { 1, 3, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, [[-1], [-1], [-1]], 6 },
        { ReduceOp.Sum, new long[] { 1, 64, 384, 384 }, new[] { 3 }, 0, true, new[] { 64 }, [], 7 },
    };

    [Theory]
    [ClassData(typeof(TestUpdatePagedAttentionCase))]
    public async Task TestUpdatePagedAttentionCase(PagedAttentionKVCacheTestFixture fixture, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.Packing = false;

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var dataGeneratorOptions = new PagedAttentionKVCacheTestFixture.DataGeneratorOptions(Random: true, IncreaseBy: [AttentionDimKind.Head], ResetForKV: true);
        var referenceResults = PagedAttentionKVCacheTestFixture.PrepareReferenceResults(fixture.QueryLens, fixture.SeqLens, fixture.NumQHeads, fixture.Config.NumKVHeads, fixture.Config.HeadDim, fixture.Config.NumLayers, fixture.Config.KVPrimType, dataGeneratorOptions);

        var (root, queryVar, kVVars, kVCacheObjVar) = Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(fixture.QueryLens, fixture.SeqLens, fixture.NumQHeads, fixture.NumBlocks, fixture.QLayout, fixture.KLayout, fixture.Config, new(true));

        var kvinputs = PagedAttentionKVCacheTestFixture.PrepareKVInputs(fixture.QueryLens, fixture.SeqLens, fixture.ContextLens, fixture.NumBlocks, placement, referenceResults, fixture.Config);

        var feedDict = new Dictionary<IVar, IValue>();
        var rtFeedDict = new Dictionary<IVar, IValue>();
        {
            var queryTensor = referenceResults.GetQueryTensor();
            feedDict.Add(queryVar, Value.FromTensor(queryTensor));
            rtFeedDict.Add(queryVar, Value.FromTensor(queryTensor));
            for (int layerId = 0; layerId < fixture.Config.NumLayers; layerId++)
            {
                feedDict.Add(kVVars[layerId][0], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 0)));
                feedDict.Add(kVVars[layerId][1], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 1)));
                rtFeedDict.Add(kVVars[layerId][0], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 0)));
                rtFeedDict.Add(kVVars[layerId][1], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 1)));
            }

            feedDict.Add(kVCacheObjVar, Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kvinputs.KVCacheObj))));

            var kvCacheAddrs = new List<long>();
            {
                var logicalKVShape = kvinputs.KVCacheObj.KVCaches.Dimensions.ToArray();
                foreach (var topoIndices in hierarchy.Select(i => Enumerable.Range(0, i)).CartesianProduct().Select(arr => arr.Select(i => (long)i).ToArray()))
                {
                    var indices = topoIndices.Concat(Enumerable.Repeat(0L, logicalKVShape.Length - hierarchy.Length)).ToArray();
                    var shape = Enumerable.Repeat(1L, hierarchy.Length).Concat(logicalKVShape[hierarchy.Length..]).ToArray();
                    var kvStorage = kvinputs.KVCacheObj.KVCaches.View(indices, shape);

                    // FIXME: Memory leak here
                    unsafe
                    {
                        kvCacheAddrs.Add((long)kvStorage.PinBuffer().Pointer);
                    }
                }
            }

            var rtkvObj = RTPagedAttentionKVCache.Create(
                    kvinputs.KVCacheObj.NumSeqs,
                    kvinputs.KVCacheObj.NumTokens,
                    RTTensor.FromTensor(kvinputs.KVCacheObj.ContextLens),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.SeqLens),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.BlockTables),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.SlotMapping),
                    RTTensor.FromTensor(kvCacheAddrs.ToArray()));
            rtFeedDict.Add(kVCacheObjVar, Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(rtkvObj))));
        }

        await RunCases($"Theory{count}", feedDict, new[] { root }, rtFeedDict);
    }

    [Theory]
    [ClassData(typeof(TestPagedAttentionCase))]
    public async Task TestPagedAttentionCase(PagedAttentionKVCacheTestFixture fixture, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.Packing = false;

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var dataGeneratorOptions = new PagedAttentionKVCacheTestFixture.DataGeneratorOptions(Random: true, IncreaseBy: [AttentionDimKind.Head, AttentionDimKind.Seq], ResetForKV: true);
        var referenceResults = PagedAttentionKVCacheTestFixture.PrepareReferenceResults(fixture.QueryLens, fixture.SeqLens, fixture.NumQHeads, fixture.Config.NumKVHeads, fixture.Config.HeadDim, fixture.Config.NumLayers, fixture.Config.KVPrimType, dataGeneratorOptions);

        var (root, queryVar, kVVars, kVCacheObjVar) = Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(fixture.QueryLens, fixture.SeqLens, fixture.NumQHeads, fixture.NumBlocks, fixture.QLayout, fixture.KLayout, fixture.Config, new());

        var kvinputs = PagedAttentionKVCacheTestFixture.PrepareKVInputs(fixture.QueryLens, fixture.SeqLens, fixture.ContextLens, fixture.NumBlocks, placement, referenceResults, fixture.Config);

        var feedDict = new Dictionary<IVar, IValue>();
        var rtFeedDict = new Dictionary<IVar, IValue>();
        {
            var queryTensor = referenceResults.GetQueryTensor();
            feedDict.Add(queryVar, Value.FromTensor(queryTensor));
            rtFeedDict.Add(queryVar, Value.FromTensor(queryTensor));
            for (int layerId = 0; layerId < fixture.Config.NumLayers; layerId++)
            {
                feedDict.Add(kVVars[layerId][0], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 0)));
                feedDict.Add(kVVars[layerId][1], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 1)));
                rtFeedDict.Add(kVVars[layerId][0], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 0)));
                rtFeedDict.Add(kVVars[layerId][1], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 1)));
            }

            feedDict.Add(kVCacheObjVar, Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kvinputs.KVCacheObj))));
            var kvCacheAddrs = new List<long>();
            {
                var logicalKVShape = kvinputs.KVCacheObj.KVCaches.Dimensions.ToArray();
                foreach (var topoIndices in hierarchy.Select(i => Enumerable.Range(0, i)).CartesianProduct().Select(arr => arr.Select(i => (long)i).ToArray()))
                {
                    var indices = topoIndices.Concat(Enumerable.Repeat(0L, logicalKVShape.Length - hierarchy.Length)).ToArray();
                    var shape = Enumerable.Repeat(1L, hierarchy.Length).Concat(logicalKVShape[hierarchy.Length..]).ToArray();
                    var kvStorage = kvinputs.KVCacheObj.KVCaches.View(indices, shape);

                    // FIXME: Memory leak here
                    unsafe
                    {
                        kvCacheAddrs.Add((long)kvStorage.PinBuffer().Pointer);
                    }
                }
            }

            var rtkvObj = RTPagedAttentionKVCache.Create(
                    kvinputs.KVCacheObj.NumSeqs,
                    kvinputs.KVCacheObj.NumTokens,
                    RTTensor.FromTensor(kvinputs.KVCacheObj.ContextLens),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.SeqLens),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.BlockTables),
                    RTTensor.FromTensor(kvinputs.KVCacheObj.SlotMapping),
                    RTTensor.FromTensor(kvCacheAddrs.ToArray()));
            rtFeedDict.Add(kVCacheObjVar, Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(rtkvObj))));
        }

        await RunCases($"Theory{count}", feedDict, new[] { root }, rtFeedDict);
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 64 }, false, new[] { 64, 48 }, false, new[] { 48, 16 }, true, new[] { 1 }, 0 })]
    [InlineData(new object[] { new[] { 128, 256 }, true, new[] { 256, 384 }, false, new[] { 384, 512 }, true, new[] { 2 }, 1 })]
    [InlineData(new object[] { new[] { 1024, 2048 }, false, new[] { 2048, 1024 }, true, new[] { 1024, 3072 }, true, new[] { 4 }, 2, true })]
    [InlineData(new object[] { new[] { 128, 256 }, true, new[] { 256, 384 }, false, new[] { 384, 512 }, true, new[] { 8 }, 3, false })]
    public async Task TestTileFlowCase(int[] ashape, bool constA, int[] bshape, bool constB, int[] eshape, bool constE, int[] hierarchy, int count, bool packing = false)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.Packing = packing;
        Expr a = constA ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate()) : new Var("a", new TensorType(DataTypes.Float32, ashape));
        Expr b = constB ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate()) : new Var("b", new TensorType(DataTypes.Float32, bshape));
        var c = IR.F.Tensors.MatMul(a, b);
        var d = IR.F.Math.Neg(c);
        Expr e = constE ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate()) : new Var("e", new TensorType(DataTypes.Float32, eshape));
        var f = IR.F.Tensors.MatMul(d, e);

        var feedDict = new Dictionary<IVar, IValue>();
        if (a is Var va)
        {
            feedDict.Add(va, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate());
        }

        if (b is Var vb)
        {
            feedDict.Add(vb, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate());
        }

        if (e is Var ve)
        {
            feedDict.Add(ve, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate());
        }

        await RunCases($"Theory{count}", feedDict, new[] { f });
    }

    [Theory]
    [MemberData(nameof(TestReshardData))]
    public async Task TestReshard(long[] shape, int[] hierarchy, List<int[][]> sbps, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var inputType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(inputType);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, shape).Evaluate() },
        };

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var ndsbps = sbps.Select(sbp => sbp.Select(s => s[0] < 0 ? (SBP)SBP.B : SBP.S(s)).ToArray()).ToArray();
        Expr boxed = input;
        foreach (var ndsbp in ndsbps)
        {
            boxed = IR.F.Distributed.Boxing(boxed, new DistributedType(inputType, ndsbp, placement));
        }

        var post = IR.F.Distributed.Boxing(boxed, inputType);
        post.Metadata = new Passes.Distributed.AutoDistributedMetaData() { Skip = true };
        await RunCases($"Theory{count}", feedDict, new[] { post });
    }

    [Theory]
    [InlineData([new long[] { 32, 64 }, new int[] { 2 }, 0])]
    [InlineData([new long[] { 8, 4 }, new int[] { 4, 2 }, 1])]
    [InlineData([new long[] { 32, 64, 128 }, new int[] { 8, 4, 2 }, 2])]
    [InlineData([new long[] { 64, 128 }, new int[] { 2, 4, 8 }, 3])]
    public async Task TestGatherReduceScatter(long[] shape, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var inputType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(inputType);
        var feedDict = new Dictionary<IVar, IValue>() {
            // { input, IR.F.Tensors.ConstantOfShape(shape, 1.0f).Evaluate() },
            { input, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, shape).Evaluate() },
        };

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var ndsbp = Enumerable.Repeat<SBP>(SBP.B, hierarchy.Length).ToArray();
        var posts = new List<Call>();
        var broadcast = IR.F.Distributed.Boxing(input, new DistributedType(inputType, ndsbp, placement));
        foreach (var comb in LinqUtility.Combination(hierarchy.Length))
        {
            var newsbp = ndsbp.ToArray();
            foreach (var axis in comb)
            {
                newsbp[axis] = SBP.P();
            }

            var partial = IR.F.Distributed.ForceBoxing(broadcast, new DistributedType(inputType, newsbp, placement));
            var sumed = IR.F.Distributed.Boxing(partial, new DistributedType(inputType, ndsbp, placement));
            var post = IR.F.Distributed.Boxing(sumed, inputType);
            post.Metadata = new Passes.Distributed.AutoDistributedMetaData() { Skip = true };
            posts.Add(post);
        }

        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Fact]
    public async Task TestMatmulBinaryBinary()
    {
        var ashape = new[] { 1, 64, 384, 128 };
        var bshape = new[] { 1, 64, 128, 384 };
        var a = new Var("a", new TensorType(DataTypes.Float32, ashape));
        var b = new Var("b", new TensorType(DataTypes.Float32, bshape));
        var c = IR.F.Tensors.MatMul(a, b);
        var dshape = new[] { 1 };
        var d = new Var("d", new TensorType(DataTypes.Float32, dshape));
        var e = IR.F.Math.Binary(BinaryOp.Div, c, d);
        var fshape = new[] { 1, 1, 384, 384 };
        var f = new Var("f", new TensorType(DataTypes.Float32, fshape));
        var g = IR.F.Math.Binary(BinaryOp.Add, e, f);

        var feedDict = new Dictionary<IVar, IValue>() {
            { a, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate() },
            { b, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate() },
            { d, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, dshape).Evaluate() },
            { f, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, fshape).Evaluate() },
        };

        await RunCases(string.Empty, feedDict, new[] { g });
    }

    [Fact]
    public async Task TestDynamicMatmulBinaryBinary()
    {
        var dimM = new DimVar("m");
        dimM.Metadata.Range = new(1, 384 * 2);
        var ashape = new long[] { 1, 64, 384, 128 };
        var bshape = new long[] { 1, 64, 128, 384 };
        var aDims = ashape.Select(x => (Dimension)x).ToArray();
        aDims[^2] = dimM;

        var a = new Var("a", new TensorType(DataTypes.Float32, new RankedShape(aDims)));
        CompileOptions.ShapeBucketOptions.VarMap.Add(a, aDims.Select(x => x).ToArray());
        var b = new Var("b", new TensorType(DataTypes.Float32, bshape));
        var c = IR.F.Tensors.MatMul(a, b);
        var dshape = new[] { 1 };
        var d = new Var("d", new TensorType(DataTypes.Float32, dshape));
        var e = IR.F.Math.Binary(BinaryOp.Div, c, d);
        var fshape = new[] { 1, 1, 384, 384 };
        var f = new Var("f", new TensorType(DataTypes.Float32, fshape));
        var g = IR.F.Math.Binary(BinaryOp.Add, e, f);

        var feedDict = new Dictionary<IVar, IValue>() {
            { a, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate() },
            { b, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate() },
            { d, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, dshape).Evaluate() },
            { f, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, fshape).Evaluate() },
            { dimM, Value.FromTensor(ashape[^2]) },
        };

        await RunCases(string.Empty, feedDict, new[] { g });
    }

    [Theory]
    [InlineData(new object[] { new long[] { 32, 512, 64, 64 }, 0 })]
    public async Task TestSwish(long[] shape, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Swish(input);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackSwish(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 32, 133, 64, 64 }, new[] { 1 }, new[] { 4 }, 0 })]
    [InlineData(new object[] { new long[] { 32, 12, 34, 49 }, new[] { 2, 3 }, new[] { 4 }, 1 })]
    public async Task TestDynamicSwish(long[] shape, int[] dynamicAxes, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var dynShape = new RankedShape(Enumerable.Range(0, shape.Length).Select(i => dynamicAxes.Contains(i) ? new DimVar($"dim{i}")
        {
            Metadata = new() { Range = new(1, Dimension.AlignUp(shape[i] * 2, 64).FixedValue) },
        } : (Dimension)shape[i]).ToArray());
        var input = new Var(new TensorType(DataTypes.Float32, dynShape));
        CompileOptions.ShapeBucketOptions.VarMap.Add(input, dynShape.ToArray());
        var pre = IR.F.NN.Swish(input);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, Value.FromTensor(Tensor.FromScalar(1f, shape)) /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() */ },
        };
        foreach (var axis in dynamicAxes)
        {
            feedDict.Add((DimVar)dynShape[axis], Value.FromTensor(shape[axis]));
        }

        var rule = new Passes.Rules.NTT.PackSwish(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 4, 8, 16, 32 }, new[] { 1 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 64, 384, 128 }, new[] { 4 }, 1 })]
    public async Task TestUnary(long[] shape, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackUnary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 4, 8, 16, 32 }, new[] { 1 }, new int[] { 0 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 64, 384, 128 }, new[] { 4 }, new int[] { 1 }, 1 })]
    [InlineData(new object[] { new long[] { 4, 64, 128, 256 }, new[] { 4 }, new int[] { 2 }, 2 })]
    [InlineData(new object[] { new long[] { 4, 64, 256, 128 }, new[] { 4 }, new int[] { 3 }, 3 })]
    public async Task TestDynamicUnary(long[] shape, int[] hierarchy, int[] dynamicAxes, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 40), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var dimVars = new[] { "n", "c", "h", "w" }.Select((x, i) =>
        {
            var v = new DimVar(x);
            v.Metadata.Range = new(1, shape[i] * 2);
            return v;
        }).ToArray();
        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(Enumerable.Range(0, shape.Length).Select(i => dynamicAxes.Contains(i) ? dimVars[i] : (Dimension)shape[i]).ToArray())));
        CompileOptions.ShapeBucketOptions.VarMap.Add(input, dimVars);

        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, Value.FromTensor(Tensor.FromScalar<float>(1f, shape)) },
            { dimVars[0], Value.FromTensor(shape[0]) },
            { dimVars[1], Value.FromTensor(shape[1]) },
            { dimVars[2], Value.FromTensor(shape[2]) },
            { dimVars[3], Value.FromTensor(shape[3]) },
        };

        var rule = new Passes.Rules.NTT.PackUnary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [MemberData(nameof(TestPackBinaryData))]
    public async Task TestPackBinary(BinaryOp op, long[] lhsShape, long[] rhsShape, int[] hierarchy, int[][] sbps, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackBinary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));

        if (sbps.Length > 0)
        {
            foreach (var post in posts)
            {
                var call = ExprCollector.Collect(post).Where(e => e is Call { Target: IR.NTT.PackedBinary or IR.Math.Binary }).First();
                call.Metadata = new() { OutputNames = new[] { "call" } };
            }

            var scheme = new Passes.Distributed.DistributedSchema("1", "llama", [new("call", sbps.Select(s => s[0] < 0 ? SBP.B : (SBP)SBP.S(s)).ToArray(), hierarchy, targetOptions.HierarchyNames)]);
            var options = new JsonSerializerOptions();
            options.Converters.Add(new SBPConverter());
            options.WriteIndented = true;
            var export = System.Text.Json.JsonSerializer.Serialize(scheme, options);
            var dumpper = Diagnostics.DumpScope.Current.CreateSubDummper($"Theory{count}");
            targetOptions.DistributedScheme = Path.Join(dumpper.Directory, "schema.json");
            using (var stream = dumpper.OpenFile("schema.json"))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(export);
                }
            }
        }

        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Max, new long[] { 56, 1 }, new long[] { 56, 1 }, new int[] { 1 }, new int[] { }, new int[] { 0, 2 }, 0 })] // note max(f32[sequence_length,1],f32[sequence_length,1])
    [InlineData(new object[] { BinaryOp.Div, new long[] { 1 }, new long[] { 36, 1 }, new int[] { 4 }, new int[] { }, new int[] { 1 }, 1 })] // note div(f32[1], f32[sequence_length,1])
    [InlineData(new object[] { BinaryOp.Mul, new long[] { 112, 32 }, new long[] { 112, 1 }, new int[] { 2 }, new int[] { }, new int[] { 0, 2 }, 2 })] // note mul(f32[sequence_length,32], f32[sequence_length,1])
    [InlineData(new object[] { BinaryOp.Mul, new long[] { 66, 64 }, new long[] { 66, 1 }, new int[] { 8 }, new int[] { }, new int[] { 0, 2 }, 3 })] // note mul(f32[sequence_length,64], f32[sequence_length,1])
    [InlineData(new object[] { BinaryOp.Mul, new long[] { 15, 64 }, new long[] { 1, 64 }, new int[] { 4 }, new int[] { }, new int[] { 0 }, 4 })] // note mul(f32[sequence_length,64], const(f32[1,64]))
    [InlineData(new object[] { BinaryOp.Mul, new long[] { 16, 101, 4 }, new long[] { 1, 101, 4 }, new int[] { 4 }, new int[] { }, new int[] { 1, 4 }, 5 })] // note mul(f32[16,sequence_length,4], f32[1,sequence_length,4])
    [InlineData(new object[] { BinaryOp.Add, new long[] { 1 }, new long[] { 32, 28 }, new int[] { 4 }, new int[] { }, new int[] { 2 }, 6 })] // note div(f32[1], f32[32, sequence_length])
    public async Task TestDynamicPackBinary(BinaryOp op, long[] lhsShape, long[] rhsShape, int[] hierarchy, int[] sbps, int[] dynamicAxes, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var dimVar = new DimVar("seq_length")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };

        var lhsDynShape = new RankedShape(Enumerable.Range(0, lhsShape.Length).Select(i => dynamicAxes.Contains(i) ? dimVar : (Dimension)lhsShape[i]).ToArray());
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsDynShape));
        CompileOptions.ShapeBucketOptions.VarMap.Add(lhs, lhsDynShape.ToArray());
        var rhsDynShape = new RankedShape(Enumerable.Range(0, rhsShape.Length).Select(i => dynamicAxes.Contains(lhsDynShape.Rank + i) ? dimVar : (Dimension)rhsShape[i]).ToArray());
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsDynShape));
        CompileOptions.ShapeBucketOptions.VarMap.Add(rhs, rhsDynShape.ToArray());
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
            { dimVar, Value.FromTensor(lhsShape.Concat(rhsShape).Skip(dynamicAxes[0]).First()) },
        };

        var rule = new Passes.Rules.NTT.PackBinary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 113 }, new[] { 0 }, new[] { 2 }, new[] { 0 }, new int[] { }, new int[] { }, 0 })] // note pack(Lanes: {32}, Axes: {1}, [seq_len, 1024])
    [InlineData(new object[] { new long[] { 68, 128 }, new[] { 0, 1 }, new[] { 4 }, new[] { 0 }, new[] { 0 }, new[] { 64 }, 1 })] // note pack(Lanes: {64, 128}, Axes: {0, 1}, [seq_len + padding, 1024])
    [InlineData(new object[] { new long[] { 64, 103 }, new[] { 1 }, new[] { 4 }, new[] { 1 }, new int[] { }, new int[] { }, 2 })] // note pack(Lanes: {32}, Axes: {0}, [64, sequence_length])
    [InlineData(new object[] { new long[] { 1, 99, 128 }, new[] { 1 }, new[] { 4 }, new[] { 1 }, new int[] { }, new int[] { }, 3 })] // note pack(Lanes: {32}, Axes: {2}, [1, sequence_length, 128])
    public async Task TestDynamicPackUnpack(long[] shape, int[] axes, int[] hierarchy, int[] dynamicAxes, int[] alignAxes, int[] alignValues, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var dimVar = new DimVar("seq_len")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };

        var dynShape = new RankedShape(Enumerable.Range(0, shape.Length).Select(i =>
        {
            if (dynamicAxes.Contains(i))
            {
                return dimVar;
            }

            return (Dimension)shape[i];
        }).ToArray());
        var input = new Var(new TensorType(DataTypes.Float32, dynShape));
        CompileOptions.ShapeBucketOptions.VarMap.Add(input, dynShape.ToArray());

        var lanes = axes.Select(i => 32).ToArray();
        for (int i = 0; i < alignAxes.Length; i++)
        {
            lanes[alignAxes[i]] = alignValues[i];
        }

        var paded = PackUtility.PadForPack(input, dynShape, axes, lanes, 0f, out var padNums);
        var packed = IR.F.Tensors.Pack(paded, lanes, axes);
        var unpacked = IR.F.Tensors.Unpack(packed, lanes, axes);
        var sliced = PackUtility.SliceForPack(unpacked, dynShape, padNums);

        // note 2d pack will cause the unpack issue.
        // var inputTensor = Tensor.FromScalar<float>(0, shape);
        // for (int i = 0; i < shape[0]; i++)
        // {
        //     for (int j = 0; j < shape[1]; j++)
        //     {
        //         inputTensor[i, j] = i;
        //     }
        // }
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, /* Value.FromTensor(inputTensor) */ IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { dimVar, Value.FromTensor(shape[dynamicAxes[0]]) },
        };

        await RunCases($"Theory{count}", feedDict, new[] { sliced });
    }

    [Theory(Skip = "Drop InstanceNorm")]
    [InlineData(new object[] { new long[] { 1, 2, 16, 32 }, 1e-5, 0 })]
    [InlineData(new object[] { new long[] { 1, 32, 2048 }, 1e-6, 1 })]
    public async Task TestInstanceNorm(long[] shape, float epsion, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = new[] { shape[1] };
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.InstanceNormalization(input, scale, bias, epsion);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackInstanceNorm(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 4, 32, 32 }, ImageResizeMode.Bilinear, new long[] { 1, 4, 64, 64 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 8, 32, 32 }, ImageResizeMode.NearestNeighbor, new long[] { 1, 8, 64, 64 }, 1 })]
    public async Task TestResizeImage(long[] shape, ImageResizeMode resizeMode, long[] newSize, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Imaging.ResizeImage(resizeMode, input, Array.Empty<float>(), newSize);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackResizeImage(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 256, 64, 64 }, Runtime.TypeCode.Float8E4M3, Runtime.TypeCode.Float32, 0 })]
    [InlineData(new object[] { new long[] { 1, 64, 64, 256 }, Runtime.TypeCode.Float16, Runtime.TypeCode.BFloat16, 1 })]
    [InlineData(new object[] { new long[] { 1, 64, 256, 64 }, Runtime.TypeCode.BFloat16, Runtime.TypeCode.Float16, 2 })]
    [InlineData(new object[] { new long[] { 64 }, Runtime.TypeCode.Float8E4M3, Runtime.TypeCode.Float32, 0 })]
    [InlineData(new object[] { new long[] { 256 }, Runtime.TypeCode.Float16, Runtime.TypeCode.BFloat16, 1 })]
    [InlineData(new object[] { new long[] { 64 }, Runtime.TypeCode.BFloat16, Runtime.TypeCode.Float16, 2 })]
    public async Task TestPackCast(long[] shape, Nncase.Runtime.TypeCode type1, Nncase.Runtime.TypeCode type2, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var casted1 = IR.F.Tensors.Cast(input, DataType.FromTypeCode(type1));
        var casted2 = IR.F.Tensors.Cast(casted1, DataType.FromTypeCode(type2));
        var pre = IR.F.Tensors.Cast(casted2, DataTypes.Float32);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackCast(1, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 384, 512 }, new long[] { 512, 512 }, false, false, new[] { 1 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 1, 384, 256 }, new long[] { 32, 256, 512 }, false, false, new[] { 1 }, 1 })]
    [InlineData(new object[] { new long[] { 384, 512 }, new long[] { 512, 512 }, false, false, new[] { 1 }, 2 })]
    [InlineData(new object[] { new long[] { 1, 384, 512 }, new long[] { 512, 512 }, false, true, new[] { 1 }, 3 })]
    [InlineData(new object[] { new long[] { 1, 1, 384, 256 }, new long[] { 32, 256, 512 }, false, true, new[] { 1 }, 4 })]
    [InlineData(new object[] { new long[] { 384, 512 }, new long[] { 512, 512 }, false, true, new[] { 1 }, 5 })]
    [InlineData(new object[] { new long[] { 384, 512 }, new long[] { 512, 256 }, false, true, new[] { 2 }, 6 })]
    public async Task TestPackMatMul(long[] lhsShape, long[] rhsShape, bool constA, bool constB, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".Skip(3 - hierarchy.Length));
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(lhsShape, 1.0f).Evaluate().AsTensor();
        var rhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(rhsShape, 1.0f).Evaluate().AsTensor();

        // var lhsTensor = Tensor.From(Enumerable.Range(0, (int)TensorUtilities.GetProduct(lhsShape)).Select(i => (float)i).ToArray(), lhsShape);
        // var rhsTensor = Tensor.From(Enumerable.Range(0, (int)TensorUtilities.GetProduct(rhsShape)).Select(i => (float)i).ToArray(), rhsShape);
        Expr lhs = constA ? lhsTensor : new Var(new TensorType(DataTypes.Float32, lhsShape));
        Expr rhs = constB ? rhsTensor : new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>();
        if (!constA)
        {
            feedDict.Add((Var)lhs, Value.FromTensor(lhsTensor));
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, Value.FromTensor(rhsTensor));
        }

        var rule = new Passes.Rules.NTT.PackMatMul(2, Lane, transB: true);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);

        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 154, 128 * 8 }, new long[] { 128 * 8, 64 * 32 }, false, true, new[] { 4 }, new[] { 0 }, 0 })] // note const(f32[sequence_length,2048]) @ [2048,4096]
    [InlineData(new object[] { new long[] { 64, 1 }, new long[] { 1, 94 }, true, false, new[] { 4 }, new[] { 3 }, 1 })] // note const(f32[64,1]) @ [1,sequence_length]
    public async Task TestDynamicPackMatMul(long[] lhsShape, long[] rhsShape, bool constA, bool constB, int[] hierarchy, int[] dynamicAxes, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".Skip(3 - hierarchy.Length));
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var dimVar = new DimVar("seq_len")
        {
            Metadata = new()
            {
                Range = new(1, 256),
            },
        };

        var lhsDynShape = new RankedShape(Enumerable.Range(0, lhsShape.Length).Select(i =>
        {
            if (dynamicAxes.Contains(i))
            {
                return dimVar;
            }

            return (Dimension)lhsShape[i];
        }).ToArray());
        var lhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(lhsShape, 1.0f).Evaluate().AsTensor();
        Expr lhs = constA ? lhsTensor : new Var(new TensorType(DataTypes.Float32, lhsDynShape));

        if (!constA)
        {
            CompileOptions.ShapeBucketOptions.VarMap.Add((Var)lhs, lhs.CheckedShape.ToArray());
        }

        var rhsDynShape = new RankedShape(Enumerable.Range(0, rhsShape.Length).Select(i =>
        {
            if (dynamicAxes.Contains(lhsShape.Length + i))
            {
                return dimVar;
            }

            return (Dimension)rhsShape[i];
        }).ToArray());
        var rhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(rhsShape, 1.0f).Evaluate().AsTensor();
        Expr rhs = constB ? rhsTensor : new Var(new TensorType(DataTypes.Float32, rhsDynShape));
        if (!constB)
        {
            CompileOptions.ShapeBucketOptions.VarMap.Add((Var)rhs, rhs.CheckedShape.ToArray());
        }

        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>();
        foreach (var axis in dynamicAxes)
        {
            feedDict.Add(dimVar, Value.FromTensor(lhsShape.Concat(rhsShape).Skip(axis).Take(1).First()));
        }

        if (!constA)
        {
            feedDict.Add((Var)lhs, Value.FromTensor(lhsTensor));
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, Value.FromTensor(rhsTensor));
        }

        var rule = new Passes.Rules.NTT.PackMatMul(2, Lane, transB: true);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);

        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 384, 512 }, new long[] { 2, 512, 512 }, false, true, new[] { 4, 4 }, 0 })]
    [InlineData(new object[] { new long[] { 2, 384, 512 }, new long[] { 2, 512, 512 }, false, false, new[] { 4, 8 }, 1 })]
    [InlineData(new object[] { new long[] { 2, 384, 512 }, new long[] { 2, 512, 512 }, false, true, new[] { 2, 8, 4 }, 2 })]
    [InlineData(new object[] { new long[] { 2, 384, 512 }, new long[] { 2, 512, 512 }, false, false, new[] { 2, 4, 8 }, 3 })]
    public async Task TestSUMMA(long[] lhsShape, long[] rhsShape, bool constA, bool constB, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".Skip(3 - hierarchy.Length));
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(lhsShape, 1.0f).Evaluate().AsTensor();
        var rhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(rhsShape, 1.0f).Evaluate().AsTensor();

        Expr lhs = constA ? lhsTensor : new Var(new TensorType(DataTypes.Float32, lhsShape));
        Expr rhs = constB ? rhsTensor : new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>();
        if (!constA)
        {
            feedDict.Add((Var)lhs, Value.FromTensor(lhsTensor));
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, Value.FromTensor(rhsTensor));
        }

        var rule = new Passes.Rules.NTT.PackMatMul(2, Lane, transB: false);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);

        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 384, 128 }, 0, new long[] { 1, 384 }, 0 })]
    public async Task TestGather(long[] shape, int axis, long[] indicesShape, int count)
    {
        var vhidden_in = new Var("vhidden_in", new TensorType(DataTypes.Float32, shape));
        var vposition_ids = new Var("vposition_ids", new TensorType(DataTypes.Int64, indicesShape));
        var pre = IR.F.Tensors.Gather(vhidden_in, axis, vposition_ids); // f32[1,384,128]
        var feedDict = new Dictionary<IVar, IValue>() {
            { vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { vposition_ids, IR.F.Random.Uniform(DataTypes.Int64, 6, 1, 1, indicesShape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackGather(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [MemberData(nameof(TestPackReduceData))]
    public async Task TestPackReduce(ReduceOp reduceOp, long[] shape, int[] axes, float init, bool keepDims, int[] hierarchy, int[][] splitedAxes, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var tensorType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(tensorType);
        var pre = IR.F.Tensors.Reduce(reduceOp, input, axes, init, keepDims);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        IEnumerable<BaseExpr> posts;
        var rule = new Passes.Rules.NTT.PackReduce(Rank, Lane);
        if (!CompilerServices.TryMatch(pre, rule.Pattern, out var result))
        {
            return;
        }

        posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result, new Passes.RunPassContext()));

        if (splitedAxes.Length > 0)
        {
            foreach (var post in posts)
            {
                if (post is Call { Target: IR.Tensors.Unpack } callUnPack && callUnPack.Arguments[0] is Call { Target: IR.NTT.PackedReduce } packedReduceCall)
                {
                    packedReduceCall.Arguments[0].Metadata = new() { OutputNames = new[] { "reduceIn" } };
                }
                else if (post is Call { Target: IR.Math.Reduce } reduceCall)
                {
                    reduceCall.Arguments[0].Metadata = new() { OutputNames = new[] { "reduceIn" } };
                }
            }

            var scheme = new Passes.Distributed.DistributedSchema("1", "llama", [new("reduceIn", splitedAxes.Select(s => s[0] < 0 ? SBP.B : (SBP)SBP.S(s)).ToArray(), hierarchy, targetOptions.HierarchyNames)]);
            var options = new JsonSerializerOptions();
            options.Converters.Add(new SBPConverter());
            options.WriteIndented = true;
            var export = System.Text.Json.JsonSerializer.Serialize(scheme, options);
            var dumpper = Diagnostics.DumpScope.Current.CreateSubDummper($"Theory{number}");
            targetOptions.DistributedScheme = Path.Join(dumpper.Directory, "schema.json");
            using (var stream = dumpper.OpenFile("schema.json"))
            {
                using (var writer = new StreamWriter(stream))
                {
                    writer.Write(export);
                }
            }
        }

        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 3, 28, 28 }, 0 })]
    public async Task TestInstanceNormal(long[] shape, int number)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr pre; // f32[1,3,28,28]
        {
            var v0 = IR.F.Tensors.Reduce(ReduceOp.Mean, input, new[] { 2, 3 }, 0f, true); // f32[1,3,1,1]
            var v1 = IR.F.Math.Binary(BinaryOp.Sub, input, v0); // f32[1,3,28,28]
            var v2 = IR.F.Math.Unary(UnaryOp.Square, v1); // f32[1,3,28,28]
            var v3 = IR.F.Tensors.Reduce(ReduceOp.Mean, v2, new[] { 2, 3 }, 0f, true); // f32[1,3,1,1]
            var v4 = IR.F.Math.Binary(BinaryOp.Add, v3, new float[] { 1E-05f }); // f32[1,3,1,1]
            var v5 = IR.F.Math.Unary(UnaryOp.Rsqrt, v4); // f32[1,3,1,1]
            var v6 = IR.F.Math.Binary(BinaryOp.Mul, v1, v5); // f32[1,3,28,28]
            var v7 = IR.F.Math.Binary(BinaryOp.Mul, v6, new float[3, 1, 1] { { { 0.24680786f } }, { { 0.065782584f } }, { { -0.9344868f } } }); // f32[1,3,28,28]
            pre = IR.F.Math.Binary(BinaryOp.Add, v7, new float[3, 1, 1] { { { 0.6403651f } }, { { -0.7995949f } }, { { 0.46802735f } } }); // f32[1,3,28,28]
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData([new long[] { 1, 384, 8192 }, new long[] { 1, 384, 64, 128 }, 1, new[] { 1 }, 0])]
    [InlineData([new long[] { 1, 8192, 384 }, new long[] { 1, 64, 128, 384 }, 1, new[] { 1 }, 1])]
    [InlineData([new long[] { 1, 8192, 384 }, new long[] { 1, 64, 128, 384 }, 1, new[] { 8 }, 2])]
    public async Task TestPackReshape(long[] inshape, long[] outshape, int packRank, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var input = new Var("input", new TensorType(DataTypes.Float32, inshape));
        Expr pre;
        {
            pre = IR.F.Tensors.Reshape(input, outshape);
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inshape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackReshape(packRank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData([new long[] { 2, 8, 16, 2 }, new int[] { 0, 2, 1, 3 }, 2, 0])]
    [InlineData([new long[] { 1, 64, 384, 128 }, new int[] { 0, 2, 1, 3 }, 2, 1])]
    public async Task TestTranspose(long[] shape, int[] perm, int rank, int number)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr pre; // f32[1,3,28,28]
        {
            var v4 = IR.F.Tensors.Transpose(input, perm); // f32[1,64,384,128]
            pre = v4;
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, Value.FromTensor(Tensor.From(Enumerable.Range(0, (int)TensorUtilities.GetProduct(shape)).Select(i => (float)i).ToArray(), shape)) },
        };

        var rule = new Passes.Rules.NTT.PackTranspose(rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData([new[] { 2, 4 }, 0])]
    public async Task TestTransposeMatmul(int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Packing = true;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var v13 = new Var("v13", new TensorType(DataTypes.Float32, new[] { 1, 1, 384, 128 }));
        var v15 = new Var("v15", new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 128 }));
        var v19 = new Var("v19", new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 128 }));
        var v24 = new Var("v24", new TensorType(DataTypes.Float32, new[] { 1, 64, 384, 128 }));
        Expr pre; // f32[1,3,28,28]
        {
            var v25 = IR.F.Math.Binary(BinaryOp.Mul, v24, v13); // f32[1,64,384,128]
            var v26 = IR.F.Math.Binary(BinaryOp.Add, v19, v25); // f32[1,64,384,128]
            var v27 = IR.F.Tensors.Transpose(v26, new[] { 0L, 1L, 3L, 2L }); // f32[1,64,128,384]
            var v28 = IR.F.Math.MatMul(v15, v27); // f32[1,64,384,384]
            pre = v28;
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { v13, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v13.CheckedShape).Evaluate() },
            { v15, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v15.CheckedShape).Evaluate() },
            { v19, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v19.CheckedShape).Evaluate() },
            { v24, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v24.CheckedShape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 1, 4, 4 }, new long[] { 8, 1, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, 0 })]
    [InlineData(new object[] { new long[] { 3, 2, 4, 4 }, new long[] { 8, 2, 3, 3 }, new int[] { 0, 0, 1, 1 }, new int[] { 1, 2 }, 1 })]
    [InlineData(new object[] { new long[] { 3, 2, 4, 4 }, new long[] { 8, 2, 3, 3 }, new int[] { 1, 0, 1, 1 }, new int[] { 2, 1 }, 2 })]
    [InlineData(new object[] { new long[] { 1, 512, 64, 64 }, new long[] { 512, 512, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, 3 })]
    public async Task TestConv2DAndIm2col(long[] inputShape, long[] wShape, int[] padding, int[] strides, int count)
    {
        var dilation = new[] { 1, 1 };
        var groups = 1;
        var input = new Var(new TensorType(DataTypes.Float32, inputShape));
        var weights = new Var(new TensorType(DataTypes.Float32, wShape));
        var bias = IR.F.Random.Normal(DataTypes.Float32, new[] { wShape[0] }).Evaluate().AsTensor();
        var pre = IR.F.NN.Conv2D(input, weights, bias, strides, new[,] { { padding[0], padding[1] }, { padding[2], padding[3] } }, dilation, PadMode.Constant, groups);
        var outShape = pre.CheckedShape.ToValueArray();

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate() },
            { weights, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, wShape).Evaluate() },
        };

        Expr post = Passes.Rules.NTT.PackConv2D.AddCandidate(input, weights, bias, strides, padding, wShape, outShape);
        Expr post2 = Passes.Rules.NTT.PackConv2D.AddPackedCandidate(input, weights, bias, strides, padding, wShape, outShape, Lane);
        var posts = new[] { pre, post, post2 };
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, 48, 64, 16 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 8 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, 64, 768 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 8 }, 1 })]
    public async Task TestMatMulReshapeUnary(long[] lhsShape, long[] rhsShape, long[] newShape, UnaryOp[] unaryOps, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        var reshaped = IR.F.Tensors.Reshape(matmul, newShape);
        var unary = reshaped;
        foreach (var item in unaryOps)
        {
            unary = IR.F.Math.Unary(item, unary);
        }

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
        };

        await RunCases($"Theory{number}", feedDict, new[] { unary });
    }

    [Theory]
    [InlineData([new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new[] { 8 }, 0])]
    public async Task TestPackPropagation(long[] lhsShape, long[] rhsShape, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
        };

        var candidates = new[] {
            IR.F.Math.Unary(UnaryOp.Abs, lhs),
            IR.F.Math.Binary(BinaryOp.Add, lhs, 1f),
            IR.F.Tensors.Unsqueeze(lhs, new[] { 0 }),
        };
        var posts = new List<BaseExpr>();

        foreach (var c in candidates)
        {
            var matmul = IR.F.Tensors.MatMul(c, rhs);

            var rule = new Passes.Rules.NTT.PackMatMul(2, Lane);
            CompilerServices.TryMatch(matmul, rule.Pattern, out var result);
            var context = new Passes.RunPassContext();
            var packed = rule.GetReplaceCandidates(result!, context);
            var rules = new IRewriteRule[] {
                new Nncase.Passes.Rules.NTT.PackUnaryPropagation(),
                new Nncase.Passes.Rules.NTT.PackBinaryPropagation(),
                new Nncase.Passes.Rules.NTT.PackUnsqueezePropagation(),
            };
            posts.AddRange(packed.Select(ret => CompilerServices.Rewrite(ret, rules, context)));
        }

        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData([new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new[] { 8 }, 0])]
    public async Task TestUnpackPropagation(long[] lhsShape, long[] rhsShape, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackMatMul(2, Lane);
        CompilerServices.TryMatch(matmul, rule.Pattern, out var result);
        var context = new Passes.RunPassContext();
        var packed = rule.GetReplaceCandidates(result!, context).Cast<Expr>();
        var posts = packed.Select(ret => CompilerServices.Rewrite(IR.F.Math.Unary(UnaryOp.Abs, ret), [new Nncase.Passes.Rules.NTT.UnaryUnpackPropagation()], context)).ToList();
        posts.AddRange(packed.Select(ret => CompilerServices.Rewrite(IR.F.Math.Binary(BinaryOp.Add, ret, 1f), [new Nncase.Passes.Rules.NTT.BinaryUnpackLhsPropagation()], context)));
        posts.AddRange(packed.Select(ret => CompilerServices.Rewrite(IR.F.Tensors.Transpose(ret, new[] { 0, 2, 1 }), [new Nncase.Passes.Rules.NTT.TransposeUnpackPropagation()], context)));
        posts.AddRange(packed.Select(ret => CompilerServices.Rewrite(IR.F.Tensors.Unsqueeze(ret, new[] { 2 }), [new Nncase.Passes.Rules.NTT.UnsqueezeUnpackPropagation()], context)));
        posts.AddRange(packed.Select(ret => CompilerServices.Rewrite(IR.F.Tensors.Reduce(ReduceOp.Max, ret, new[] { 2 }, 0f, true), [new Nncase.Passes.Rules.NTT.ReduceUnpackPropagation()], context)));
        await RunCases($"Theory{number}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, -1, 64, 16 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 1 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, -1, 768 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 1 }, 1 })]
    public async Task TestDynamicMatMulReshapeUnary(long[] lhsShape, long[] rhsShape, long[] newShape, UnaryOp[] unaryOps, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var dimM = new DimVar("m");
        dimM.Metadata.Range = new(1, 48);
        var lhsDims = lhsShape.Select(x => (Dimension)x).ToArray();
        lhsDims[^2] = dimM;

        var lhs = new Var(new TensorType(DataTypes.Float32, new RankedShape(lhsDims)));
        CompileOptions.ShapeBucketOptions.VarMap.Add(lhs, lhsDims.Select(x => x).ToArray());
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        var reshaped = IR.F.Tensors.Reshape(matmul, newShape);
        var unary = reshaped;
        foreach (var item in unaryOps)
        {
            unary = IR.F.Math.Unary(item, unary);
        }

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
            { dimM, Value.FromTensor(lhsShape[^2]) },
        };

        await RunCases($"Theory{number}", feedDict, new[] { unary });
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, 48, 64, 16 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 1 }, 0 })]
    public async Task TestReshapeAndUnsqueeze(long[] lhsShape, long[] rhsShape, long[] newShape, UnaryOp[] unaryOps, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var lhsDims = lhsShape.Select(x => (Dimension)x).ToArray();

        var lhs = new Var(new TensorType(DataTypes.Float32, new RankedShape(lhsDims)));
        CompileOptions.ShapeBucketOptions.VarMap.Add(lhs, lhsDims.Select(x => x).ToArray());
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        var reshaped = IR.F.Tensors.Reshape(matmul, newShape);
        var unary = reshaped;
        foreach (var item in unaryOps)
        {
            unary = IR.F.Math.Unary(item, unary);
        }

        var unsqueezed = IR.F.Tensors.Unsqueeze(unary, new RankedShape(0));

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
        };

        await RunCases($"Theory{number}", feedDict, new[] { unsqueezed });
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 48, 512 }, new long[] { 1, 512, 1024 }, new long[] { 1, 48, -1, 16 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 1 }, 0 })]
    public async Task TestDynamicReshapeAndUnsqueeze(long[] lhsShape, long[] rhsShape, long[] newShape, UnaryOp[] unaryOps, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var dimN = new DimVar("n");
        dimN.Metadata.Range = new(1, 1200);
        var rhsDims = rhsShape.Select(x => (Dimension)x).ToArray();
        rhsDims[^1] = dimN;

        var lhs = new Var(new TensorType(DataTypes.Float32, new RankedShape(lhsShape)));
        var rhs = new Var(new TensorType(DataTypes.Float32, new RankedShape(rhsDims)));
        CompileOptions.ShapeBucketOptions.VarMap.Add(rhs, rhsDims.Select(x => x).ToArray());
        var matmul = IR.F.Tensors.MatMul(lhs, rhs);
        var reshaped = IR.F.Tensors.Reshape(matmul, newShape);
        var unary = reshaped;
        foreach (var item in unaryOps)
        {
            unary = IR.F.Math.Unary(item, unary);
        }

        var unsqueezed = IR.F.Tensors.Unsqueeze(unary, new RankedShape(0));

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
            { dimN, Value.FromTensor(rhsShape[^1]) },
        };

        await RunCases($"Theory{number}", feedDict, new[] { unsqueezed });
    }

    [Theory]
    [InlineData(new object[] { new long[] { 2, 48, 512 }, new long[] { 0 }, new[] { 1 }, 0 })]
    public async Task TestGetItem(long[] inShape, long[] indices, int[] hierarchy, int number)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var inDims = inShape.Select(x => (Dimension)x).ToArray();

        var input = new Var(new TensorType(DataTypes.Float32, new RankedShape(inDims)));
        CompileOptions.ShapeBucketOptions.VarMap.Add(input, inDims.Select(x => x).ToArray());

        var output = IR.F.Tensors.GetItem(input, indices);
        output = IR.F.Math.Unary(UnaryOp.Cos, output);

        var feedDict = new Dictionary<IVar, IValue>()
        {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inShape).Evaluate() },
        };

        await RunCases($"Theory{number}", feedDict, new[] { output });
    }

    [Theory(Skip = "ToBig")]
    [InlineData(new object[] { false, 0 })]
    [InlineData(new object[] { true, 1 })] // enable packing
    public async Task TestDecodeLayer(bool packing, int count)
    {
        // Memory usage is too high for CI env
        if (bool.TryParse(Environment.GetEnvironmentVariable("CI"), out var inCI) && inCI)
        {
            return;
        }

        ((NTTTargetOptions)CompileOptions.TargetOptions).Packing = packing;
        var hierarchy = new[] { 2, 4 };
        ((NTTTargetOptions)CompileOptions.TargetOptions).Hierarchies[0] = hierarchy;
        ((NTTTargetOptions)CompileOptions.TargetOptions).HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        ((NTTTargetOptions)CompileOptions.TargetOptions).HierarchySizes = Enumerable.Repeat((long)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        var vhidden_in = new Var("vhidden_in", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));
        var vattn_mask = new Var("vattn_mask", new TensorType(DataTypes.Float32, new[] { 1, 1, 384, 384 }));
        var vposition_ids = new Var("vposition_ids", new TensorType(DataTypes.Int64, new[] { 1, 384 }));
        Expr pre;
        {
            var v0 = IR.F.NN.LayerNorm(2, 1E-05f, vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 1, new[] { 8192 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 2, new[] { 8192 }).Evaluate().AsTensor(), false); // f32[1,384,8192]
            var v1 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 3, new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v2 = IR.F.Tensors.Reshape(v1, new long[] { 1L, 384L, 64L, 128L }); // f32[1,384,64,128]
            var v3 = IR.F.Tensors.Transpose(v2, new long[] { 0L, 2L, 1L, 3L }); // f32[1,64,384,128]
            var v4 = IR.F.Tensors.Gather(IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 4, new[] { 384, 128 }).Evaluate().AsTensor(), 0, vposition_ids); // f32[1,384,128]
            var v5 = IR.F.Tensors.Reshape(v4, new[] { 1, 1, 384, 128 }); // f32[1,1,384,128]
            var v6 = IR.F.Math.Binary(BinaryOp.Mul, v3, v5); // f32[1,64,384,128]
            var v7 = IR.F.Tensors.Slice(v3, new long[] { 64L }, new long[] { 128L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v8 = IR.F.Math.Unary(UnaryOp.Neg, v7); // f32[1,64,384,64]
            var v9 = IR.F.Tensors.Slice(v3, new long[] { 0L }, new long[] { 64L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v10 = new IR.Tuple(v8, v9); // (f32[1,64,384,64], f32[1,64,384,64])
            var v11 = IR.F.Tensors.Concat(v10, 3); // f32[1,64,384,128]
            var v12 = IR.F.Tensors.Gather(IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 5, new[] { 384, 128 }).Evaluate().AsTensor(), 0, vposition_ids); // f32[1,384,128]
            var v13 = IR.F.Tensors.Reshape(v12, new[] { 1, 1, 384, 128 }); // f32[1,1,384,128]
            var v14 = IR.F.Math.Binary(BinaryOp.Mul, v11, v13); // f32[1,64,384,128]
            var v15 = IR.F.Math.Binary(BinaryOp.Add, v6, v14); // f32[1,64,384,128]
            var v16 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 6, new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v17 = IR.F.Tensors.Reshape(v16, new long[] { 1L, 384L, 64L, 128L }); // f32[1,384,64,128]
            var v18 = IR.F.Tensors.Transpose(v17, new long[] { 0L, 2L, 1L, 3L }); // f32[1,64,384,128]
            var v19 = IR.F.Math.Binary(BinaryOp.Mul, v18, v5); // f32[1,64,384,128]
            var v20 = IR.F.Tensors.Slice(v18, new long[] { 64L }, new long[] { 128L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v21 = IR.F.Math.Unary(UnaryOp.Neg, v20); // f32[1,64,384,64]
            var v22 = IR.F.Tensors.Slice(v18, new long[] { 0L }, new long[] { 64L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v23 = new IR.Tuple(v21, v22); // (f32[1,64,384,64], f32[1,64,384,64])
            var v24 = IR.F.Tensors.Concat(v23, 3); // f32[1,64,384,128]
            var v25 = IR.F.Math.Binary(BinaryOp.Mul, v24, v13); // f32[1,64,384,128]
            var v26 = IR.F.Math.Binary(BinaryOp.Add, v19, v25); // f32[1,64,384,128]
            var v27 = IR.F.Tensors.Transpose(v26, new long[] { 0L, 1L, 3L, 2L }); // f32[1,64,128,384]
            var v28 = IR.F.Tensors.MatMul(v15, v27); // f32[1,64,384,384]
            var v29 = IR.F.Math.Binary(BinaryOp.Div, v28, new[] { 11.31370f }); // f32[1,64,384,384]
            var v30 = IR.F.Math.Binary(BinaryOp.Add, v29, vattn_mask); // f32[1,64,384,384]
            var v31 = IR.F.NN.Softmax(v30, 3); // f32[1,64,384,384]
            var v32 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 7, new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v33 = IR.F.Tensors.Reshape(v32, new long[] { 1L, 384L, 64L, 128L }); // f32[1,384,64,128]
            var v34 = IR.F.Tensors.Transpose(v33, new long[] { 0L, 2L, 1L, 3L }); // f32[1,64,384,128]
            var v35 = IR.F.Tensors.MatMul(v31, v34); // f32[1,64,384,128]
            var v36 = IR.F.Tensors.Transpose(v35, new long[] { 0L, 2L, 1L, 3L }); // f32[1,384,64,128]
            var v37 = IR.F.Tensors.Reshape(v36, new long[] { 1L, 384L, 8192L }); // f32[1,384,8192]
            var v38 = IR.F.Tensors.MatMul(v37, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 8, new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v39 = IR.F.Math.Binary(BinaryOp.Add, vhidden_in, v38); // f32[1,384,8192]
            var v40 = IR.F.NN.LayerNorm(2, 1E-05f, v39, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 9, new[] { 8192 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 2, new[] { 8192 }).Evaluate().AsTensor(), false); // f32[1,384,8192]
            var v41 = IR.F.Tensors.MatMul(v40, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 10, new[] { 8192, 22016 }).Evaluate().AsTensor()); // f32[1,384,22016]
            var v42 = IR.F.NN.Swish(v41, 1.0f); // f32[1,384,22016]
            var v43 = IR.F.Tensors.MatMul(v40, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 11, new[] { 8192, 22016 }).Evaluate().AsTensor()); // f32[1,384,22016]
            var v44 = IR.F.Math.Binary(BinaryOp.Mul, v42, v43); // f32[1,384,22016]
            var v45 = IR.F.Tensors.MatMul(v44, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 12, new[] { 22016, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v46 = IR.F.Math.Binary(BinaryOp.Add, v39, v45); // f32[1,384,8192]
            pre = v46;
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 13,  new[] { 1, 384, 8192 }).Evaluate() },
            { vattn_mask, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 14,  new[] { 1, 1, 384, 384 }).Evaluate() },
            { vposition_ids, IR.F.Random.Uniform(DataTypes.Int64, 383, 1, 15, new[] { 1, 384 }).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]

    // [InlineData(new object[] { false, 0 })]
    [InlineData(new object[] { true, 1 })] // enable packing
    public async Task TestVAEDecRes(bool packing, int count)
    {
        CompileOptions.TargetOptions = new NTTTargetOptions() { Packing = packing };
        var vlatent_sample = new Var("vlatent_sample", new TensorType(DataTypes.Float32, new[] { 1, 4, 64, 64 }));
        Expr pre;
        {
            var v0 = IR.F.NN.Conv2D(vlatent_sample, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 1, new[] { 4, 4, 1, 1 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 2, new[] { 4 }).Evaluate().AsTensor(), new[] { 1L, 1L }, new[,] { { 0L, 0L }, { 0L, 0L } }, new[] { 1L, 1L }, PadMode.Constant, 1L, new[] { float.NegativeInfinity, float.PositiveInfinity }); // f32[1,4,64,64]
            var v1 = IR.F.NN.Conv2D(v0, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 3, new[] { 512, 4, 3, 3 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 4, new[] { 512 }).Evaluate().AsTensor(), new[] { 1L, 1L }, new[,] { { 1L, 1L }, { 1L, 1L } }, new[] { 1L, 1L }, PadMode.Constant, 1L, new[] { float.NegativeInfinity, float.PositiveInfinity }); // f32[1,512,64,64]
            var v2 = IR.F.Tensors.Reshape(v1, new[] { 1L, 32L, 65536L }); // f32[1,32,65536]
            var v3 = IR.F.NN.InstanceNormalization(v2, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 5, new[] { 32 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 6, new[] { 32 }).Evaluate().AsTensor(), 1E-06f); // f32[1,32,65536]
            var v4 = IR.F.Tensors.Reshape(v3, new[] { 1L, 512L, 64L, 64L }); // f32[1,512,64,64]
            var v5 = IR.F.Math.Binary(BinaryOp.Mul, v4, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 7, new[] { 512, 1, 1 }).Evaluate().AsTensor()); // f32[1,512,64,64]
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 8, new[] { 512, 1, 1 }).Evaluate().AsTensor()); // f32[1,512,64,64]
            var v7 = IR.F.NN.Swish(v6, 1f); // f32[1,512,64,64]
            var v8 = IR.F.NN.Conv2D(v7, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 9, new[] { 512, 512, 3, 3 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 10, new[] { 512 }).Evaluate().AsTensor(), new[] { 1L, 1L }, new[,] { { 1L, 1L }, { 1L, 1L } }, new[] { 1L, 1L }, PadMode.Constant, 1L, new[] { float.NegativeInfinity, float.PositiveInfinity }); // f32[1,512,64,64]
            var v9 = IR.F.Tensors.Reshape(v8, new[] { 1L, 32L, 65536L }); // f32[1,32,65536]
            var v10 = IR.F.NN.InstanceNormalization(v9, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 11, new[] { 32 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 12, new[] { 32 }).Evaluate().AsTensor(), 1E-06f); // f32[1,32,65536]
            var v11 = IR.F.Tensors.Reshape(v10, new[] { 1L, 512L, 64L, 64L }); // f32[1,512,64,64]
            var v12 = IR.F.Math.Binary(BinaryOp.Mul, v11, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 13, new[] { 512, 1, 1 }).Evaluate().AsTensor()); // f32[1,512,64,64]
            var v13 = IR.F.Math.Binary(BinaryOp.Add, v12, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 14, new[] { 512, 1, 1 }).Evaluate().AsTensor()); // f32[1,512,64,64]
            var v14 = IR.F.NN.Swish(v13, 1f); // f32[1,512,64,64]
            var v15 = IR.F.NN.Conv2D(v14, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 15, new[] { 512, 512, 3, 3 }).Evaluate().AsTensor(), IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 16, new[] { 512 }).Evaluate().AsTensor(), new[] { 1L, 1L }, new[,] { { 1L, 1L }, { 1L, 1L } }, new[] { 1L, 1L }, PadMode.Constant, 1L, new[] { float.NegativeInfinity, float.PositiveInfinity }); // f32[1,512,64,64]
            pre = IR.F.Math.Binary(BinaryOp.Add, v1, v15); // f32[1,512,64,64]
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { vlatent_sample, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 13,  new[] { 1, 4, 64, 64 }).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 33, 512 }, new long[] { 512, 255 }, false, false, new[] { 8 }, 0 })]
    public async Task TestNonUiniformDistMatmul(long[] lhsShape, long[] rhsShape, bool constA, bool constB, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".Skip(3 - hierarchy.Length));
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var lhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(lhsShape, 1.0f).Evaluate().AsTensor();
        var rhsTensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate().AsTensor(); // IR.F.Tensors.ConstantOfShape(rhsShape, 1.0f).Evaluate().AsTensor();

        Expr lhs = constA ? lhsTensor : new Var(new TensorType(DataTypes.Float32, lhsShape));
        Expr rhs = constB ? rhsTensor : new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>();
        if (!constA)
        {
            feedDict.Add((Var)lhs, Value.FromTensor(lhsTensor));
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, Value.FromTensor(rhsTensor));
        }

        var rule = new Passes.Rules.NTT.PackMatMul(2, Lane, transB: false);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);

        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 4, 4, 255 }, new[] { 8 }, 0 })]
    public async Task TestNonUiniformDistUnary(long[] shape, int[] hierarchy, int count)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);
        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackUnary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { CompareOp.LowerThan, new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 0 })]
    public async Task TestPackCompare(CompareOp op, long[] lhsShape, long[] rhsShape, int count)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Compare(op, lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var maskVectorStyle = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 or Architecture.Arm64 => MaskVectorStyle.Fat,
            _ => throw new NotSupportedException($"Unsupported architecture: {RuntimeInformation.ProcessArchitecture}"),
        };
        var rule = new Passes.Rules.NTT.PackCompare(maskVectorStyle, Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 16, 1, 32 }, new long[] { 1, 16, 32, 32 }, 0 })]
    [InlineData(new object[] { new long[] { 1, 1, 32, 32 }, new long[] { 1, 16, 32, 32 }, 1 })]
    public async Task TestPackExpand(long[] shape, long[] newShape, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Tensors.Expand(input, newShape);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackExpand(1, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 0 })]
    [InlineData(new object[] { new long[] { 1 }, new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 1 })]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1 }, new long[] { 1, 8, 64, 16 }, 2 })]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1, 1, 64, 16 }, new long[] { 1, 8, 64, 16 }, 3 })]
    public async Task TestPackWhere(long[] condShape, long[] lhsShape, long[] rhsShape, int count)
    {
        var cond = new Var(new TensorType(DataTypes.Boolean, condShape));
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.Where(cond, lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { cond, IR.F.Random.Normal(DataTypes.Boolean, 0, 1, 1, condShape).Evaluate() },
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var maskVectorStyle = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 or Architecture.Arm64 => MaskVectorStyle.Fat,
            _ => throw new NotSupportedException($"Unsupported architecture: {RuntimeInformation.ProcessArchitecture}"),
        };
        var rule = new Passes.Rules.NTT.PackWhere(maskVectorStyle, Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 1, 0 })]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 2, 1 })]
    [InlineData(new object[] { new long[] { 1, 8, 64, 16 }, new long[] { 1, 8, 64, 16 }, 3, 2 })]
    public async Task TestPackConcat(long[] inShape1, long[] inShape2, int axis, int count)
    {
        var input1 = new Var(new TensorType(DataTypes.Float32, inShape1));
        var input2 = new Var(new TensorType(DataTypes.Float32, inShape2));
        var pre = IR.F.Tensors.Concat(new IR.Tuple(input1, input2), axis);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input1, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inShape1).Evaluate() },
            { input2, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, inShape2).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackConcat(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new long[] { 16, 16, 16 }, new long[] { 2, 1 }, new long[] { 16, 16, 16 }, 0 })]
    [InlineData(new object[] { new long[] { 16, 16, 16 }, new long[] { 3, 2 }, new long[] { 16, 16 }, 1 })]
    [InlineData(new object[] { new long[] { 16, 16, 256, 256 }, new long[] { 16, 16, 256, 256, 4 }, new long[] { 16, 16, 256, 256 }, 2 })]
    public async Task TestPackScatterND(long[] inShape, long[] indicesShape, long[] updatesShape, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, inShape));
        var indices = IR.F.Random.Uniform(DataTypes.Int64, 15, 0, 1, indicesShape).Evaluate().AsTensor();
        var updates = new Var(new TensorType(DataTypes.Float32, updatesShape));
        var pre = IR.F.Tensors.ScatterND(input, indices, updates);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inShape).Evaluate() },
            { updates, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, updatesShape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackScatterND(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases($"Theory{count}", feedDict, posts);
    }

    internal async Task RunCases(string dumpDir, Dictionary<IVar, IValue> feedDict, IEnumerable<BaseExpr> posts, Dictionary<IVar, IValue>? feedDictRT = null)
    {
        var postArray = posts.ToArray();
        using var pinner = new ExprPinner(postArray);
        for (int i = 0; i < postArray.Length; i++)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(postArray[i]));
#endif
            var kernelCase = new CpuKernelCase($"Case{i}", new Fusion("kernel", CPUTarget.Kind, postArray[i], feedDict.Keys.ToArray()), feedDict.Keys.ToArray(), feedDict.Values.Select(v => v.AsTensor()).ToArray(), feedDictRT?.Values.Select(v => v.AsTensor()).ToArray() ?? []);
            await Run(dumpDir, kernelCase);
        }
    }

    internal async Task Run(string dumpDir, CpuKernelCase kernelCase)
    {
        using var dumpScope = new Diagnostics.DumpScope(Path.Join(dumpDir, kernelCase.Name), CompileOptions.DumpFlags);

        // convert fusion to prim func
        var fusion = kernelCase.Fusion;
        if (fusion.Body.CheckedType is InvalidType)
        {
            return;
        }

        var main = new Function(fusion.Body, kernelCase.Vars.ToArray());
        main.Metadata = fusion.Body.Metadata;

        var module = new IR.IRModule(main);
        var inputs = kernelCase.Inputs.ToArray();
        var outputs = fusion.Body.Evaluate(kernelCase.Vars.Zip(inputs).ToDictionary(p => p.First, p => (IValue)Value.FromTensor(p.Second))).AsTensors();

#if DEBUG
        for (var i = 0; i < inputs.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"input_{i}.json"))
            {
                JsonSerializer.Serialize(fs, inputs[i], JsonSerializerOptions.Default);
            }
        }

        for (int i = 0; i < outputs.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"output_{i}.json"))
            {
                JsonSerializer.Serialize(fs, outputs[i], JsonSerializerOptions.Default);
            }
        }
#endif
        await Compile(module);
        var (kmodel_path, _) = Testing.BuildKModel("test", module, CompileSession, false);
        Tensor[] actuals;
        if (kernelCase.RTInputs.Any())
        {
            actuals = Testing.RunKModel(kmodel_path, Diagnostics.DumpScope.Current.Directory, kernelCase.RTInputs.ToArray()).AsTensors();
        }
        else
        {
            actuals = Testing.RunKModel(kmodel_path, Diagnostics.DumpScope.Current.Directory, inputs).AsTensors();
        }
#if DEBUG
        for (int i = 0; i < actuals.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"actual_{i}.json"))
            {
                JsonSerializer.Serialize(fs, actuals[i], JsonSerializerOptions.Default);
            }
        }
#endif
        for (int i = 0; i < outputs.Length; i++)
        {
            var cos = Comparator.CosSimilarity(outputs[i], actuals[i]);
            Assert.True(cos > 0.999, $"the {CompileOptions.DumpDir} output {i} cos: {cos} ");
        }
    }

    private async Task Compile(IRModule module)
    {
        var pmgr = CompileSession.CreatePassManager("pmgr");
        var compiler = (Nncase.Compiler.Compiler)CompileSession.Compiler;
        compiler.TargetIndependentPass(pmgr);
        compiler.AutoDistributedPass(pmgr);
        compiler.AutoTilingPass(pmgr);
        compiler.TIRPass(pmgr);
        await pmgr.RunAsync(module);
    }
}
