// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;

namespace Nncase.Tests.EvaluatorTest;

public sealed class PagedAttentionKVCacheTestData : TheoryData<TestFixture.PagedAttentionKVCacheTestFixture, Placement>
{
    private static readonly (string Name, long[] QueryLens, long[] SeqLens)[] TestScenarios =
    [
        ("prefill", [4L], [4L]),
        ("prefill*2", [12L, 15L], [12L, 15L]),
        ("extend", [4L], [8L]),
        ("prefill+extend", [4L, 4L], [4L, 8L]),
        ("prefill+decode", [4L, 1L], [4L, 9L]),
    ];

    private static readonly Runtime.TypeCode[] TypeConfigs = [
        Runtime.TypeCode.Float32,
        Runtime.TypeCode.Float16,
    ];

    private static readonly (int NumQ, int NumKV, int Dim)[] HeadConfigs =
    [
        (1, 1, 64),
        (2, 2, 64),
        (4, 4, 128),
        (32, 8, 128),
        (64, 8, 128),
    ];

    private static readonly (int Layer, int BlockSize, int NumBlocks)[] CacheConfigs = [
        (1, 4, 8),
        (1, 16, 8),
        (1, 32, 16),
        (1, 128, 32),
        (1, 256, 32),
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

    private static readonly (PagedKVCacheDimKind[] Sharding, SBPSplit[] Policies, Placement Placement)[] ShardingConfigs =
    [
        (Array.Empty<PagedKVCacheDimKind>(), Array.Empty<SBPSplit>(), new Placement(new[] { 1 }, "t")),
        (new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, new Placement(new[] { 1 }, "t")),
    ];

    private static readonly (AttentionDimKind[] QLayout, AttentionDimKind[] KLayout)[] QKLayoutConfigs =
    [
        ([AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim],
         [AttentionDimKind.Seq, AttentionDimKind.Dim, AttentionDimKind.Head]),
        ([AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq],
         [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]),
    ];

    public PagedAttentionKVCacheTestData()
    {
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
                            foreach (var (shardingAxes, axisPolicies, placement) in ShardingConfigs)
                            {
                                foreach (var (qlayout, klayout) in QKLayoutConfigs)
                                {
                                    Add(new TestFixture.PagedAttentionKVCacheTestFixture(queryLens, seqLens, numQHeads, numKVHeads, headDim, blockSize, numBlocks, typeCode, numLayer, cacheLayout, packedAxes, shardingAxes, axisPolicies, qlayout, klayout), placement);
                                }
                            }
                        }
                    }
                }
            }
        }

        Add(new TestFixture.PagedAttentionKVCacheTestFixture([4], [4], 4, 2, 32, 8, 32, Runtime.TypeCode.Float32, 1, [PagedKVCacheDimKind.NumLayers, PagedKVCacheDimKind.NumBlocks, PagedKVCacheDimKind.KV, PagedKVCacheDimKind.NumKVHeads, PagedKVCacheDimKind.HeadDim, PagedKVCacheDimKind.BlockSize], [PagedKVCacheDimKind.HeadDim], new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq], [AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]), new Placement(new[] { 1 }, "t"));
    }
}

public sealed class PagedAttentionSchedulerTestData : TheoryData<PagedAttentionConfig, int[]>
{
    private static readonly Runtime.TypeCode[] TypeConfigs = [
        Runtime.TypeCode.Float32,
        Runtime.TypeCode.Float16,
    ];

    private static readonly (int NumQ, int NumKV, int Dim)[] HeadConfigs =
    [
        (64, 8, 128),
    ];

    private static readonly (int Layer, int BlockSize, int NumBlocks)[] CacheConfigs = [
        (1, 256, 32),
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
        (Array.Empty<PagedKVCacheDimKind>(), Array.Empty<SBPSplit>(), Array.Empty<int>()),
        (new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, new[] { 1 }),
        (new[] { PagedKVCacheDimKind.NumBlocks }, new[] { SBP.S(0) }, new[] { 8 }),
    ];

    public PagedAttentionSchedulerTestData()
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
                            var kvType = PrimType.FromTypeCode(typeCode);
                            var config = new PagedAttentionConfig(numLayer, numKVHeads, headDim, kvType, blockSize, cacheLayout, packedAxes, new[] { 128 / kvType.SizeInBytes }, shardingAxes, axisPolicies);
                            Add(config, hierarchy);
                        }
                    }
                }
            }
        }
    }
}

public class UnitTestEvaluatorNN : TestClassBase
{
    [Fact]
    public void TestActivationCelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 0.8F;
        var expect = OrtKI.Celu(input, alpha);
        var expr = IR.F.NN.Celu(input.ToTensor(), alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationElu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 0.8F;
        var expect = OrtKI.Elu(input, alpha);
        var expr = IR.F.NN.Elu(input.ToTensor(), alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationHardSwish()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 1F / 6F;
        var beta = 0.5F;
        var expect = input * OrtKI.HardSigmoid(input, alpha, beta);
        var expr = IR.F.NN.HardSwish(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationSwish()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = input * OrtKI.Sigmoid(input);
        var expr = IR.F.NN.Swish(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationLeakyRelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 0.6F;
        var expect = OrtKI.LeakyRelu(input, alpha);
        var expr = IR.F.NN.LeakyRelu(input.ToTensor(), alpha);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationRelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Relu(input);
        var expr = IR.F.NN.Relu(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationRelu6()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Clip(input, 0F, 6F);
        var expr = IR.F.NN.Relu6(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationSelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 1.2F;
        var gamma = 1.3F;
        var expect = OrtKI.Selu(input, alpha, gamma);
        var expr = IR.F.NN.Selu(input.ToTensor(), alpha, gamma);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationSigmoid()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Sigmoid(input);
        var expr = IR.F.NN.Sigmoid(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationHardSigmoid()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var alpha = 1.2F;
        var gamma = 1.3F;
        var expect = OrtKI.HardSigmoid(input, alpha, gamma);
        var expr = IR.F.NN.HardSigmoid(input.ToTensor(), alpha, gamma);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationPRelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var slope = 0.2F;
        var expect = OrtKI.PRelu(input, slope);
        var expr = IR.F.NN.PRelu(input.ToTensor(), slope);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationErf()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Erf(input);
        var expr = IR.F.NN.Erf(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestActivationGelu()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var scaledInput = OrtKI.Mul(0.5f, input);
        var expect = OrtKI.Mul(0.5f, OrtKI.Mul(scaledInput, OrtKI.Add(OrtKI.Erf(OrtKI.Div(scaledInput, OrtKI.Sqrt(2f))), 1f)));
        var expr = IR.F.NN.Gelu(input.ToTensor(), 0.5);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBatchToSpace()
    {
        var a = new float[] { 1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16 };
        var input = Tensor.From(a, [4, 1, 2, 2]);
        var shape = new long[] { 2, 2 };
        var b = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var expect = Tensor.From(b, [1, 1, 4, 4]);
        var crops = new long[] { 0, 0, 0, 0 };
        var expr = IR.F.NN.BatchToSpace(
            input,
            Tensor.From(shape, [2]),
            Tensor.From(crops, [2, 2]));
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestConv2D()
    {
        var input = OrtKI.Random(1, 4, 5, 5);
        var weight = OrtKI.Random(8, 4, 3, 3);
        var bias = OrtKI.Random(8);
        var expect = OrtKI.Conv(
            input,
            weight,
            bias,
            "NOTSET",
            new long[] { 1, 1 },
            1,
            new long[] { 3, 3 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 1, 1 });

        var expr = IR.F.NN.Conv2D(
            input.ToTensor(),
            weight.ToTensor(),
            bias.ToTensor(),
            stride: new[] { 1, 1 },
            padding: Tensor.From<int>(new int[] { 1, 1, 1, 1 }, [2, 2]),
            dilation: new[] { 1, 1 },
            PadMode.Constant,
            1);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConv2D_1()
    {
        var input = OrtKI.Random(1, 4, 5, 5);
        var weight = OrtKI.Random(8, 4, 3, 3);
        var bias = OrtKI.Random(8);
        var min = 0.5F;
        var max = 1F;
        var conv = OrtKI.Conv(
            input,
            weight,
            bias,
            "NOTSET",
            new long[] { 1, 1 },
            1,
            new long[] { 3, 3 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 1, 1 });
        var expect = OrtKI.Clip(conv, min, max);

        var expr = IR.F.NN.Conv2D(
            input.ToTensor(),
            weight.ToTensor(),
            bias.ToTensor(),
            stride: new[] { 1, 1 },
            padding: Tensor.From<int>(new int[] { 1, 1, 1, 1 }, [2, 2]),
            dilation: new[] { 1, 1 },
            Nncase.PadMode.Constant,
            1,
            new[] { min, max });
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConv2DTranspose()
    {
        var input = OrtKI.Random(1, 1, 5, 5);
        var weight = OrtKI.Random(2, 1, 3, 3);
        var bias = OrtKI.Random(2);
        var expect = OrtKI.ConvTranspose(
            input,
            OrtKI.Transpose(weight, new long[] { 1, 0, 2, 3 }),
            bias,
            "NOTSET",
            new long[] { 1, 1 },
            1,
            kernel_shape: new long[] { 3, 3 },
            output_padding: new long[] { 0, 0 },
            output_shape: new long[] { 1, 2, 5, 5 },
            pads: new long[] { 1, 1, 1, 1 },
            strides: new long[] { 1, 1 });

        var outShape = Tensor.From(new long[] { 1, 2, 5, 5 }, new RankedShape(4));
        var expr = IR.F.NN.Conv2DTranspose(
            input.ToTensor(),
            weight.ToTensor(),
            bias.ToTensor(),
            outShape,
            stride: new[] { 1, 1 },
            padding: Tensor.From<long>(new long[] { 1, 1, 1, 1 }, [2, 2]),
            outputPadding: Tensor.From<long>(new long[] { 0, 0 }, [2]),
            dilation: new[] { 1, 1 },
            PadMode.Constant,
            1);
        CompilerServices.InferenceType(expr);
        var expectValue = expect.ToArray<float>();
        var realValue = expr.Evaluate().AsTensor().ToArray<float>();
        var cos = Nncase.Tests.Comparator.CosSimilarity(expectValue, realValue);
        Assert.True(cos >= 0.99);
    }

    [Fact]
    public void TestHardmax()
    {
        var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var nncaseTensor = ortTensor.ToTensor();
        DoHardmax(ortTensor, nncaseTensor, -1L);
        DoHardmax(ortTensor, nncaseTensor, 1L);
    }

    [Fact]
    public void TestLayerNorm()
    {
        var epsilon = 1e-05f;
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var x = OrtKI.Random(shape);
            for (int i = 0; i < shape.Length; i++)
            {
                var scale = OrtKI.Random(new[] { shape[i] });
                var b = OrtKI.Random(new[] { shape[i] });
                var axis = i;

                // var expect = OrtKI.LayerNormalization(x, scale, b, axis, epsilon, 1L);
                var expect = IR.F.NN.LayerNorm((int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor()).Evaluate().AsTensor();
                DoLayerNorm(expect, (int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor());
            }
        }

        {
            var shape = new long[] { 1, 3, 16, 16 };
            var x = OrtKI.Random(shape);
            for (int i = -shape.Length + 1; i != 0; i++)
            {
                var axis = i;
                var scale = OrtKI.Random(new[] { shape[^System.Math.Abs(i)] });
                var b = OrtKI.Random(new[] { shape[^System.Math.Abs(i)] });

                // var expect = OrtKI.LayerNormalization(x, scale, b, axis, epsilon, 1L);
                var expect = IR.F.NN.LayerNorm((int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor()).Evaluate().AsTensor();
                DoLayerNorm(expect, (int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor());
            }
        }

        {
            var shape = new long[] { 1, 16 };
            var x = OrtKI.Random(shape);
            for (int i = 0; i < shape.Length; i++)
            {
                var scale = OrtKI.Random(new[] { shape[i] });
                var b = OrtKI.Random(new[] { shape[i] });
                var axis = i;

                // var expect = OrtKI.LayerNormalization(x, scale, b, axis, epsilon, 1L);
                var expect = IR.F.NN.LayerNorm((int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor()).Evaluate().AsTensor();
                DoLayerNorm(expect, (int)axis, epsilon, x.ToTensor(), scale.ToTensor(), b.ToTensor());
            }
        }
    }

    [Fact]
    public void TestL2Normalization()
    {
        var a = new float[] { 0F, 2F, 3F, 2F, 2F, 2F };
        var b = new float[] { 0F, 0.4F, 0.6F, 0.4F, 0.4F, 0.4F };
        {
            var expect = Tensor.From(b, [6]);
            var input = Tensor.From(a, [6]);
            DoL2Normalization(expect, input);
        }

        {
            var expect = Tensor.From(b, [1, 2, 3]);
            var input = Tensor.From(a, [1, 2, 3]);
            DoL2Normalization(expect, input);
        }
    }

    [Fact]
    public void TestBatchNormalization()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var x = OrtKI.Random(shape);
        var scale = OrtKI.Random(new long[] { shape[1] });
        var b = OrtKI.Random(new long[] { shape[1] });
        var mean = OrtKI.Random(new long[] { shape[1] });
        var var = OrtKI.Random(new long[] { shape[1] });
        var epsilon = 0.01F;
        var momentum = 0.5F;

        var expect = OrtKI.BatchNormalization(x, scale, b, mean, var, epsilon, momentum);
        var expr = IR.F.NN.BatchNormalization(x.ToTensor(), scale.ToTensor(), b.ToTensor(), mean.ToTensor(), var.ToTensor(), epsilon, momentum);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestInstanceNormalization()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var x = OrtKI.Random(shape);
        var scale = OrtKI.Random(new long[] { shape[1] });
        var b = OrtKI.Random(new long[] { shape[1] });
        var epsilon = 0.01F;

        var expect = OrtKI.InstanceNormalization(x, scale, b, epsilon);
        var expr = IR.F.NN.InstanceNormalization(x.ToTensor(), scale.ToTensor(), b.ToTensor(), epsilon);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestLpNormalization()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        DoLpNormalization(input, 0, 1);
        DoLpNormalization(input, 0, 2);
        DoLpNormalization(input, 1, 1);
        DoLpNormalization(input, 1, 2);
    }

    [Fact]
    public void TestLRN()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);
        var alpha = 0.001F;
        var beta = 0.5F;
        var bias = 0.8F;
        var size = 3L;

        var expect = OrtKI.LRN(input, alpha, beta, bias, size);
        var expr = IR.F.NN.LRN(input.ToTensor(), alpha, beta, bias, size);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestOneHotTF()
    {
        var a = new int[] { 1, 2, 0, 3 };
        var indices = Tensor.From(a, [4]);
        var depth = 5;
        var values = Tensor.From(new float[] { 0, 1 }, new RankedShape(new[] { 2 }));
        var axis = 0L;

        var b = new float[] { 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
        var expect = OrtKISharp.Tensor.MakeTensor(b, new long[] { 5, 4 });

        var expr = NN.OneHot(OneHotMode.Normal, indices, depth, values, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestOneHotOnnx()
    {
        var a = new float[] { 1, 2, 0, 3 };
        var indices = Tensor.From(a, [4]);
        var depth = 5F;
        var values = Tensor.From(new float[] { 0, 1 }, new RankedShape(new[] { 2 }));
        var axis = 1L;

        var expect = OrtKI.OneHot(indices.ToOrtTensor(), depth, values.ToOrtTensor(), axis);

        var expr = NN.OneHot(OneHotMode.ProcessNeg, indices, depth, values, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPad()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.From<int>(new[] { 0, 0, 0, 0, 1, 1, 2, 2 }, new RankedShape(new[] { 4, 2 }));
        var value = Tensor.FromScalar<float>(1.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new long[] { 1, 1, 4, 7 }, result.Shape);
    }

    [Fact]
    public void TestPad2()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.From<long>(new long[] { 0, 0, 1, 2, 2, 4, 5, 6 }, new RankedShape(4, 2));
        var value = Tensor.FromScalar<float>(2.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new long[] { 1, 4, 8, 14 }, result.Shape);
    }

    [Fact]
    public void TestPadConstant()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ortPads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(input, ortPads, constant, "constant");

        var nncaesPads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new RankedShape(4, 2));
        var expr = NN.Pad(input.ToTensor(), nncaesPads, Nncase.PadMode.Constant, constant.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadReflect()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ortPads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(input, ortPads, constant, "reflect");

        var nncasePads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new RankedShape(4, 2));
        var expr = NN.Pad(input.ToTensor(), nncasePads, Nncase.PadMode.Reflect, constant.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadReflect2()
    {
        var input = new Var();
        var feed = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 12, new long[] { 1, 3, 4, 5 }).Evaluate() }, };
        var output = NN.Pad(IR.F.Math.Abs(input), Tensor.From(new long[,] { { 1, 1 }, { -1, -1 }, { 1, 1 }, { 3, 3 } }), PadMode.Reflect, 0.0f);
        CompilerServices.InferenceType(output);
        var outputArray = output.Evaluate(feed).AsTensor().ToArray<float>();
        Assert.Contains(outputArray, f => f > 0.0f);
    }

    [Fact]
    public void TestPadSymmetric()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6 };
        var expect = OrtKISharp.Tensor.MakeTensor(b, new long[] { 1, 1, 4, 5 });

        var input = Tensor.From(a, new RankedShape(1, 1, 2, 3));
        var pads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new RankedShape(4, 2));
        var constant = 0F;
        var expr = NN.Pad(input, pads, Nncase.PadMode.Symmetric, constant);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestPadEdge()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ortPads = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, new long[] { 8 });
        var constant = OrtKISharp.Tensor.FromScalar(1F);
        var expect = OrtKI.Pad(input, ortPads, constant, "edge");

        var nncaePads = Tensor.From<long>(new long[] { 0, 0, 0, 0, 1, 1, 1, 1 }, new RankedShape(4, 2));
        var expr = NN.Pad(input.ToTensor(), nncaePads, Nncase.PadMode.Edge, constant.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestLogSoftmax()
    {
        var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var nncaseTensor = ortTensor.ToTensor();
        DoLogSoftmax(ortTensor, nncaseTensor, -1L);
        DoLogSoftmax(ortTensor, nncaseTensor, 1L);
    }

    [Fact]
    public void TestReduceWindow2DMean()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ceilMode = false;
        var countIncludePad = false;
        var dilations = new long[] { 1, 1 };
        var filter = new long[] { 3, 3 };
        var stride = new long[] { 1, 1 };
        var onnxPads = new long[] { 1, 1, 1, 1 };
        var expect = OrtKI.AveragePool(
            input,
            "NOTSET",
            ceilMode ? 1 : 0,
            countIncludePad ? 1 : 0,
            filter,
            onnxPads,
            stride);

        var expr = IR.F.NN.ReduceWindow2D(
            ReduceOp.Mean,
            input.ToTensor(),
            0.0f,
            filter,
            stride,
            new[,] { { 1, 1 }, { 1, 1 }, },
            dilations,
            false,
            false);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReduceWindow2DMax()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var ceilMode = false;
        var dilations = new long[] { 1, 1 };
        var filter = new long[] { 3, 3 };
        var stride = new long[] { 1, 1 };
        var onnxPads = new long[] { 1, 1, 1, 1 };
        var storageOrder = 0L;
        var expect = OrtKI.MaxPool(
            input,
            "NOTSET",
            ceilMode ? 1 : 0,
            dilations,
            filter,
            onnxPads,
            storageOrder,
            stride)[0];

        var expr = IR.F.NN.ReduceWindow2D(
            ReduceOp.Max,
            input.ToTensor(),
            0.0f,
            filter,
            stride,
            new[,] { { 1, 1 }, { 1, 1 }, },
            dilations,
            false,
            false);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSoftmax()
    {
        var ortTensor = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var nncaseTensor = ortTensor.ToTensor();
        DoSoftmax(ortTensor, nncaseTensor, -1);
        DoSoftmax(ortTensor, nncaseTensor, 1);
    }

    [Fact]
    public void TestSoftplus()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Softplus(input);
        var expr = IR.F.NN.Softplus(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSoftsign()
    {
        var input = OrtKI.Random(new long[] { 1, 3, 16, 16 });
        var expect = OrtKI.Softsign(input);
        var expr = IR.F.NN.Softsign(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSpaceToBatch()
    {
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var input = Tensor.From(a, [1, 4, 4, 1]);
        var shape = new long[] { 2, 2 };

        var output = new float[] { 1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16 };
        var expect = Tensor.From(output, [4, 2, 2, 1]);
        var crops = new long[] { 0, 0, 0, 0 };
        var expr = IR.F.Tensors.NCHWToNHWC(IR.F.NN.SpaceToBatch(
            IR.F.Tensors.NHWCToNCHW(input).Evaluate().AsTensor(),
            Tensor.From(shape, [2]),
            Tensor.From(crops, [2, 2])));
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Theory]
    [InlineData([true])]
    [InlineData([false])]
    public void TestScaledDotProductAttention(bool isCausal)
    {
        var q = OrtKISharp.Tensor.MakeTensor(Enumerable.Range(0, 5 * 4).Select(i => (float)i).ToArray(), [5, 4]);
        var k = OrtKISharp.Tensor.MakeTensor(Enumerable.Range(1, 5 * 4).Select(i => (float)i).ToArray(), [5, 4]);
        var v = OrtKISharp.Tensor.MakeTensor(Enumerable.Range(2, 5 * 4).Select(i => (float)i).ToArray(), [5, 4]);
        var s = TestFixture.PagedAttentionKVCacheTestFixture.ScaledDotProductAttention(q, k, v, isCausal: isCausal, scale: 1.0f);
        var span = s.GetBuffer<float>().ToArray();

        if (isCausal)
        {
            Assert.True(span.SequenceEqual([
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
                9.0f,
                10.0f,
                11.0f,
                12.0f,
                13.0f,
                14.0f,
                15.0f,
                16.0f,
                17.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f]));
        }
        else
        {
            Assert.True(span.SequenceEqual([
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f]));
        }
    }

    [Theory]
    [InlineData([true])]
    public void TestScaledDotProductAttentionGQA(bool isCausal)
    {
        var q = OrtKISharp.Tensor.MakeTensor(new float[] { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 }, [4, 4, 1]);
        var k = OrtKISharp.Tensor.MakeTensor(new float[] { 0, 0, 0, 0, 1, 1, 1, 1 }, [2, 4, 1]);
        var v = OrtKISharp.Tensor.MakeTensor(new float[] { 2, 2, 2, 2, 3, 3, 3, 3 }, [2, 4, 1]);
        var s = TestFixture.PagedAttentionKVCacheTestFixture.ScaledDotProductAttention(q, k, v, isCausal: isCausal, scale: 1.0f);
        var span = s.GetBuffer<float>().ToArray();

        if (isCausal)
        {
            Assert.True(span.SequenceEqual([
                2.0f,
                2.0f,
                2.0f,
                2.0f,
                2.0f,
                2.0f,
                2.0f,
                2.0f,
                3.0f,
                3.0f,
                3.0f,
                3.0f,
                3.0f,
                3.0f,
                3.0f,
                3.0f,]));
        }
        else
        {
            Assert.True(span.SequenceEqual([
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f,
                18.0f,
                19.0f,
                20.0f,
                21.0f]));
        }
    }

    [Theory]
    [ClassData(typeof(PagedAttentionKVCacheTestData))]
    public void TestPagedAttention(TestFixture.PagedAttentionKVCacheTestFixture testFixture, Placement placement)
    {
        var dataGeneratorOptions = new TestFixture.PagedAttentionKVCacheTestFixture.DataGeneratorOptions(Random: true, IncreaseBy: [AttentionDimKind.Head], ResetForKV: true);
        var referenceResults = TestFixture.PagedAttentionKVCacheTestFixture.PrepareReferenceResults(testFixture.QueryLens, testFixture.SeqLens, testFixture.NumQHeads, testFixture.Config.NumKVHeads, testFixture.Config.HeadDim, testFixture.Config.NumLayers, testFixture.Config.KVPrimType, dataGeneratorOptions);

        var (root, queryVar, kVVars, kVCacheObjVar) = Evaluator.NN.RefPagedAttentionKVCache.BuildPagedAttentionKernel(testFixture.QueryLens, testFixture.SeqLens, testFixture.NumQHeads, testFixture.NumBlocks, testFixture.QLayout, testFixture.KLayout, testFixture.Config);

        var kvinputs = TestFixture.PagedAttentionKVCacheTestFixture.PrepareKVInputs(testFixture.QueryLens, testFixture.SeqLens, testFixture.ContextLens, testFixture.NumBlocks, placement, referenceResults, testFixture.Config);

        var feedDict = new Dictionary<IVar, IValue>();
        {
            feedDict.Add(queryVar, Value.FromTensor(referenceResults.GetQueryTensor()));
            for (int layerId = 0; layerId < testFixture.Config.NumLayers; layerId++)
            {
                feedDict.Add(kVVars[layerId][0], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 0)));
                feedDict.Add(kVVars[layerId][1], Value.FromTensor(kvinputs.GetKeyValueTensor(layerId, 1)));
            }

            feedDict.Add(kVCacheObjVar, Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kvinputs.KVCacheObj))));
        }

        // 4. evaluate and compare
        {
            // var refTensor = OrtKI.Concat(refOutputs.ToArray(), 1L).ToTensor();
            var refTensor = referenceResults.GetOutputTensor();
            var actualTensor = root.Evaluate(feedDict).AsTensor();

            var cos = Comparator.CosSimilarity(refTensor, actualTensor);
            Assert.True(cos > 0.999, $"cos: {cos} ");
        }
    }

    [Theory]
    [ClassData(typeof(PagedAttentionSchedulerTestData))]
    public void TestPagedAttentionScheduler(PagedAttentionConfig config, int[] hierarchy)
    {
        // 1. singele session prefill test
        {
            int numBlocks = 32;
            int maxSessions = 4;
            int maxModelLen = config.BlockSize * (numBlocks / maxSessions);

            var scheduler = new Evaluator.NN.RefPagedAttentionScheduler(config, numBlocks, maxModelLen, hierarchy);
            long[] sessionIds = [1];
            long[] queryLens = [64];
            var kvCache = scheduler.Schedule(sessionIds, queryLens);

            Assert.Equal(1, kvCache.NumSeqs);
            Assert.Equal(64, kvCache.NumTokens);

            Assert.True(kvCache.SeqLens.SequenceEqual([64]));
            Assert.True(kvCache.ContextLens.SequenceEqual([0]));

            // session id to large.
            Assert.Throws<InvalidOperationException>(() =>
            {
                long[] sessionIds = [maxSessions + 1];
                long[] queryLens = [64];
                var kvCache = scheduler.Schedule(sessionIds, queryLens);
            });

            var qHead = 8;
            var func = scheduler.CreateTestFunction(qHead, [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim], [AttentionDimKind.Seq, AttentionDimKind.Head, AttentionDimKind.Dim]);

            func.Evaluate(new Dictionary<IVar, IValue>()
            {
                { func.Parameters[0], Value.FromTensor(Tensor.Zeros(config.KVPrimType, [64, config.NumKVHeads, config.HeadDim])) },
                { func.Parameters[1], Value.FromTensor(Tensor.Zeros(config.KVPrimType, [64, qHead, config.HeadDim])) },
                { func.Parameters[2], Value.FromTensor(Tensor.Zeros(config.KVPrimType, [64, qHead, config.HeadDim])) },
                { func.Parameters[3], Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kvCache))) },
            });
        }

        // 2. multi prefills.
        {
            int numBlocks = 32;
            int maxSessions = 8;
            int maxModelLen = config.BlockSize * (numBlocks / maxSessions);

            var scheduler = new Evaluator.NN.RefPagedAttentionScheduler(config, numBlocks, maxModelLen, hierarchy);

            long[] sessionIds = [1, 2, 3];
            long[] queryLens = [32, 48, 64];

            var kvCache = scheduler.Schedule(sessionIds, queryLens);

            Assert.Equal(3, kvCache.NumSeqs);
            Assert.Equal(144, kvCache.NumTokens); // 32+48+64

            var seqLens = kvCache.SeqLens.ToArray<long>();
            Assert.Equal(32, seqLens[0]);
            Assert.Equal(48, seqLens[1]);
            Assert.Equal(64, seqLens[2]);

            var contextLens = kvCache.ContextLens.ToArray<long>();
            Assert.Equal(0, contextLens[0]);
            Assert.Equal(0, contextLens[1]);
            Assert.Equal(0, contextLens[2]);
        }

        // 3. single session prefill and decode.
        {
            int numBlocks = 32;
            int maxSessions = 8;
            int maxModelLen = config.BlockSize * (numBlocks / maxSessions);

            var scheduler = new Evaluator.NN.RefPagedAttentionScheduler(config, numBlocks, maxModelLen, hierarchy);

            // prefill
            long[] sessionIds1 = [2];
            long[] queryLens1 = [48];

            var kvCache1 = scheduler.Schedule(sessionIds1, queryLens1);

            Assert.Equal(1, kvCache1.NumSeqs);
            Assert.Equal(48, kvCache1.NumTokens);

            var seqLens1 = kvCache1.SeqLens.ToArray<long>();
            Assert.Equal(48, seqLens1[0]);

            var contextLens1 = kvCache1.ContextLens.ToArray<long>();
            Assert.Equal(0, contextLens1[0]);

            long[] sessionIds2 = [2];
            long[] queryLens2 = [1];

            var kvCache2 = scheduler.Schedule(sessionIds2, queryLens2);

            Assert.Equal(1, kvCache2.NumSeqs);
            Assert.Equal(1, kvCache2.NumTokens);

            var seqLens2 = kvCache2.SeqLens.ToArray<long>();
            Assert.Equal(49, seqLens2[0]); // 48 + 1

            var contextLens2 = kvCache2.ContextLens.ToArray<long>();
            Assert.Equal(48, contextLens2[0]);

            long[] sessionIds3 = [2];
            long[] queryLens3 = [1];

            var kvCache3 = scheduler.Schedule(sessionIds3, queryLens3);

            Assert.Equal(1, kvCache3.NumSeqs);
            Assert.Equal(1, kvCache3.NumTokens);

            var seqLens3 = kvCache3.SeqLens.ToArray<long>();
            Assert.Equal(50, seqLens3[0]); // 49 + 1

            var contextLens3 = kvCache3.ContextLens.ToArray<long>();
            Assert.Equal(49, contextLens3[0]);
        }
    }

    private void DoHardmax(OrtKISharp.Tensor ortTensor, Tensor nncaseTensor, long axis)
    {
        var expect = OrtKI.Hardmax(ortTensor, axis);
        var expr = IR.F.NN.Hardmax(nncaseTensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoLayerNorm(Tensor expect, int axis, float epsilon, Tensor input, Tensor scale, Tensor bias)
    {
        var expr = IR.F.NN.LayerNorm(axis, epsilon, input, scale, bias);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    private void DoLogSoftmax(OrtKISharp.Tensor ortTensor, Tensor nncaseTensor, long axis)
    {
        var expect = OrtKI.LogSoftmax(ortTensor, axis);
        var expr = IR.F.NN.LogSoftmax(nncaseTensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoSoftmax(OrtKISharp.Tensor ortTensor, Tensor nncaseTensor, int axis)
    {
        var expect = OrtKI.Softmax(ortTensor, axis);
        var expr = IR.F.NN.Softmax(nncaseTensor, axis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void DoL2Normalization(Tensor expect, Tensor input)
    {
        var expr = IR.F.NN.L2Normalization(input);
        CompilerServices.InferenceType(expr);

        // fix precision issue on Macos
        var cos = Comparator.CosSimilarity(expect, expr.Evaluate().AsTensor());
        Assert.True(cos > 0.999F);
    }

    private void DoLpNormalization(OrtKISharp.Tensor input, long axis, long p)
    {
        var expect = OrtKI.LpNormalization(input, axis, p);
        var expr = IR.F.NN.LpNormalization(input.ToTensor(), axis, p);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }
}
