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
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.TargetTest;

public class CpuKernelCase
{
    public CpuKernelCase(string name, Fusion fusion, Var[] vars, Tensor[] inputs)
    {
        Name = name;
        Fusion = fusion;
        Vars = vars;
        Inputs = inputs;
    }

    public string Name { get; }

    public Fusion Fusion { get; }

    public IReadOnlyList<Var> Vars { get; }

    public IReadOnlyList<Tensor> Inputs { get; }
}

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestCPUKernels : TestClassBase
{
    public UnitTestCPUKernels()
    {
        DefaultTargetName = CPUTarget.Kind;
        CompileOptions.TargetOptions = new CpuTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static Placement DefaultPlacement => new Placement(new[] { 1 }, "t");

    public static int Lane => Vector256.IsHardwareAccelerated ? 8 : 4;

    public static int Rank => 1;

    [Theory]
    [InlineData(new object[] { new[] { 32, 64 }, new[] { 64, 48 }, new[] { 48, 16 }, 0 })]
    [InlineData(new object[] { new[] { 128, 256 }, new[] { 256, 384 }, new[] { 384, 512 }, 1 })]
    [InlineData(new object[] { new[] { 1024, 2048 }, new[] { 2048, 1024 }, new[] { 1024, 3072 }, 2, true })]
    public async Task TestTileFlowCase(int[] ashape, int[] bshape, int[] eshape, int count, bool packing = false)
    {
        if (CompileOptions.TargetOptions is not CpuTargetOptions options)
        {
            return;
        }

        options.Packing = packing;
        options.MemoryBandWidths = [128, 64, 16, 2];
        options.MemoryCapacities = [0, 65536, 4194304, 2147483647];
        var a = new Var("a", new TensorType(DataTypes.Float32, ashape));
        var b = new Var("b", new TensorType(DataTypes.Float32, bshape));
        var c = IR.F.Tensors.MatMul(a, b);
        var d = IR.F.Math.Neg(c);
        var e = new Var("e", new TensorType(DataTypes.Float32, eshape));
        var f = IR.F.Tensors.MatMul(d, e);

        var feedDict = new Dictionary<Var, IValue>() {
            { a, IR.F.Tensors.ConstantOfShape(ashape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate() */ },
            { b, IR.F.Tensors.ConstantOfShape(bshape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate() */ },
            { e, IR.F.Tensors.ConstantOfShape(eshape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate() */ },
        };

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, new[] { f });
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

        var feedDict = new Dictionary<Var, IValue>() {
            { a, IR.F.Tensors.ConstantOfShape(ashape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate() */ },
            { b, IR.F.Tensors.ConstantOfShape(bshape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate() */ },
            { d, IR.F.Tensors.ConstantOfShape(dshape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate() */ },
            { f, IR.F.Tensors.ConstantOfShape(fshape, 1.0f).Evaluate() /* IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate() */ },
        };

        // var inputs = feedDict.Values.Select(v => v.AsTensor()).ToArray();
        // var actuals = Testing.RunKModel("/Users/lisa/Documents/nncase/tests_output/UnitTestCPUKernels/TestMatmulBinaryBinary/Case0/test.kmodel", Diagnostics.DumpScope.Current.Directory, inputs).AsTensors();
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), string.Empty), feedDict, new[] { g });
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 512, 64, 64 }, 0 })]
    public async Task TestSwish(int[] shape, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Swish(input);
        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackSwish(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()).Where(e => e is not Call { Target: Slice }));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 4, 8, 16, 32 }, 0 })]
    public async Task TestUnary(int[] shape, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);
        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackUnary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()).Where(e => e is not Call { Target: Slice }));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 8, 2 }, new int[] { 8, 2 }, 0 })] // normal
    [InlineData(new object[] { BinaryOp.Mul, new[] { 1, 8, 64, 2 * 8 }, new int[] { 1, 1, 64, 2 * 8 }, 1 })] // broadcast
    [InlineData(new object[] { BinaryOp.Add, new[] { 8, 16 }, new int[] { 16 }, 2 })] // normal
    public async Task TestPackBinary(BinaryOp op, int[] lhsShape, int[] rhsShape, int count)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackBinary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 2, 16 }, 2, 1e-6, true, 0 })]
    [InlineData(new object[] { new[] { 1, 2, 16 }, 2, 1e-6, false, 1 })]
    public async Task TestLayerNorm(int[] shape, int axis, float epsion, bool useMean, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = shape.Skip(axis).ToArray();
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.LayerNorm(axis, epsion, input, scale, bias, useMean);

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackLayerNorm(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext())).Where(e => e is not Call { Target: Slice });
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 2, 16, 32 }, 1e-5, 0 })]
    [InlineData(new object[] { new[] { 1, 32, 2048 }, 1e-6, 1 })]
    public async Task TestInstanceNorm(int[] shape, float epsion, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = new[] { shape[1] };
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.InstanceNormalization(input, scale, bias, epsion);

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackInstanceNorm(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext())).Where(e => e is not Call { Target: Slice });
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 4, 32, 32 }, ImageResizeMode.Bilinear, new[] { 1, 4, 64, 64 }, 0 })]
    [InlineData(new object[] { new[] { 1, 8, 32, 32 }, ImageResizeMode.NearestNeighbor, new[] { 1, 8, 64, 64 }, 1 })]
    public async Task TestResizeImage(int[] shape, ImageResizeMode resizeMode, int[] newSize, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Imaging.ResizeImage(resizeMode, input, Array.Empty<float>(), newSize);

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackResizeImage(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext())).Where(e => e is not Call { Target: Slice });
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]

    [InlineData(new object[] { new[] { 1, 384, 512 }, new[] { 512, 512 }, false, false, 0 })]
    [InlineData(new object[] { new[] { 1, 1, 384, 256 }, new[] { 32, 256, 512 }, false, false, 1 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 512 }, false, false, 2 })]
    [InlineData(new object[] { new[] { 1, 384, 512 }, new[] { 512, 512 }, false, true, 3 })]
    [InlineData(new object[] { new[] { 1, 1, 384, 256 }, new[] { 32, 256, 512 }, false, true, 4 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 512 }, false, true, 5 })]
    public async Task TestPackMatMul(int[] lhsShape, int[] rhsShape, bool constA, bool constB, int count)
    {
        Expr lhs = constA ? IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate().AsTensor() : new Var(new TensorType(DataTypes.Float32, lhsShape));
        Expr rhs = constB ? IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate().AsTensor() : new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<Var, IValue>();
        if (!constA)
        {
            feedDict.Add((Var)lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate());
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate());
        }

        var rule = new Passes.Rules.CPU.PackMatMul(2, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 384, 128 }, 0, new[] { 1, 384 }, 0 })]
    public async Task TestGather(int[] shape, int axis, int[] indicesShape, int count)
    {
        var vhidden_in = new Var("vhidden_in", new TensorType(DataTypes.Float32, shape));
        var vposition_ids = new Var("vposition_ids", new TensorType(DataTypes.Int64, indicesShape));
        var pre = IR.F.Tensors.Gather(vhidden_in, axis, vposition_ids); // f32[1,384,128]
        var feedDict = new Dictionary<Var, IValue>() {
            { vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { vposition_ids, IR.F.Random.Uniform(DataTypes.Int64, 6, 1, 1, indicesShape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 3, 28, 28 }, 0 })]
    public async Task TestInstanceNormal(int[] shape, int number)
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

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, posts);
    }

    [Theory]
    [InlineData([new[] { 1, 384, 8192 }, new[] { 1, 384, 64, 128 }, 1, 0])]
    [InlineData([new[] { 1, 8192, 384 }, new[] { 1, 64, 128, 384 }, 1, 1])]
    public async Task TestReshape(int[] inshape, int[] outshape, int packRank, int number)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, inshape));
        Expr pre;
        {
            pre = IR.F.Tensors.Reshape(input, outshape);
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inshape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackReshape(packRank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext())).Where(e => e is not Call { Target: Slice });
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, posts);
    }

    [Theory]
    [InlineData([new int[] { 2, 8, 16, 2 }, new int[] { 0, 2, 1, 3 }, 2, 0])]
    public async Task TestTranspose(int[] shape, int[] perm, int rank, int number)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        Expr pre; // f32[1,3,28,28]
        {
            var v4 = IR.F.Tensors.Transpose(input, perm); // f32[1,64,384,128]
            pre = v4;
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { input, Value.FromTensor(Tensor.From(Enumerable.Range(0, (int)TensorUtilities.GetProduct(shape)).Select(i => (float)i).ToArray(), shape)) },
        };

        var rule = new Passes.Rules.CPU.PackTranspose(rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, posts);
    }

    [Fact]
    public async Task TestTransposeMatmul()
    {
        ((CpuTargetOptions)CompileOptions.TargetOptions).Packing = true;

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

        var feedDict = new Dictionary<Var, IValue>() {
            { v13, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v13.CheckedShape).Evaluate() },
            { v15, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v15.CheckedShape).Evaluate() },
            { v19, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v19.CheckedShape).Evaluate() },
            { v24, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v24.CheckedShape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{0}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new int[] { 1, 1, 4, 4 }, new int[] { 8, 1, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, 0 })]
    [InlineData(new object[] { new int[] { 3, 2, 4, 4 }, new int[] { 8, 2, 3, 3 }, new int[] { 0, 0, 1, 1 }, new int[] { 1, 2 }, 1 })]
    [InlineData(new object[] { new int[] { 3, 2, 4, 4 }, new int[] { 8, 2, 3, 3 }, new int[] { 1, 0, 1, 1 }, new int[] { 2, 1 }, 2 })]
    [InlineData(new object[] { new int[] { 1, 512, 64, 64 }, new int[] { 512, 512, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, 3 })]
    public async Task TestConv2DAndIm2col(int[] inputShape, int[] wShape, int[] padding, int[] strides, int count)
    {
        var dilation = new[] { 1, 1 };
        var groups = 1;
        var input = new Var(new TensorType(DataTypes.Float32, inputShape));
        var weights = new Var(new TensorType(DataTypes.Float32, wShape));
        var bias = IR.F.Random.Normal(DataTypes.Float32, new[] { wShape[0] }).Evaluate().AsTensor();
        var pre = IR.F.NN.Conv2D(input, weights, bias, strides, new[,] { { padding[0], padding[1] }, { padding[2], padding[3] } }, dilation, PadMode.Constant, groups);
        var outShape = pre.CheckedShape.ToValueArray();

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate() },
            { weights, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, wShape).Evaluate() },
        };

        Expr post = Passes.Rules.CPU.PackConv2D.AddCandidate(input, weights, bias, strides, padding, wShape, outShape);
        Expr post2 = Passes.Rules.CPU.PackConv2D.AddPackedCandidate(input, weights, bias, strides, padding, wShape, outShape, Lane);
        var posts = new[] { pre, post, post2 };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { false, 0 })]
    [InlineData(new object[] { true, 1 })] // enable packing
    public async Task TestDecodeLayer(bool packing, int count)
    {
        // Memory usage is too high for CI env
        if (bool.TryParse(Environment.GetEnvironmentVariable("CI"), out var inCI) && inCI)
        {
            return;
        }

        ((CpuTargetOptions)CompileOptions.TargetOptions).Packing = packing;
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

        var feedDict = new Dictionary<Var, IValue>() {
            { vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 13,  new[] { 1, 384, 8192 }).Evaluate() },
            { vattn_mask, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 14,  new[] { 1, 1, 384, 384 }).Evaluate() },
            { vposition_ids, IR.F.Random.Uniform(DataTypes.Int64, 383, 1, 15, new[] { 1, 384 }).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { false, 0 })]
    [InlineData(new object[] { true, 1 })] // enable packing
    public async Task TestVAEDecRes(bool packing, int count)
    {
        CompileOptions.TargetOptions = new CpuTargetOptions() { Packing = packing };
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

        var feedDict = new Dictionary<Var, IValue>() {
            { vlatent_sample, IR.F.Random.Normal(DataTypes.Float32, 0, 0.1, 13,  new[] { 1, 4, 64, 64 }).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    internal async Task RunCases(string dumpDir, Dictionary<Var, IValue> feedDict, IEnumerable<Expr> posts)
    {
        var postArray = posts.ToArray();
        using var pinner = new ExprPinner(postArray);
        for (int i = 0; i < postArray.Length; i++)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(postArray[i]));
#endif
            var kernelCase = new CpuKernelCase($"Case{i}", new Fusion("kernel", CPUTarget.Kind, postArray[i], feedDict.Keys.ToArray()), feedDict.Keys.ToArray(), feedDict.Values.Select(v => v.AsTensor()).ToArray());
            await Run(dumpDir, kernelCase);
        }
    }

    internal async Task Run(string dumpDir, CpuKernelCase kernelCase)
    {
        CompileOptions.DumpDir = Path.Join(dumpDir, kernelCase.Name);
        using var dumpScope = new Diagnostics.DumpScope(string.Empty, CompileOptions.DumpFlags);

        // convert fusion to prim func
        var fusion = kernelCase.Fusion;
        if (fusion.Body.CheckedType is InvalidType)
        {
            return;
        }

        var main = new Function(fusion.Body, kernelCase.Vars.ToArray());

        var module = new IR.IRModule(main);
        var inputs = kernelCase.Inputs.ToArray();
        var outputs = fusion.Body.Evaluate(kernelCase.Vars.Zip(inputs).ToDictionary(p => p.First, p => (IValue)Value.FromTensor(p.Second))).AsTensors();

#if DEBUG
        for (var i = 0; i < inputs.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"input_{i}.bin"))
            {
                fs.Write(inputs[i].BytesBuffer);
            }
        }

        for (int i = 0; i < outputs.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"output_{i}.bin"))
            {
                fs.Write(outputs[i].BytesBuffer);
            }
        }
#endif
        await Compile(module);
        var (kmodel_path, _) = Testing.BuildKModel("test", module, CompileSession, false);
        var actuals = Testing.RunKModel(kmodel_path, Diagnostics.DumpScope.Current.Directory, inputs).AsTensors();
#if DEBUG
        for (int i = 0; i < actuals.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"actual_{i}.bin"))
            {
                fs.Write(actuals[i].BytesBuffer);
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
        CompileSession.Target.RegisterTargetDependentAfterQuantPass(pmgr, CompileSession.CompileOptions);
        CompileSession.Target.RegisterTargetDependentBeforeCodeGen(pmgr, CompileSession.CompileOptions);
        await pmgr.RunAsync(module);
    }
}
