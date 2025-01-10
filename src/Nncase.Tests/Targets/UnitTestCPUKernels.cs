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
        CompileOptions.TargetOptions = new CpuTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static Placement DefaultPlacement => new Placement(new[] { 1 }, "t");

    public static int Lane => Vector256.IsHardwareAccelerated ? 8 : 4;

    public static int Rank => 2;

    [Theory]
    [InlineData(new object[] { new[] { 32, 64 }, false, new[] { 64, 48 }, false, new[] { 48, 16 }, true, new[] { 1 }, 0 })]
    [InlineData(new object[] { new[] { 128, 256 }, true, new[] { 256, 384 }, false, new[] { 384, 512 }, true, new[] { 2 }, 1 })]
    [InlineData(new object[] { new[] { 1024, 2048 }, false, new[] { 2048, 1024 }, true, new[] { 1024, 3072 }, true, new[] { 4 }, 2, true })]
    [InlineData(new object[] { new[] { 128, 256 }, true, new[] { 256, 384 }, false, new[] { 384, 512 }, true, new[] { 8 }, 3, false })]
    public async Task TestTileFlowCase(int[] ashape, bool constA, int[] bshape, bool constB, int[] eshape, bool constE, int[] hierarchy, int count, bool packing = false)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.Packing = packing;
        Expr a = constA ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate()) : new Var("a", new TensorType(DataTypes.Float32, ashape));
        Expr b = constB ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate()) : new Var("b", new TensorType(DataTypes.Float32, bshape));
        var c = IR.F.Tensors.MatMul(a, b);
        var d = IR.F.Math.Neg(c);
        Expr e = constE ? Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, eshape).Evaluate()) : new Var("e", new TensorType(DataTypes.Float32, eshape));
        var f = IR.F.Tensors.MatMul(d, e);

        var feedDict = new Dictionary<Var, IValue>();
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

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, new[] { f });
    }

    [Theory]
    [InlineData([new[] { 1, 77, 768 }, new[] { 2, 32, 4 }, new int[] { 2, -1, 2, 2, 2, -1 }, 0])]
    public async Task TestReshard(int[] shape, int[] hierarchy, int[] sbps, int count)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var inputType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(inputType);
        var feedDict = new Dictionary<Var, IValue>() {
            // { input, IR.F.Tensors.ConstantOfShape(shape, 1.0f).Evaluate() },
            { input, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, shape).Evaluate() },
        };

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var ndsbps = sbps.Chunk(hierarchy.Length).Select<int[], SBP[]>(sbp => sbp.Select<int, SBP>(s => s > 0 ? SBP.S(s) : SBP.B).ToArray()).ToArray();
        Expr boxed = input;
        foreach (var ndsbp in ndsbps)
        {
            boxed = IR.F.CPU.Boxing(boxed, new DistributedType(inputType, ndsbp, placement));
        }

        var post = IR.F.CPU.Boxing(boxed, inputType);
        post.Metadata = new Passes.Distributed.AutoDistributedMetadata(true);
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, new[] { post });
    }

    [Theory]
    [InlineData([new[] { 32, 64 }, new[] { 2 }, 0])]
    [InlineData([new[] { 8, 4 }, new[] { 4, 2 }, 1])]
    [InlineData([new[] { 32, 64, 128 }, new[] { 8, 4, 2 }, 2])]
    [InlineData([new[] { 64, 128 }, new[] { 2, 4, 8 }, 3])]
    public async Task TestGatherReduceScatter(int[] shape, int[] hierarchy, int count)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var inputType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(inputType);
        var feedDict = new Dictionary<Var, IValue>() {
            // { input, IR.F.Tensors.ConstantOfShape(shape, 1.0f).Evaluate() },
            { input, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, shape).Evaluate() },
        };

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var ndsbp = Enumerable.Repeat<SBP>(SBP.B, hierarchy.Length).ToArray();
        var posts = new List<Call>();
        var broadcast = IR.F.CPU.Boxing(input, new DistributedType(inputType, ndsbp, placement));
        foreach (var comb in LinqUtility.Combination(hierarchy.Length))
        {
            var newsbp = ndsbp.ToArray();
            foreach (var axis in comb)
            {
                newsbp[axis] = SBP.P();
            }

            var partial = IR.F.CPU.ForceBoxing(broadcast, new DistributedType(inputType, newsbp, placement));
            var sumed = IR.F.CPU.Boxing(partial, new DistributedType(inputType, ndsbp, placement));
            var post = IR.F.CPU.Boxing(sumed, inputType);
            post.Metadata = new Passes.Distributed.AutoDistributedMetaData() { Skip = true };
            posts.Add(post);
        }

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Fact]
    public async Task TestPartialReshapeBoxing()
    {
        var hierarchy = new[] { 2, 4 };
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        var lhsType = new TensorType(DataTypes.Float32, new[] { 1, 4, 8 });
        var rhsType = new TensorType(DataTypes.Float32, new[] { 8, 16 });
        var lhs = new Var(lhsType);
        var rhs = new Var(rhsType);

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, lhsType.Shape.ToValueArray()).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 1.0f, 1.0f, 1, rhsType.Shape.ToValueArray()).Evaluate() },
        };

        var placement = new Placement(hierarchy, targetOptions.HierarchyNames);
        var lhsBoxing = IR.F.CPU.Boxing(lhs, new DistributedType(lhsType, new SBP[] { SBP.S(2), SBP.B }, placement));
        var rhsBoxing = IR.F.CPU.Boxing(rhs, new DistributedType(rhsType, new SBP[] { SBP.S(0), SBP.S(1) }, placement));
        var matmul = IR.F.Tensors.MatMul(lhsBoxing, rhsBoxing);
        var newShape = new[] { 1, 4, 8, 2 };
        var reshape = IR.F.CPU.Boxing(matmul, new DistributedType(new TensorType(DataTypes.Float32, newShape), new SBP[] { SBP.B, SBP.S(2) }, placement), true);
        var sumed = IR.F.CPU.Boxing(reshape, new DistributedType(new TensorType(DataTypes.Float32, newShape), new SBP[] { SBP.S(1), SBP.S(2) }, placement));
        var post = IR.F.CPU.Boxing(sumed, new TensorType(DataTypes.Float32, newShape));
        post.Metadata = new Passes.Distributed.AutoDistributedMetaData() { Skip = true };

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{0}"), feedDict, new[] { post });
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
            { a, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, ashape).Evaluate() },
            { b, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, bshape).Evaluate() },
            { d, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, dshape).Evaluate() },
            { f, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, fshape).Evaluate() },
        };

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
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 4, 8, 16, 32 }, new[] { 1 }, 0 })]
    [InlineData(new object[] { new[] { 4, 8, 16, 32 }, new[] { 2 }, 1 })]
    [InlineData(new object[] { new[] { 4, 8, 16, 32 }, new[] { 4 }, 2 })]
    public async Task TestUnary(int[] shape, int[] hierarchy, int count)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);
        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackUnary(Rank, Lane);
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 8, 2 }, new int[] { 8, 2 }, new int[] { 1 }, new int[] { }, 0 })] // normal
    [InlineData(new object[] { BinaryOp.Mul, new[] { 1, 8, 64, 2 * 8 }, new int[] { 1, 1, 64, 2 * 8 }, new int[] { 1 }, new int[] { }, 1 })] // broadcast
    [InlineData(new object[] { BinaryOp.Add, new[] { 8, 16 }, new int[] { 16 }, new int[] { 1 }, new int[] { }, 2 })] // normal
    [InlineData(new object[] { BinaryOp.Mul, new[] { 1, 8, 64, 2 * 8 }, new int[] { 1, 1, 64, 2 * 8 }, new[] { 4 }, new[] { 1 }, 3 })] // broadcast
    public async Task TestPackBinary(BinaryOp op, int[] lhsShape, int[] rhsShape, int[] hierarchy, int[] sbps, int count)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

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

        if (sbps.Any(i => i != -1))
        {
            foreach (var post in posts)
            {
                var call = ExprCollector.Collect(post).Where(e => e is Call { Target: IR.CPU.PackedBinary or IR.Math.Binary }).First();
                call.Metadata = new() { OutputNames = new[] { "call" } };
            }

            var scheme = new Passes.Distributed.DistributedSchema("1", "llama", [new("call", sbps.Select<int, SBP>(s => s > 0 ? SBP.S(s) : SBP.B).ToArray(), hierarchy, targetOptions.HierarchyNames)]);
            var export = System.Text.Json.JsonSerializer.Serialize(scheme, new System.Text.Json.JsonSerializerOptions() { WriteIndented = true });
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

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory(Skip = "Drop InstanceNorm")]
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
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
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
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 384, 512 }, new[] { 512, 512 }, false, false, new[] { 1 }, 0 })]
    [InlineData(new object[] { new[] { 1, 1, 384, 256 }, new[] { 32, 256, 512 }, false, false, new[] { 1 }, 1 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 512 }, false, false, new[] { 1 }, 2 })]
    [InlineData(new object[] { new[] { 1, 384, 512 }, new[] { 512, 512 }, false, true, new[] { 1 }, 3 })]
    [InlineData(new object[] { new[] { 1, 1, 384, 256 }, new[] { 32, 256, 512 }, false, true, new[] { 1 }, 4 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 512 }, false, true, new[] { 1 }, 5 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 256 }, false, true, new[] { 2 }, 6 })]
    [InlineData(new object[] { new[] { 384, 512 }, new[] { 512, 512 }, false, true, new[] { 2, 4 }, 7 })]
    public async Task TestPackMatMul(int[] lhsShape, int[] rhsShape, bool constA, bool constB, int[] hierarchy, int count)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
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

        var feedDict = new Dictionary<Var, IValue>();
        if (!constA)
        {
            feedDict.Add((Var)lhs, Value.FromTensor(lhsTensor));
        }

        if (!constB)
        {
            feedDict.Add((Var)rhs, Value.FromTensor(rhsTensor));
        }

        var rule = new Passes.Rules.CPU.PackMatMul(2, Lane, transB: true);
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
    [InlineData([ReduceOp.Sum, new[] { 1, 64, 384, 128 }, new[] { 3 }, 0, true, new[] { 1 }, new int[] { }, 0])]
    [InlineData([ReduceOp.Mean, new[] { 1, 384, 128 }, new[] { 2 }, 0, true, new[] { 1 }, new int[] { }, 1])]
    [InlineData([ReduceOp.Mean, new[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, new int[] { 2 }, 2])]
    [InlineData([ReduceOp.Max, new[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, new int[] { 2 }, 3])]
    [InlineData([ReduceOp.Min, new[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, new int[] { 2 }, 4])]
    [InlineData([ReduceOp.Sum, new[] { 1, 384, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, new int[] { 2 }, 5])]
    [InlineData([ReduceOp.Mean, new[] { 1, 3, 1024 }, new[] { 2 }, 0, true, new[] { 4 }, new int[] { 2 }, 6])]
    [InlineData([ReduceOp.Sum, new[] { 1, 64, 384, 384 }, new[] { 3 }, 0, true, new[] { 64 }, new int[] { }, 7])]
    public async Task TestPackReduce(ReduceOp reduceOp, int[] shape, int[] axes, float init, bool keepDims, int[] hierarchy, int[] splitedAxes, int number)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();

        var tensorType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(tensorType);
        var pre = IR.F.Tensors.Reduce(reduceOp, input, axes, init, keepDims);

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };

        IEnumerable<Expr> posts;
        var rule = new Passes.Rules.CPU.PackReduce(Rank, Lane);
        if (!CompilerServices.TryMatch(pre, rule.Pattern, out var result))
        {
            return;
        }

        posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result, new Passes.RunPassContext()));

        if (splitedAxes.Any(i => i != -1))
        {
            foreach (var post in posts)
            {
                if (post is Call { Target: IR.CPU.Unpack } callUnPack && callUnPack.Arguments[0] is Call { Target: IR.CPU.PackedReduce } packedReduceCall)
                {
                    packedReduceCall.Arguments[0].Metadata = new() { OutputNames = new[] { "reduceIn" } };
                }
                else if (post is Call { Target: IR.Math.Reduce } reduceCall)
                {
                    reduceCall.Arguments[0].Metadata = new() { OutputNames = new[] { "reduceIn" } };
                }
            }

            var scheme = new Passes.Distributed.DistributedSchema("1", "llama", [new("reduceIn", splitedAxes.Select<int, SBP>(s => s > 0 ? SBP.S(s) : SBP.B).ToArray(), hierarchy, targetOptions.HierarchyNames)]);
            var export = System.Text.Json.JsonSerializer.Serialize(scheme, new System.Text.Json.JsonSerializerOptions() { WriteIndented = true });
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

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, posts);
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
    [InlineData([new[] { 1, 384, 8192 }, new[] { 1, 384, 64, 128 }, 1, new[] { 1 }, 0])]
    [InlineData([new[] { 1, 8192, 384 }, new[] { 1, 64, 128, 384 }, 1, new[] { 1 }, 1])]
    [InlineData([new[] { 1, 8192, 384 }, new[] { 1, 64, 128, 384 }, 1, new[] { 8 }, 2])]
    public async Task TestPackReshape(int[] inshape, int[] outshape, int packRank, int[] hierarchy, int number)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
        targetOptions.HierarchyLatencies = Enumerable.Repeat(1, hierarchy.Length).ToArray();
        targetOptions.HierarchyBandWidths = Enumerable.Repeat(1, hierarchy.Length).ToArray();
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
        var posts = new[] { pre }.Concat(rule.GetReplaceCandidates(result!, new Passes.RunPassContext()));
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

    [Theory]
    [InlineData([new[] { 2, 4 }, 0])]
    public async Task TestTransposeMatmul(int[] hierarchy, int number)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Packing = true;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
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

        var feedDict = new Dictionary<Var, IValue>() {
            { v13, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v13.CheckedShape).Evaluate() },
            { v15, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v15.CheckedShape).Evaluate() },
            { v19, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v19.CheckedShape).Evaluate() },
            { v24, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, v24.CheckedShape).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, posts);
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
    [InlineData(new object[] { new[] { 1, 48, 512 }, new[] { 1, 512, 1024 }, new[] { 1, 48, 64, 16 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 8 }, 0 })]
    [InlineData(new object[] { new[] { 1, 48, 512 }, new[] { 1, 512, 1024 }, new[] { 1, 64, 768 }, new[] { UnaryOp.Neg, UnaryOp.Cos }, new[] { 8 }, 1 })]
    public async Task TestMatMulReshapeUnary(int[] lhsShape, int[] rhsShape, int[] newShape, UnaryOp[] unaryOps, int[] hierarchy, int number)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;
        targetOptions.Hierarchies[0] = hierarchy;
        targetOptions.HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        targetOptions.HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
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

        var feedDict = new Dictionary<Var, IValue>()
        {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, rhsShape).Evaluate() },
        };

        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{number}"), feedDict, new[] { unary });
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

        ((CpuTargetOptions)CompileOptions.TargetOptions).Packing = packing;
        var hierarchy = new[] { 2, 4 };
        ((CpuTargetOptions)CompileOptions.TargetOptions).Hierarchies[0] = hierarchy;
        ((CpuTargetOptions)CompileOptions.TargetOptions).HierarchyNames = string.Join(string.Empty, "cbt".TakeLast(hierarchy.Length));
        ((CpuTargetOptions)CompileOptions.TargetOptions).HierarchySizes = Enumerable.Repeat((int)MathF.Pow(2, 30), hierarchy.Length).ToArray();
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

    // [InlineData(new object[] { false, 0 })]
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
        main.Metadata = fusion.Body.Metadata;

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
