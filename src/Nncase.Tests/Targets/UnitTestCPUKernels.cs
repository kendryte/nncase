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

namespace Nncase.Tests.Targets;

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
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static Placement DefaultPlacement => new Placement(new[] { 1 }, "t");

    public static int Lane => Vector256.IsHardwareAccelerated ? 8 : 4;

    public static int Rank => 1;

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 64, 768 }, new int[] { 64, 768 }, 0 })] // normal
    public async void TestPackBinary(BinaryOp op, int[] lhsShape, int[] rhsShape, int count)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var rule = new Passes.Rules.CPU.PackBinary() { Lane = Lane, Rank = Rank };
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 384, 8192 }, 2, 1e-5, true, 0 })]
    public async Task TestLayerNorm(int[] shape, int axis, float epsion, bool useMean, int count)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = shape.Skip(axis).ToArray();
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.LayerNorm(axis, 1e-6f, input, scale, bias, false);

        var feedDict = new Dictionary<Var, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };

        // var rule = new Passes.Rules.CPU.PackLayerNorm() { Lane = Lane, Rank = Rank };
        // CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        // var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString(), $"Theory{count}"), feedDict, posts);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 384, 512 }, new[] { 512, 512 }, 0 })]
    [InlineData(new object[] { new[] { 1, 1, 384, 256 }, new[] { 32, 256, 512 }, 1 })]
    public async Task TestMatMul(int[] lhsShape, int[] rhsShape, int count)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<Var, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        // var rule = new Passes.Rules.CPU.PackMatMul();
        // CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        // var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var posts = new[] { pre };
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

    [Fact]
    public async Task TestDecodeLayer()
    {
        var vhidden_in = new Var("vhidden_in", new TensorType(DataTypes.Float32, new[] { 1, 384, 8192 }));
        var vattn_mask = new Var("vattn_mask", new TensorType(DataTypes.Float32, new[] { 1, 1, 384, 384 }));
        var vposition_ids = new Var("vposition_ids", new TensorType(DataTypes.Int64, new[] { 1, 384 }));
        Expr pre;
        {
            var v0 = IR.F.NN.LayerNorm(2, 1E-05f, vhidden_in, IR.F.Random.Normal(new[] { 8192 }).Evaluate().AsTensor(), IR.F.Random.Normal(new[] { 8192 }).Evaluate().AsTensor(), false); // f32[1,384,8192]
            var v1 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v2 = IR.F.Tensors.Reshape(v1, new long[] { 1L, 384L, 64L, 128L }); // f32[1,384,64,128]
            var v3 = IR.F.Tensors.Transpose(v2, new long[] { 0L, 2L, 1L, 3L }); // f32[1,64,384,128]
            var v4 = IR.F.Tensors.Gather(IR.F.Random.Normal(new[] { 384, 128 }).Evaluate().AsTensor(), 0, vposition_ids); // f32[1,384,128]
            var v5 = IR.F.Tensors.Reshape(v4, new[] { 1, 1, 384, 128 }); // f32[1,1,384,128]
            var v6 = IR.F.Math.Binary(BinaryOp.Mul, v3, v5); // f32[1,64,384,128]
            var v7 = IR.F.Tensors.Slice(v3, new long[] { 64L }, new long[] { 128L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v8 = IR.F.Math.Unary(UnaryOp.Neg, v7); // f32[1,64,384,64]
            var v9 = IR.F.Tensors.Slice(v3, new long[] { 0L }, new long[] { 64L }, new long[] { 3L }, new long[] { 1L }); // f32[1,64,384,64]
            var v10 = new IR.Tuple(v8, v9); // (f32[1,64,384,64], f32[1,64,384,64])
            var v11 = IR.F.Tensors.Concat(v10, 3); // f32[1,64,384,128]
            var v12 = IR.F.Tensors.Gather(IR.F.Random.Normal(new[] { 384, 128 }).Evaluate().AsTensor(), 0, vposition_ids); // f32[1,384,128]
            var v13 = IR.F.Tensors.Reshape(v12, new[] { 1, 1, 384, 128 }); // f32[1,1,384,128]
            var v14 = IR.F.Math.Binary(BinaryOp.Mul, v11, v13); // f32[1,64,384,128]
            var v15 = IR.F.Math.Binary(BinaryOp.Add, v6, v14); // f32[1,64,384,128]
            var v16 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
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
            var v32 = IR.F.Tensors.MatMul(v0, IR.F.Random.Normal(new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v33 = IR.F.Tensors.Reshape(v32, new long[] { 1L, 384L, 64L, 128L }); // f32[1,384,64,128]
            var v34 = IR.F.Tensors.Transpose(v33, new long[] { 0L, 2L, 1L, 3L }); // f32[1,64,384,128]
            var v35 = IR.F.Tensors.MatMul(v31, v34); // f32[1,64,384,128]
            var v36 = IR.F.Tensors.Transpose(v35, new long[] { 0L, 2L, 1L, 3L }); // f32[1,384,64,128]
            var v37 = IR.F.Tensors.Reshape(v36, new long[] { 1L, 384L, 8192L }); // f32[1,384,8192]
            var v38 = IR.F.Tensors.MatMul(v37, IR.F.Random.Normal(new[] { 8192, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v39 = IR.F.Math.Binary(BinaryOp.Add, vhidden_in, v38); // f32[1,384,8192]
            var v40 = IR.F.NN.LayerNorm(2, 1E-05f, v39, IR.F.Random.Normal(new[] { 8192 }).Evaluate().AsTensor(), IR.F.Random.Normal(new[] { 8192 }).Evaluate().AsTensor(), false); // f32[1,384,8192]
            var v41 = IR.F.Tensors.MatMul(v40, IR.F.Random.Normal(new[] { 8192, 22016 }).Evaluate().AsTensor()); // f32[1,384,22016]
            var v42 = IR.F.NN.Swish(v41, 1.0f); // f32[1,384,22016]
            var v43 = IR.F.Tensors.MatMul(v40, IR.F.Random.Normal(new[] { 8192, 22016 }).Evaluate().AsTensor()); // f32[1,384,22016]
            var v44 = IR.F.Math.Binary(BinaryOp.Mul, v42, v43); // f32[1,384,22016]
            var v45 = IR.F.Tensors.MatMul(v44, IR.F.Random.Normal(new[] { 22016, 8192 }).Evaluate().AsTensor()); // f32[1,384,8192]
            var v46 = IR.F.Math.Binary(BinaryOp.Add, v39, v45); // f32[1,384,8192]
            pre = v46;
        }

        var feedDict = new Dictionary<Var, IValue>() {
            { vhidden_in, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1,  new[] { 1, 384, 8192 }).Evaluate() },
            { vattn_mask, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1,  new[] { 1, 1, 384, 384 }).Evaluate() },
            { vposition_ids, IR.F.Random.Uniform(DataTypes.Int64, 383, 1, 1, new[] { 1, 384 }).Evaluate() },
        };

        var posts = new[] { pre };
        await RunCases(Path.Join(CompileOptions.DumpDir.ToString()), feedDict, posts);
    }

    internal async Task RunCases(string dumpDir, Dictionary<Var, IValue> feedDict, IEnumerable<Expr> posts)
    {
        int count = 0;
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            var kernelCase = new CpuKernelCase($"Case{count++}", new Fusion("kernel", CPUTarget.Kind, post, feedDict.Keys.ToArray()), feedDict.Keys.ToArray(), feedDict.Values.Select(v => v.AsTensor()).ToArray());
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

        var main = new Function(new Call(fusion, kernelCase.Vars.ToArray()), kernelCase.Vars.ToArray());

        var module = new IR.IRModule(main);
        var inputs = kernelCase.Inputs.ToArray();
        var output = fusion.Body.Evaluate(kernelCase.Vars.Zip(inputs).ToDictionary(p => p.First, p => (IValue)Value.FromTensor(p.Second))).AsTensor();

#if DEBUG
        for (var i = 0; i < inputs.Length; i++)
        {
            using (var fs = Diagnostics.DumpScope.Current.OpenFile($"input_{i}.bin"))
            {
                fs.Write(inputs[i].BytesBuffer);
            }
        }

        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"output_0.bin"))
        {
            fs.Write(output.BytesBuffer);
        }
#endif
        await Compile(module);
        var (kmodel_path, _) = Testing.BuildKModel("test", module, CompileSession, false);
        var actual = Testing.RunKModel(kmodel_path, Diagnostics.DumpScope.Current.Directory, inputs).AsTensor();
#if DEBUG
        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"actual_0.bin"))
        {
            fs.Write(actual.BytesBuffer);
        }
#endif
        var cos = Comparator.CosSimilarity(output, actual);
        Assert.True(cos > 0.999, $"the cos is {cos}");
    }

    private async Task Compile(IRModule module)
    {
        var pmgr = CompileSession.CreatePassManager("pmgr");
        CompileSession.Target.RegisterTargetDependentAfterQuantPass(pmgr, CompileSession.CompileOptions);
        CompileSession.Target.RegisterTargetDependentBeforeCodeGen(pmgr, CompileSession.CompileOptions);
        await pmgr.RunAsync(module);
    }
}
