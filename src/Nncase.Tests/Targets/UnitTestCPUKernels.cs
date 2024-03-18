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
    [InlineData(new object[] { BinaryOp.Add, new[] { 64, 768 }, new int[] { 64, 768 } })] // normal
    public async void TestPackBinary(BinaryOp op, int[] lhsShape, int[] rhsShape)
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
        int count = 0;
        var dump_dir = CompileOptions.DumpDir.ToString();
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            var kernelCase = new CpuKernelCase($"PackBinaryCase{count++}", new Fusion("kernel", CPUTarget.Kind, post, feedDict.Keys.ToArray()), feedDict.Keys.ToArray(), feedDict.Values.Select(v => v.AsTensor()).ToArray());
            await Run(dump_dir, kernelCase);
        }
    }

    internal async Task Run(string dumpDir, CpuKernelCase kernelCase)
    {
        CompileOptions.DumpDir = Path.Join(CompileOptions.DumpDir, kernelCase.Name);
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
        var (kmodel_path, kmodel) = Testing.BuildKModel("test", module, CompileSession);
        var actual = Testing.RunKModel(kmodel, Diagnostics.DumpScope.Current.Directory, inputs).AsTensor();
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
