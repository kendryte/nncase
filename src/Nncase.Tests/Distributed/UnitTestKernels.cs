// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
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

namespace Nncase.Tests.DistributedTest;

public interface IDistributedKernelCase
{
    string Name { get; }

    Fusion Fusion { get; }

    IReadOnlyList<Var> Vars { get; }

    IReadOnlyList<Tensor> Inputs { get; }
}

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestKernels : TestClassBase
{
    public static readonly TheoryData<IDistributedKernelCase> Cases = new()
    {
        new BinaryCase1(),
        new SoftmaxCase1(),
    };

    public UnitTestKernels()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Theory]
    [MemberData(nameof(Cases))]
    internal async Task Run(IDistributedKernelCase kernelCase)
    {
        using var dumpScope = new Diagnostics.DumpScope(kernelCase.Name, CompileOptions.DumpFlags);

        // convert fusion to prim func
        var primBody = new List<Expr>();
        var visitor = new Passes.Tile.TIRConvertVisitor(primBody);
        var fusion = kernelCase.Fusion;
        visitor.Visit(fusion);
        var primFunc = TIR.T.PrimFunc(fusion.Name, fusion.ModuleKind, visitor.InputBuffers.Concat(visitor.OutputBuffers).ToArray()).Body(primBody.ToArray()).Build();
        var primWrapper = new PrimFunctionWrapper(primFunc, primFunc.Parameters.Length - 1);
        var main = new Function(new Call(primWrapper, kernelCase.Vars.ToArray()), kernelCase.Vars.ToArray());

        var module = new IR.IRModule(main);
        module.Add(primWrapper);
        module.Add(primFunc);
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
        Assert.True(cos > 0.999);
    }

    private async Task Compile(IRModule module)
    {
        var pmgr = CompileSession.CreatePassManager("pmgr");
        CompileSession.Target.RegisterTargetDependentAfterQuantPass(pmgr, CompileSession.CompileOptions);
        CompileSession.Target.RegisterTargetDependentBeforeCodeGen(pmgr, CompileSession.CompileOptions);
        await pmgr.RunAsync(module);
    }
}

internal sealed class BinaryCase1 : IDistributedKernelCase
{
    public BinaryCase1()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 1, 16, 768 });
        var place = new Placement(new[] { 8, 4 }, "bt");
        var lhs = new Var(type);
        var rhs = new Var(type);
        {
            var l0 = IR.F.XPU.Boxing(lhs, new DistributedType(type, new SBP[] { SBP.S(2), SBP.S(2) }, place));
            var r0 = IR.F.XPU.Boxing(rhs, new DistributedType(type, new SBP[] { SBP.S(2), SBP.S(2) }, place));
            Fusion = new Fusion(Name + "_fusion", XPUTarget.Kind, IR.F.XPU.Boxing(l0 + r0, type), new[] { lhs, rhs });
        }

        Vars = new[] { lhs, rhs };
    }

    public string Name => "BinaryCase1";

    public Fusion Fusion { get; }

    public IReadOnlyList<Var> Vars { get; }

    public IReadOnlyList<Tensor> Inputs
    {
        get
        {
            return Vars.Select(v => IR.F.Random.Uniform(v.CheckedDataType, 30, 0, 1, v.CheckedShape).Evaluate().AsTensor()).ToArray();
        }
    }

    public override string ToString() => Name;
}

internal sealed class SoftmaxCase1 : IDistributedKernelCase
{
    public SoftmaxCase1()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 16, 1024, 1024 });
        var input = new Var(type);
        Vars = new[] { input };
    }

    public string Name => "SoftmaxCase1";

    public Fusion Fusion
    {
        get
        {
            var type = new TensorType(DataTypes.Float32, new[] { 16, 1024, 1024 });
            var place = new Placement(new[] { 8, 4 }, "bt");
            var axis = 2L;
            {
                var input0 = IR.F.XPU.Boxing(Vars[0], new DistributedType(type, new SBP[] { SBP.S(0), SBP.S(1) }, place));
                return new Fusion(Name + "_fusion", XPUTarget.Kind, IR.F.XPU.Boxing(IR.F.NN.Softmax(input0, axis), type), new[] { Vars[0] });
            }
        }
    }

    public IReadOnlyList<Var> Vars { get; }

    public IReadOnlyList<Tensor> Inputs
    {
        get
        {
            return Vars.Select(v => IR.F.Random.Uniform(v.CheckedDataType, 30, 0, 1, v.CheckedShape).Evaluate().AsTensor()).ToArray();
        }
    }

    public override string ToString() => Name;
}
