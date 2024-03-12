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

public interface ICpuKernelCase
{
    string Name { get; }

    Fusion Fusion { get; }

    IReadOnlyList<Var> Vars { get; }

    IReadOnlyList<Tensor> Inputs { get; }

    public static Placement DefaultPlacement { get; } = new Placement(new[] { 1 }, "t");
}

public sealed class PackUnpackCaseData : TheoryData<ICpuKernelCase>
{
    public PackUnpackCaseData()
    {
        if (RuntimeInformation.OSArchitecture == Architecture.Arm64 && RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            // Add(new PackUnpackCase("PackUnpack0", new[] { 1, 32, 128 }, new[] { 32, 32 }, new[] { 1, 2 }));
            Add(new PackUnpackCase("PackUnpack0", new[] { 1, 32, 128 }, new[] { 4 }, new[] { 1 }));
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            Add(new PackUnpackCase("PackUnpack0", new[] { 1, 32, 128 }, new[] { 8 }, new[] { 1 }));
            Add(new PackUnpackCase("PackUnpack1", new[] { 1, 32, 128 }, new[] { 8 }, new[] { 2 }));
        }
    }
}

public sealed class PackLayerNormCaseData : TheoryData<ICpuKernelCase>
{
    public PackLayerNormCaseData()
    {
        var lane = 4;
        if (Vector256.IsHardwareAccelerated)
        {
            lane = 8;
        }

        Add(new PackLayerNormCase("PackLayerNorm0", new[] { 1, 16, 2 }, 1, new[] { 1 }, lane)); // pack within the axis
    }
}

public sealed class PackSoftMaxCaseData : TheoryData<ICpuKernelCase>
{
    public PackSoftMaxCaseData()
    {
        var lane = 4;
        if (Vector256.IsHardwareAccelerated)
        {
            lane = 8;
        }

        Add(new PackSoftMaxCase("PackSoftMax0", new[] { 1, 16, 2 }, 1, lane, new[] { 1 })); // pack axis == axis
        Add(new PackSoftMaxCase("PackSoftMax1", new[] { 1, 16, 32 }, 2, lane, new[] { 2 })); // pack axis == axis
    }
}

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestCPUKernels : TestClassBase
{
    public static readonly TheoryData<ICpuKernelCase> Cases = new()
    {
    };

    public UnitTestCPUKernels()
    {
        DefaultTargetName = CPUTarget.Kind;
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Theory]
    [ClassData(typeof(PackUnpackCaseData))]
    [ClassData(typeof(PackLayerNormCaseData))]
    [ClassData(typeof(PackSoftMaxCaseData))]
    internal async Task Run(ICpuKernelCase kernelCase)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return;
        }

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

internal sealed class PackUnpackCase : ICpuKernelCase
{
    public PackUnpackCase(string name, int[] inShape, int[] lanes, int[] packAxes)
    {
        Name = name;
        var type = new TensorType(DataTypes.Float32, inShape);
        var input = new Var(type);
        {
            var l0 = IR.F.CPU.Boxing(input, new DistributedType(type, new SBP[] { SBP.B }, ICpuKernelCase.DefaultPlacement));
            var packed = IR.F.CPU.Pack(l0, lanes, packAxes);
            var unpacked = IR.F.CPU.Unpack(packed, packAxes);
            Fusion = new Fusion(Name + "_kernel", CPUTarget.Kind, IR.F.CPU.Boxing(unpacked, type), new[] { input });
        }

        Vars = new[] { input };
    }

    public string Name { get; }

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

internal sealed class PackSoftMaxCase : ICpuKernelCase
{
    public PackSoftMaxCase(string name, int[] shape, int axis, int lane, int[] packedAxes)
    {
        Name = name;
        var inputType = new TensorType(DataTypes.Float32, shape);
        var input = new Var(inputType);
        Vars = new[] { input };
        {
            var finput = IR.F.CPU.Boxing(input, new DistributedType(inputType, new SBP[] { SBP.B }, ICpuKernelCase.DefaultPlacement));
            var lanes = Enumerable.Repeat(lane, packedAxes.Length).ToArray();
            var packed = IR.F.CPU.Pack(PackUtility.PadForPack(finput, shape, packedAxes, lanes, float.NegativeInfinity, out var pads), lanes, packedAxes);
            var softmax = IR.F.CPU.PackedSoftmax(packed, axis, packedAxes);
            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(softmax, packedAxes), shape, pads);

            Fusion = new Fusion(Name + "_kernel", CPUTarget.Kind, IR.F.CPU.Boxing(post, inputType), Vars.ToArray());
        }
    }

    public string Name { get; }

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

internal sealed class PackLayerNormCase : ICpuKernelCase
{
    public PackLayerNormCase(string name, int[] shape, int axis, int[] packedAxes, int lane)
    {
        Name = name;
        var inputType = new TensorType(DataTypes.Float32, shape);
        Expr input = new Var(inputType);
        var pshape = shape.Skip(axis).ToArray();
        var scaleType = new TensorType(DataTypes.Float32, pshape);
        Expr scale = new Var(scaleType);
        var biasType = new TensorType(DataTypes.Float32, pshape);
        Expr bias = new Var(biasType);
        Vars = new[] { (Var)input, (Var)scale, (Var)bias };
        {
            input = IR.F.CPU.Boxing(input, new DistributedType(inputType, new SBP[] { SBP.B }, ICpuKernelCase.DefaultPlacement));
            scale = IR.F.CPU.Boxing(scale, new DistributedType(scaleType, new SBP[] { SBP.B }, ICpuKernelCase.DefaultPlacement));
            bias = IR.F.CPU.Boxing(bias, new DistributedType(biasType, new SBP[] { SBP.B }, ICpuKernelCase.DefaultPlacement));

            var lanes = Enumerable.Repeat(lane, packedAxes.Length).ToArray();
            var packedInput = IR.F.CPU.Pack(PackUtility.PadForPack(input, shape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var pAxes = packedAxes.Where(i => i >= axis).Select(i => i - axis).ToArray();
            var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.CPU.Pack(packedScale, Enumerable.Repeat(lane, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.CPU.Pack(packedBias, Enumerable.Repeat(lane, pAxes.Length).ToArray(), pAxes);
            }

            var layernorm = IR.F.CPU.PackedLayerNorm(packedInput, packedScale, packedBias, axis, 1e-6f, true, packedAxes, padsInput);

            var post = PackUtility.SliceForPack(IR.F.CPU.Unpack(layernorm, packedAxes), shape, padsInput);

            Fusion = new Fusion(Name + "_kernel", CPUTarget.Kind, IR.F.CPU.Boxing(post, inputType), Vars.ToArray());
        }
    }

    public string Name { get; }

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


#if false

public sealed class MatMulCaseGenerator : TheoryData<ICpuKernelCase>
{
    public MatMulCaseGenerator(int[] lhsShape, int[] rhsShape)
    {
        var lhs = new TensorType(DataTypes.Float32, lhsShape);
        var rhs = new TensorType(DataTypes.Float32, rhsShape);
        var outputType = new TensorType(DataTypes.Float32, lhsShape.SkipLast(1).Concat(rhsShape.TakeLast(1)).ToArray());
        var place = new Placement(new[] { 8, 4 }, "bt");
        var lhsTypes = DistributedUtility.GetLeafCandidateNDSBPs(lhs, place);
        var rhsTypes = DistributedUtility.GetLeafCandidateNDSBPs(rhs, place);
        int count = 0;
        foreach (var lhsType in lhsTypes.Select(ndsbp => new DistributedType(lhs, ndsbp, place)))
        {
            foreach (var rhsType in rhsTypes.Select(ndsbp => new DistributedType(rhs, ndsbp, place)))
            {
                var isConsts = new[] { false, false };
                Add(new MatMulCase($"gen_{count++}", lhsType, rhsType, isConsts, outputType));
                foreach (var constIndex in new[] { 0, 1 })
                {
                    var tp = isConsts.ToArray();
                    tp[constIndex] = true;
                    Add(new MatMulCase($"gen_{count++}", lhsType, rhsType, tp, outputType));
                }
            }
        }
    }
}

internal sealed class MatMulCase : ICpuKernelCase
{
    private readonly string _name;
    private readonly DistributedType _lhsType;
    private readonly DistributedType _rhsType;
    private readonly bool[] _isConsts;
    private readonly TensorType _outputType;

    public MatMulCase(string count, DistributedType lhsType, DistributedType rhsType, ReadOnlySpan<bool> isConsts, TensorType outputType)
    {
        _name = count;
        _lhsType = lhsType;
        _rhsType = rhsType;
        _isConsts = isConsts.ToArray();
        _outputType = outputType;
        Vars = new Var[] { new Var(_lhsType.TensorType), new Var(_rhsType.TensorType) };
    }

    public string Name => $"MatmulCase_{_name}";

    public Fusion Fusion
    {
        get
        {
            Expr l0 = IR.F.XPU.Boxing(_isConsts[0] ? Const.FromValue(IR.F.Random.Normal(_lhsType.TensorType.DType, 0, 1, 0, _lhsType.TensorType.Shape.ToValueArray()).Evaluate()) : Vars[0], _lhsType);
            Expr r0 = IR.F.XPU.Boxing(_isConsts[1] ? Const.FromValue(IR.F.Random.Normal(_rhsType.TensorType.DType, 0, 1, 2, _rhsType.TensorType.Shape.ToValueArray()).Evaluate()) : Vars[1], _rhsType);

            var body = IR.F.Math.MatMul(l0, r0);
            var d = (DistributedType)body.CheckedType;
            if (d.NdSBP.Any(s => s is SBPPartialSum))
            {
                body = IR.F.XPU.Boxing(body, new DistributedType(d.TensorType, d.NdSBP.Select(s => s is SBPPartialSum ? SBP.B : s).ToArray(), d.Placement));
            }

            var fusion = new Fusion(Name + "_kernel", XPUTarget.Kind, IR.F.XPU.Boxing(body, _outputType), Enumerable.Range(0, 2).Zip(Vars).Where(p => !_isConsts[p.First]).Select(p => p.Second).ToArray());

            return fusion;
        }
    }

    public IReadOnlyList<Var> Vars { get; }

    public IReadOnlyList<Tensor> Inputs => Enumerable.Range(0, 2).
        Zip(Vars).
        Where(p => !_isConsts[p.First]).
        Select(p => (p.Second.CheckedDataType, p.Second.CheckedShape)).
        Select((p, i) => IR.F.Random.Normal(p.CheckedDataType, 0, 1, i, p.CheckedShape.ToValueArray()).Evaluate().AsTensor()).
        ToArray();
}


internal sealed class SoftmaxCase1 : ICpuKernelCase
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
                return new Fusion(Name + "_kernel", XPUTarget.Kind, IR.F.XPU.Boxing(IR.F.NN.Softmax(input0, axis), type), new[] { Vars[0] });
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

internal sealed class UnarySharedInputCase : ICpuKernelCase
{
    public UnarySharedInputCase()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 1, 16, 768 });
        var place = new Placement(new[] { 8, 4 }, "bt");
        var lhs = new Var(type);
        {
            var l0 = IR.F.XPU.Boxing(lhs, new DistributedType(type, new SBP[] { SBP.S(2), SBP.B }, place));
            Fusion = new Fusion(Name + "_kernel", XPUTarget.Kind, IR.F.XPU.Boxing(IR.F.Math.Cos(l0), type), new[] { lhs });
        }

        Vars = new[] { lhs };
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

internal sealed class UnaryNonUniformCase : ICpuKernelCase
{
    public UnaryNonUniformCase()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 1, 49 });
        var place = new Placement(new[] { 8, 4 }, "bt");
        var lhs = new Var(type);
        {
            var l0 = IR.F.XPU.Boxing(lhs, new DistributedType(type, new SBP[] { SBP.B, SBP.S(1) }, place));
            Fusion = new Fusion(Name + "_kernel", XPUTarget.Kind, IR.F.XPU.Boxing(IR.F.Math.Cos(l0), type), new[] { lhs });
        }

        Vars = new[] { lhs };
    }

    public string Name => "UnaryNonUniformCase";

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

internal sealed class ReshapeNonUniformCase : ICpuKernelCase
{
    public ReshapeNonUniformCase()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 77, 12, 64 });
        var input = new Var(type);
        Vars = new[] { input };
    }

    public string Name => "ReshapeNonUniformCase";

    public Fusion Fusion
    {
        get
        {
            var type = new TensorType(DataTypes.Float32, new[] { 77, 12, 64 });
            var place = new Placement(new[] { 8, 4 }, "bt");
            var l0 = IR.F.XPU.Boxing(Vars[0], new DistributedType(type, new SBP[] { SBP.S(1), SBP.S(0) }, place));
            var l2 = IR.F.XPU.Boxing(l0, new DistributedType(new TensorType(DataTypes.Float32, new[] { 1, 77, 768 }), new SBP[] { SBP.S(1), SBP.S(1) }, place));
            return new Fusion(Name + "_kernel", XPUTarget.Kind, IR.F.XPU.Boxing(l2, new TensorType(DataTypes.Float32, new[] { 1, 77, 768 })), new[] { Vars[0] });
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
#endif
