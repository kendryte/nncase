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
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Runtime.Interop;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.Tensors;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Tests.Targets;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCPUTargetTiling : TestClassBase
{
    public UnitTestCPUTargetTiling()
    {
        DefaultTargetName = CPUTarget.Kind;
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    [Fact]
    public async Task TestCpuUnary()
    {
        var shape = new[] { 1, 384, 2048 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var main = new Function("main", IR.F.Math.Unary(UnaryOp.Neg, input), new[] { input });
        var module = new IR.IRModule(main);
        await Compile(module);

        var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, shape).Evaluate().AsTensor();
        using (var fs = Dumpper.OpenFile("input_0.bin"))
        {
            fs.Write(input_tensor.BytesBuffer);
        }

        Testing.RunKModel(File.ReadAllBytes(Path.Join(Dumpper.Directory, "test.kmodel")), Dumpper.Directory, new[] { input_tensor });
    }

    [Fact]
    public async Task TestCpuBinary()
    {
        var lhsShape = new[] { 1, 64, 384, 128 };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, lhsShape));
        var rhsShape = new[] { 1, 1, 384, 128 };
        var rhs = new Var("lhs", new TensorType(DataTypes.Float32, rhsShape));
        var main = new Function("main", lhs * rhs, new[] { lhs, rhs });
        var module = new IR.IRModule(main);
        await Compile(module);
    }

    [Fact]
    public async Task TestCpuMatMul()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 1, 1, 384, 8192 }));
        var rhs = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 64, 8192, 128 });

        var main = new Function("main", IR.F.Tensors.MatMul(lhs, rhs), new[] { lhs });
        var module = new IR.IRModule(main);
        await Compile(module);

        // var input_tensor = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { 3, 4 }).Evaluate().AsTensor();
        // using (var fs = Dumpper.OpenFile("input_0.bin"))
        // {
        //     fs.Write(input_tensor.BytesBuffer);
        // }

        // Testing.RunKModel(File.ReadAllBytes(Path.Join(Dumpper.Directory, "test.kmodel")), Dumpper.Directory, new[] { input_tensor });
    }

    private async Task Compile(IRModule module)
    {
        var compiler = CompileSession.Compiler;
        compiler.ImportIRModule(module);
        await compiler.CompileAsync();
        using (var fs = Dumpper.OpenFile("test.kmodel"))
        {
            compiler.Gencode(fs);
        }
    }
}
