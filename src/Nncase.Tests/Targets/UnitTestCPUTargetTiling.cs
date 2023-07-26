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
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen;
#endif
    }

    [Fact]
    public async Task TestCpuUnary()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 2, 3, 4, 5 }));
        var main = new Function("main", IR.F.Math.Unary(UnaryOp.Asin, input), new[] { input });
        var module = new IR.IRModule(main);

        var compiler = CompileSession.Compiler;
        compiler.ImportIRModule(module);
        await compiler.CompileAsync();
        using (var fs = Dumpper.OpenFile("test.kmodel"))
        {
            compiler.Gencode(fs);
        }

        using (var fs = Dumpper.OpenFile("input_0.bin"))
        {
            fs.Write(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { 1, 2, 3, 4, 5 }).Evaluate().AsTensor().BytesBuffer);
        }
    }

    [Fact]
    public async Task TestCpuMatMul()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 3, 4 }));
        var rhs = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 4, 6 });

        // new Var("rhs", new TensorType(DataTypes.Float32, new[] { 4, 6 }));
        var main = new Function("main", IR.F.Tensors.MatMul(lhs, rhs), new[] { lhs });
        var module = new IR.IRModule(main);

        var compiler = CompileSession.Compiler;
        compiler.ImportIRModule(module);
        await compiler.CompileAsync();
        using (var fs = Dumpper.OpenFile("test.kmodel"))
        {
            compiler.Gencode(fs);
        }

        using (var fs = Dumpper.OpenFile("input_0.bin"))
        {
            fs.Write(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { 1, 2, 3, 4, 5 }).Evaluate().AsTensor().BytesBuffer);
        }
    }
}
