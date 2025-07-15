// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestPackWhere : TransformTestBase
{
    [Fact]
    public void UnitTestPackWherePropagation()
    {
        var cond = Testing.Rand<bool>(1, 24);
        var condVar = new Var(new TensorType(cond.ElementType, cond.Shape));
        var lhs = Testing.Rand<float>(24);
        var lhsVar = new Var(new TensorType(lhs.ElementType, lhs.Shape));
        var rhs = Testing.Rand<float>(1, 24);
        var rhsVar = new Var(new TensorType(rhs.ElementType, rhs.Shape));
        Expr expr = Where(condVar, lhsVar, rhsVar);
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        TestMatchedCore(
            expr,
            new Dictionary<IVar, IValue> {
                { condVar, Value.FromTensor(cond) },
                { lhsVar, Value.FromTensor(lhs) },
                { rhsVar, Value.FromTensor(rhs) },
            },
            new PackWherePropagation(MaskVectorStyle.Fat));
    }

    [Fact]
    public void UnitTestPackDynamicWherePropagation()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 128) } };
        var cond = Testing.Rand<bool>(3, 1);
        var condVar = new Var(new TensorType(cond.ElementType, [dimX, 1]));
        var lhs = Testing.Rand<float>(1);
        var lhsVar = new Var(new TensorType(lhs.ElementType, [1]));
        var rhs = Testing.Rand<float>(3, 24);
        var rhsVar = new Var(new TensorType(rhs.ElementType, [dimX, 24]));
        Expr expr = Where(condVar, lhsVar, rhsVar);
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        TestMatchedCore(
            expr,
            new Dictionary<IVar, IValue> {
                { condVar, Value.FromTensor(cond) },
                { lhsVar, Value.FromTensor(lhs) },
                { rhsVar, Value.FromTensor(rhs) },
            },
            new PackWherePropagation(MaskVectorStyle.Fat));
    }

    [Fact]
    public void UnitTestPackDynamicWherePropagationEGraph()
    {
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite;
        var dimX = new DimVar("x") { Metadata = { Range = (1, 1024) } };
        var cond = Testing.Rand<bool>(3, 1);
        var condVar = new Var(new TensorType(cond.ElementType, [dimX, 1]));
        var lhs = Testing.Rand<float>(1);
        var rhs = Testing.Rand<float>(3, 24);
        var rhsVar = new Var(new TensorType(rhs.ElementType, [dimX, 24]));
        Expr expr = Where(condVar, lhs, rhsVar);
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        var func = new Function("main", expr, [condVar, rhsVar]);
        var module = new IRModule(func);

        var pmgr = CompileSession.CreatePassManager("Pack");
        pmgr.Add<EGraphRulesPass>()
            .Configure(c =>
            {
                c.Add<PackWherePropagation>(MaskVectorStyle.Fat);
            });
        pmgr.RunAsync(module).Wait();
        Assert.True(module.Entry is Function { Body: Call { Target: IR.Tensors.Unpack, Arguments: var unpackArgs } }
            && unpackArgs[0] is Call { Target: IR.Tensors.Where });
    }
}
