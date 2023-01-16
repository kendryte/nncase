﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Tests.ReWriteTest;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Tensorboard;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.DiagnosticsTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestDumpper : TestClassBase
{
    public UnitTestDumpper()
    {
        CompileOptions.DumpFlags = DumpFlags.ImportOps | DumpFlags.Evaluator | DumpFlags.EGraphCost
            | DumpFlags.Calibration | DumpFlags.Compile | DumpFlags.PassIR | DumpFlags.Rewrite;
    }

    [Fact]
    public void TestDumpIR()
    {
        Var x = "x", y = "y";
        var z = x + y;
        var tuple = new IR.Tuple(x, y, z);
        CompilerServices.InferenceType(tuple);

        Dumpper.DumpIR(tuple, "main");
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "main_Tuple.il")));
    }

    [Fact]
    public void TestDumpFusion()
    {
        var fusionCase = new ReWrite.FusionTest.DataFlowType7FusionCaseLeft();
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function("main", fusionCase.BuildBody(input), new[] { input });
        CompilerServices.InferenceType(main);

        Dumpper.DumpIR(main, string.Empty);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "main.il")));

        Dumpper.DumpDotIR(main, string.Empty);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "main.dot")));
    }

    [Fact]
    public void TestDumpScript()
    {
        var prim_func_1 = T.PrimFunc("prim_func_1", "k?", T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Input, new[] { 1, 2, 3, 4 }, out _), T.PhysicalBuffer(DataTypes.Float32, Schedule.MemoryLocation.Output, new[] { 1, 2, 3, 4 }, out _)).Body(T.Nop()).Build();

        Assert.True(CompilerServices.InferenceType(prim_func_1));

        Dumpper.DumpIR(prim_func_1, string.Empty);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "prim_func_1.script")));
    }

    [Fact]
    public void TestDumpModule()
    {
        var x = new Var("x", new TensorType(DataTypes.Float32, new[] { 1, 2 }));
        var y = new Var("y", new TensorType(DataTypes.Float32, new[] { 3, 2 }));
        var z = x + y;
        var tuple = new IR.Tuple(x, y, z);
        var module = new IRModule(new Function("main", tuple, new[] { x, y }));
        CompilerServices.InferenceType(module.Entry!);

        Dumpper.DumpModule(module);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "main.il")));
    }

    [Fact]
    public void TestDumpEGraph()
    {
        Expr a = (Const)1 + 2;
        Expr b = (Const)1 << 2;
        Expr c = a * b;
        var graph = new EGraph();
        graph.Add(c);
        using var fs = Dumpper.OpenFile("example.dot");
        EGraphPrinter.DumpEgraphAsDot(graph, fs);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "example.dot")));
    }

    [Fact]
    public async Task TestDumpDataflowRewrite()
    {
        var weights = new Var("weights", new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        _ = Util.ShapeIndex(weights, 0);
        var expand = Expand(0f, Cast(Util.ShapeIndex(weights, 0), DataTypes.Int64));
        await RunShapeInferPass("main", expand, weights);
        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "0_ShapeInfer_main", "main", "Start_main.il")));
    }

    [Fact]
    public void TestDumpEGraphRewrite()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 224, 224, 3 }));
        Expr pre;
        {
            var v_0 = Transpose(input, new[] { 0, 3, 1, 2 }); // f32[1,3,224,224]
            var v_1 = IR.F.Math.RangeOfMarker(v_0, new[] { -4.91261, 4.4099503 });
            var v_2 = IR.F.Math.RangeOfMarker(Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 224, 224 }).Evaluate()), new[] { -1.0, 1.0 });
            var v_3 = v_1 * v_2;
            var v_4 = IR.F.Math.RangeOfMarker(v_3, new[] { -6.8198624, 7.4711213 });
            pre = v_4;
        }

        Assert.True(pre.InferenceType());

        _ = CompilerServices.ERewrite(pre, new IRewriteRule[]
        {
              new Transform.Rules.Lower.RemoveMarker(),
              new TestMulToAdd(),
        }, new());

        Assert.True(File.Exists(Path.Join(Dumpper.Directory, "Costs", "V4.dot")));
    }

    [Fact]
    public async Task TestSubDumperDumpFlags()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 3, 224, 224 }));
        Function main;
        {
            var weights = Const.FromValue(IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 64, 3, 3, 3 }).Evaluate());
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = 2;
            var strideW = 2;
            var dilationH = 1;
            var dilationW = 1;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, true);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, true);
            var padding = Stack(
              new IR.Tuple(
                Stack(new IR.Tuple(new Expr[] { 0, 0 }), 0),
                Stack(new IR.Tuple(new Expr[] { 0, 0 }), 0),
                Stack(new IR.Tuple(padH), 0),
                Stack(new IR.Tuple(padW), 0)),
              0);
            var body = IR.F.NN.Pad(input, padding, PadMode.Constant, 0.0f);
            main = new Function("main", body, ImmutableArray.Create(input));
        }

        var pass = new ShapeInferPass { Name = $"ShapeInfer" };

        using (var _ = new DumpScope("DisableEvaluator", DumpFlags.ImportOps | DumpFlags.EGraphCost
            | DumpFlags.Calibration | DumpFlags.Compile | DumpFlags.PassIR | DumpFlags.Rewrite))
        {
            var post = (Function)await pass.RunAsync(main, new());
        }

        Assert.False(Directory.Exists(Path.Join(Dumpper.Directory, "DisableEvaluator", "0_ShapeInfer", "main", "Run_0", "Evaluate")));
        Assert.True(Directory.Exists(Path.Join(Dumpper.Directory, "DisableEvaluator", "0_ShapeInfer", "main", "Run_0", "Rewrite")));

        using (var _ = new DumpScope("DisableRewrite", DumpFlags.ImportOps | DumpFlags.EGraphCost | DumpFlags.Evaluator
            | DumpFlags.Calibration | DumpFlags.Compile | DumpFlags.PassIR))
        {
            var post = (Function)await pass.RunAsync(main, new());
        }

        Assert.True(Directory.Exists(Path.Join(Dumpper.Directory, "DisableRewrite", "0_ShapeInfer", "main", "Run_0", "Evaluate")));
        Assert.False(Directory.Exists(Path.Join(Dumpper.Directory, "DisableRewrite", "0_ShapeInfer", "main", "Run_0", "Rewrite")));
    }

    private async Task<Expr> RunShapeInferPass(string name, Expr expr, params Var[] parameters)
    {
        var f = new Function(name, expr, parameters);
        var result = ((Function)await new ShapeInferPass { Name = $"ShapeInfer_{name}" }.RunAsync(f, new())).Body;
        Assert.True(CompilerServices.InferenceType(CompilerServices.InferenceType(f)));
        return result;
    }
}
