// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestEGraphRewrite : TestClassBase
{
    [Fact]
    public void RewriteNoSenceAdd()
    {
        Var x = "a";
        var lhs = x + (100 / 120.0f) - 100;
        var y = lhs + 0;

        var pattern = PatternMatch.F.Math.Add(IsWildcard("lhs"), IsConst(0));

        var egraph = new EGraph();
        var root = egraph.Add(y);

        Assert.True(CompilerServices.TryMatchRoot(root.Nodes, pattern, out var eResults));
        Assert.Single(eResults);
        var wcxv = (Expr)eResults[0][pattern.Arguments[0]];
        Assert.Equal(wcxv, lhs);
        var to_eid = egraph.Add(wcxv);
        /*
          lhs + 0 <=> lhs
         */
        egraph.Union(to_eid, root);
        egraph.Rebuild();
    }

    [Fact]
    public void TestReassociate()
    {
        Expr pre = (Const)10 * 11 * 12;
        var rule = new Passes.Rules.Neutral.ReassociateMul();
        CompilerServices.ERewrite(pre, new[] { rule }, new(), CompileOptions);

        // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
    }

    [Fact]
    public void TestClassicDemo()
    {
        var g = new EGraph();
        Var x = "x";
        var e1 = g.Add(x * 2);
        _ = g.Add(x * 2 / 2);
        var e2 = g.Add(x << 1);
        g.Union(e2, e1);
        g.Rebuild();
    }

    [Fact]
    public void TestTransposeBinaryMotion()
    {
        var c0 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 3, 4 }));
        var c1 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 1, 1 }));
        Assert.Equal(c0.Arguments[1].GetHashCode(), c1.Arguments[1].GetHashCode());

        Expr pre = c0 + c1;

        Assert.True(pre.InferenceType());

        var post = CompilerServices.ERewrite(pre, new[] { new Passes.Rules.Neutral.CombineBinaryTranspose() }, new(), CompileOptions);

        Assert.True(post.InferenceType());
        Assert.Equal(pre.Evaluate(), post.Evaluate());
    }

    [Fact]
    [AutoSetupTestMethod(InitSession = true)]
    public void TestEgraphRemoveMarkerWithoutConstMarker()
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

        var post = CompilerServices.ERewrite(
            pre,
            new IRewriteRule[]
            {
                  new Passes.Rules.Lower.RemoveMarker(),
                  new TestMulToAdd(),
            },
            new(),
            CompileOptions);

        Assert.True(post.InferenceType());

        Assert.True(
          post is Call { Arguments: var param } &&
          param.Length == 2 &&
          param[1] is not Marker);
    }

    [Fact]
    public async Task TestEgraphRemoveMarkerPreserveCosts()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost;
#endif
        var v8 = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 64, 56, 56 }));
        var v9 = IR.F.NN.Conv2D(v8, Testing.Rand<float>(64, 64, 1, 1), Testing.Rand<float>(64), new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1, new[] { float.NegativeInfinity, float.PositiveInfinity });
        var v10 = IR.F.Math.RangeOfMarker(v9, new[] { -9.914007, 38.64287 }); // f32[1,56,56,64]
        var v11 = IR.F.NN.Conv2D(v10, Testing.Rand<float>(64, 64, 1, 1), Testing.Rand<float>(64), new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1, new[] { float.NegativeInfinity, float.PositiveInfinity });
        var v12 = IR.F.Math.RangeOfMarker(v11, new[] { -14.803145, 40.543793 }); // f32[1,64,56,56]
        var v13 = IR.F.NN.Relu(v12); // f32[1,64,56,56]
        var func = new Function("main", v13, new[] { v8 });
        var module = new IRModule(func);

        var passes = CompileSession.CreatePassManager("passes");
        passes.AddWithName<EGraphRulesPass>("Opt").Configure(p =>
        {
            p.Add<Passes.Rules.Lower.RemoveMarker>();
            p.Add<Passes.Rules.Neutral.ReluToClamp>();
            p.Add<Passes.Rules.Neutral.FuseClampConv2D>();
        });

        await passes.RunAsync(module);
        var post = (Function)module.Entry!;
        Assert.True(post.Body is Call { Target: IR.NN.Conv2D });
    }

    [Fact]
    public async Task TestTwoBranchQuantizeCSE()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost;
#endif
        var v8 = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 64, 56, 56 }));
        var v9 = IR.F.NN.Conv2D(v8, Testing.Rand<float>(64, 64, 1, 1), Testing.Rand<float>(64), new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1, new[] { float.NegativeInfinity, float.PositiveInfinity });
        var v10_1 = IR.F.Math.Quantize(v9, Const.FromTensor(Tensor.FromScalar<QuantParam>(new(10, 5.5f))), DataTypes.Int8);
        var v10_2 = IR.F.Math.Quantize(v9, Const.FromTensor(Tensor.FromScalar<QuantParam>(new(10, 5.5f))), DataTypes.Int8); // int8[1,56,56,64]
        var v10_11 = IR.F.Math.Dequantize(v10_1 + IR.F.Random.Normal(DataTypes.Int8, 0, 1, 3, new[] { 1, 64, 56, 56 }), Const.FromTensor(Tensor.FromScalar<QuantParam>(new(10, 5.5f))), DataTypes.Float32);
        v10_11 = v10_11 * IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, new[] { 1, 64, 56, 56 });
        var v10_22 = IR.F.Math.Dequantize(v10_2, Const.FromTensor(Tensor.FromScalar<QuantParam>(new(10, 5.5f))), DataTypes.Float32);
        var v11 = IR.F.NN.Conv2D(v10_11 + v10_22, Testing.Rand<float>(64, 64, 1, 1), Testing.Rand<float>(64), new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1, new[] { float.NegativeInfinity, float.PositiveInfinity });
        var v12 = IR.F.Math.RangeOfMarker(v11, new[] { -14.803145, 40.543793 }); // f32[1,64,56,56]
        var v13 = IR.F.NN.Relu(v12); // f32[1,64,56,56]
        var func = new Function("main", v13, new[] { v8 });
        Assert.True(func.InferenceType());
#if DEBUG
        CompilerServices.DumpDotIR(func, "pre", Dumpper.Directory);
#endif
        var module = new IRModule(func);
        var passes = CompileSession.CreatePassManager("passes");
        passes.AddWithName<EGraphRulesPass>("CSE").Configure(p =>
        {
        });

        await passes.RunAsync(module);
        var post = (Function)module.Entry!;
#if DEBUG
        CompilerServices.DumpDotIR(post, "post", Dumpper.Directory);
#endif
        var v = new TestVisitor();
        v.Visit(post);
        Assert.Equal(1, v.CountCallOp<IR.Math.Quantize>());
    }
}

public sealed class TestMulToAdd : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBinary(op => op.BinaryOp == BinaryOp.Mul, IsWildcard("lhs"), IsWildcard("rhs"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext options)
    {
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        return lhs + rhs;
    }
}
