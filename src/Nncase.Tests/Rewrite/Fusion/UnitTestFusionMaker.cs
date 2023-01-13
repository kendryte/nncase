// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusionMaker : TestClassBase
{
    [Fact]
    public async void TestMultiFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v0 = Quantize(input, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,24,32,3]
            var v1 = Transpose(v0, new[] { 0, 3, 1, 2 }); // bf16[1,3,24,32]
            var v2 = Dequantize(v1, new QuantParam(0, 1), DataTypes.Float32); // f32[1,3,24,32]
            var v3 = v2 + IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 });
            var v4 = Quantize(v3, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,3,24,32]
            var v5 = Unary(UnaryOp.Abs, v4); // bf16[1,3,24,32]
            var v6 = Dequantize(v5, new QuantParam(0, 1), DataTypes.Float32); // f32[1,3,24,32]
            var v7 = v6 - IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 });
            var v8 = Quantize(v7, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,3,24,32]
            var v9 = Transpose(v8, new[] { 0, 2, 3, 1 }); // bf16[1,24,32,3]
            var v10 = Dequantize(v9, new QuantParam(0, 1), DataTypes.Float32); // f32[1,24,32,3]
            pre = new Function("main", v10, new Var[] { input });
        }

        CompilerServices.InferenceType(pre);

        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestUnaryFusion>();
        pass.Add<TestTransposeFusion>();

        var post = await pass.RunAsync(pre, new());
        var isMatch = CompilerServices.TryMatch(post, IsPairLayerFusion<Unary, Transpose, Quantize, Dequantize>("StackVM", "unary"), out _);
        Assert.True(isMatch);
    }

    [Fact]
    public async void TestMatchPairLayerFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v4 = Quantize(input, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,3,24,32]
            var v5 = Unary(UnaryOp.Abs, v4); // bf16[1,3,24,32]
            var v6 = Dequantize(v5, new QuantParam(0, 1), DataTypes.Float32); // f32[1,3,24,32]
            var v8 = Quantize(v6, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,3,24,32]
            var v9 = Transpose(v8, new[] { 0, 2, 3, 1 }); // bf16[1,24,32,3]
            var v10 = Dequantize(v9, new QuantParam(0, 1), DataTypes.Float32); // f32[1,24,32,3]
            pre = new Function("main", v10, new Var[] { input });
        }

        CompilerServices.InferenceType(pre);

        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestUnaryFusion>();
        pass.Add<TestTransposeFusion>();

        var post = await pass.RunAsync(pre, new());

        var rewriter = new DataFlowMergeRewriter();
        var post2 = (Function)rewriter.Rewrite(post, new IMergeRewriteRule[]
        {
            new SameInputFusionMergeRule(),
            new MultiInputFusionMergeRule(),
            new ShortCutFusionMergeRule(),
        }, (usedby, rule, option) => new FusionGroupMutator(usedby, rule, option),
          new());

        var isMatch = CompilerServices.TryMatch(post2, IsPairLayerFusion<Unary, Transpose, Quantize, Dequantize>("StackVM", "unary"), out _);
        Assert.True(isMatch);
    }

    [Fact]
    public async void TestMatchPairLayerFusionForSingleFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v4 = Quantize(input, new QuantParam(0, 1), DataTypes.BFloat16); // bf16[1,3,24,32]
            var v5 = Unary(UnaryOp.Abs, v4); // bf16[1,3,24,32]
            var v6 = Dequantize(v5, new QuantParam(0, 1), DataTypes.Float32); // f32[1,3,24,32]
            pre = new Function("main", v6, new Var[] { input });
        }

        CompilerServices.InferenceType(pre);

        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestUnaryFusion>();

        var post = await pass.RunAsync(pre, new());
        var isMatch = CompilerServices.TryMatch(post, IsPairLayerFusion<Unary, Transpose, Quantize, Dequantize>("StackVM", "unary"), out _);
        Assert.True(isMatch);
    }

    [Fact]
    public async void TestMakeDoubleInputFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
            var v3 = WrapperWith(x => x[0] + x[1], v1, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 }));
            var v5 = WrapperWith(x => Unary(UnaryOp.Abs, x[0]), v3); // f32[1,3,24,32]
            var v9 = WrapperWith(x => Transpose(x[0], new[] { 0, 2, 3, 1 }), v5); // f32[1,24,32,3]
            pre = new Function("main", v9, new Var[] { input });
        }

        CompilerServices.InferenceType(pre);

        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestUnaryFusion>();
        pass.Add<TestTransposeFusion>();
        pass.Add<TestBinaryFusion>();

        var post = (Function)await pass.RunAsync(pre, new());
        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(4, visitor.Count);
    }

    [Fact]
    public async void TestMakeDoubleInputWithConstFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
            var v3 = WrapperWith(x => x[0] + x[1], IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 }).Evaluate().AsTensor(), v1);
            var v5 = WrapperWith(x => Unary(UnaryOp.Abs, x[0]), v3); // f32[1,3,24,32]
            var v6 = WrapperWith(x => x[0], v5);
            var v9 = WrapperWith(x => Transpose(x[0], new[] { 0, 2, 3, 1 }), v6); // f32[1,24,32,3]
            pre = new Function("main", v9, new Var[] { input });
        }

        CompilerServices.InferenceType(pre);

        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestUnaryFusion>();
        pass.Add<TestTransposeFusion>();
        pass.Add<TestBinaryFusion>();
        pass.Add<TestDataTransFusion>();

        var post = (Function)await pass.RunAsync(pre, new());
        var visitor = new FusionCounterVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(5, visitor.Count);
    }

    private Expr WrapperWith(Func<Expr[], Expr> ctor, params Expr[] inputs)
    {
        var new_inputs = inputs.Select(i => Quantize(i, new QuantParam(0, 1), DataTypes.BFloat16)).ToArray();
        var output = ctor(new_inputs);
        return Dequantize(output, new QuantParam(0, 1), DataTypes.Float32);
    }
}

internal sealed class TestDataTransFusion : DataTransferFusion<IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "DataTransFusion";
}

internal sealed class TestUnaryFusion : SingleInputFusion<IR.Math.Unary, IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "UnaryFusion";
}

internal sealed class TestTransposeFusion : SingleInputFusion<IR.Tensors.Transpose, IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "TransposeFusion";
}

internal sealed class TestBinaryFusion : DoubleInputFusion<IR.Math.Binary, IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "BinaryFusion";
}
