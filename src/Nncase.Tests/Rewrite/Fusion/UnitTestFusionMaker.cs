// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Nncase.Transform.Mutators;
using Nncase.Transform.Rules.Neutral;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Transpose = Nncase.IR.Tensors.Transpose;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.ReWrite.FusionTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestFusionMaker : TestClassBase
{
    private readonly ITestOutputHelper _testOutputHelper;

    public UnitTestFusionMaker(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    [Fact]
    public async Task TestMultiFusion()
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
    public async Task TestMatchPairLayerFusion()
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
        var post2 = (Function)rewriter.Rewrite(
            post,
            new IMergeRewriteRule[]
            {
                new SameInputFusionMergeRule(),
                new MultiInputFusionMergeRule(),
                new ShortCutFusionMergeRuleLeft(),
                new ShortCutFusionMergeRuleRight(),
            },
            (usedby, rule, option) => new FusionGroupMutator(usedby, rule, option),
            new());

        var isMatch = CompilerServices.TryMatch(
            post2,
            IsPairLayerFusion<Unary, Transpose, Quantize, Dequantize>("StackVM", "unary"),
            out _);
        Assert.True(isMatch);
    }

    [Fact]
    public async Task TestMatchPairLayerFusionForSingleFusion()
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
    public async Task TestMakeDoubleInputFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
            var v3 = WrapperWith(
                x => x[0] + x[1],
                v1,
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 }));
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
    public async Task TestMakeDoubleInputWithConstFusion()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 24, 32, 3 }));
        Function pre;
        {
            var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
            var v3 = WrapperWith(
                x => x[0] + x[1],
                IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 24, 32 }).Evaluate().AsTensor(),
                v1);
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

    [Fact]
    public async Task TestComplexFusionSingleOutput()
    {
        var inShape = new[] { 1, 24, 32, 3 };
        var input = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
        var pre = new Function("main", v1, new Var[] { input });
        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestTransposeComplexFusion>();
        var post = (Function)await pass.RunAsync(pre, new());
        var newFusion = (Fusion)((Call)post.Body).Target;
        Assert.Single(newFusion.Parameters);
        var newVar = newFusion.Parameters[0];
        Assert.Equal("input_0", newVar.Name);
        Assert.Equal(input.TypeAnnotation, newVar.TypeAnnotation);
        var expectBody = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), newVar);
        Assert.Equal(newFusion.Body, expectBody);
    }

    [Fact]
    public async void TestComplexFusionTensorConstInput()
    {
        var inShape = new[] { 1, 24, 32, 3 };
        var input = DataGenerator.DefaultRandom(inShape);
        var v1 = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input); // f32[1,3,24,32]
        var pre = new Function("main", v1, new Var[]{});
        var pass = new DataflowPass { Name = "Fusion" };
        pass.Add<TestTransposeComplexFusion>();
        var post = (Function)await pass.RunAsync(pre, new());
        var newFusion = (Fusion)((Call)post.Body).Target;
        Assert.Empty(newFusion.Parameters);
        var expectBody = WrapperWith(x => Transpose(x[0], new[] { 0, 3, 1, 2 }), input);
        Assert.Equal(newFusion.Body, expectBody);
    }

    [Fact]
    public async Task TestComplexFusionMultiOutput()
    {
        // used for find which expr is not expected
        void Compare(Tuple oldBody, Tuple expectBody, int i)
        {
            var oldDeq = oldBody[0];
            var oldGetItem = ((Call)oldDeq).Parameters[0];
            var oldLSTM = ((Call)oldGetItem).Parameters[0];
            var oldQuant = ((Call)oldLSTM).Parameters[i];
            var oldVar = ((Call)oldQuant).Parameters[0];

            var expectDeq = ((IR.Tuple)expectBody).Fields[0];
            var expectGetItem = ((Call)expectDeq).Parameters[0];
            var expectLSTM = ((Call)expectGetItem).Parameters[0];
            var expectQuant = ((Call)expectLSTM).Parameters[i];
            var expectVar = ((Call)expectQuant).Parameters[0];
            Assert.Equal(oldVar, expectVar);
            Assert.Equal(oldQuant, expectQuant);
            Assert.Equal(oldLSTM, expectLSTM);

            Assert.Equal(oldGetItem, expectGetItem);
            Assert.Equal(oldDeq, expectDeq);
        }

        Call WrapInput(Var var)
        {
            return Quantize(var, default(QuantParam), DataTypes.Int8);
        }

        IR.Tuple WrapOutput(Call call)
        {
            var outputs = Enumerable.Range(0, 2).Select(i => GetItem(call, i)).ToArray();
            var exprs = outputs.Select(f => (Expr)Dequantize(f, default(QuantParam), DataTypes.Float32)).ToArray();
            return new IR.Tuple(exprs);
        }

        var inputSize = 2;
        var hiddenSize = 3;
        var numberOfGates = 4;
        var outputSize = 2;
        var x = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, 2 }));
        var init_c = new Var(new TensorType(DataTypes.Float32, new[] { 1, numberOfGates * hiddenSize, inputSize }));
        // var init_h = new Var(new TensorType(DataTypes.Float32, new[] { 1, numberOfGates * hiddenSize, hiddenSize }));
        var init_h = new Var(new TensorType(DataTypes.Float32, new[] { 1, numberOfGates * hiddenSize, hiddenSize }));
        var b = DataGenerator.DefaultRandom(new[] { 1, 1, 1, 1 });
        var w = DataGenerator.DefaultRandom(new[] { 1, 1, 1, 1 });
        var r = DataGenerator.DefaultRandom(new[] { 1, 1, 1, 1 });
        var dt = DataTypes.Int8;
        var lstm = IR.F.RNN.LSTM(LSTMDirection.Bidirectional, LSTMLayout.One, new[] { "act" },
            WrapInput(x),
            w,
            r,
            b,
            0, WrapInput(init_h),
            WrapInput(init_c),
            0, 0, 0, 0, hiddenSize, 0, outputSize);

        var oldBody = WrapOutput(lstm);
        Assert.True(oldBody.InferenceType());
        var f = new Function("main", oldBody, new[] { x, init_c, init_h });
        var pass = new DataflowPass { Name = "TestComplexFusion" };
        pass.Add<LSTMFusion>();
        var afterCall = (Call)((Function)await pass.RunAsync(f, new())).Body;

        var newVars = ((Fusion)afterCall.Target).Parameters;
        var newVarNames = newVars.Select(v => v.Name).ToArray();

        // check var name
        Assert.Equal(newVarNames, new[] { "input_0", "input_1", "input_2" });

        // construct a expect Tuple
        // avoiding the error of comparing var, because comparing var is by ref
        var newVar0 = x;
        var newVar1 = init_c;
        var newVar2 = init_h;
        var expectLSTM = ReplaceUtility.ReplaceParams(lstm, (LSTM.X, WrapInput(newVar0)), (LSTM.InitialC, WrapInput(newVar1)), (LSTM.InitialH, WrapInput(newVar2)));
        var expectBody = WrapOutput(expectLSTM);
        var expectCall = new Call(new Fusion("FusionMaker_0", "StackVM", expectBody, new[] { newVar0, newVar1, newVar2 }), x, w, r);
        expectCall.InferenceType();
        Assert.True(CompilerServices.TryMatch(afterCall.Target, new LSTMFusion().Pattern, out _));
        var idxList = LSTMFusion.FusionCondMaker.InputsPattern.Select(x => x.Item1.Index).ToArray();
        foreach (int idx in idxList)
        {
            Compare(oldBody, expectBody, idx);
        }
    }

    private Expr WrapperWith(Func<Expr[], Expr> ctor, params Expr[] inputs)
    {
        var new_inputs = inputs.Select(i => Quantize(i, new QuantParam(0, 1), DataTypes.BFloat16)).ToArray();
        var output = ctor(new_inputs);
        return Dequantize(output, new QuantParam(0, 1), DataTypes.Float32);
    }

    internal sealed class TestTransposeComplexFusionSingleOutput : ComplexFusion<Transpose, Quantize, Dequantize>
    {
        public override (ParameterInfo, CallPattern)[] InputPatterns { get; } = GenerateInputPatterns(Transpose.Input);
    }

    internal sealed class LSTMFusion : ComplexFusion<LSTM, Quantize, Dequantize>
    {
        public override (ParameterInfo, CallPattern)[] InputPatterns { get; } = GenerateInputPatterns(LSTM.X, LSTM.InitialC, LSTM.InitialH);
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

internal sealed class
    TestTransposeFusion : SingleInputFusion<IR.Tensors.Transpose, IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "TransposeFusion";
}

internal sealed class TestBinaryFusion : DoubleInputFusion<IR.Math.Binary, IR.Math.Quantize, IR.Math.Dequantize>
{
    public override string Name { get; } = "BinaryFusion";
}
