// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.PatternMatch.Utility;
using Random = Nncase.IR.F.Random;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestAddMarker : TestClassBase
{
    [Fact]
    public void TestAddMarkerRelu()
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var rootPre = Relu(a);
        var rootPost = CompilerServices.Rewrite(rootPre.Clone(), new[] { new AddRangeOfAndMarker() }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    }

    [Fact]
    public void TestAddMarkerTargetConst()
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Relu(a);
        var pre = IR.F.Math.RangeOfMarker(new[] { 1, 2, 3, 4 }, b);
        CompilerServices.InferenceType(pre);
    }

    [Fact]
    public void TestAddMarkerAttrConst()
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 8, 8 });
        var b = Relu(a);
        var pre = IR.F.Math.RangeOfMarker(b, new[] { 1, 2, 3, 4 });
        CompilerServices.InferenceType(pre);
    }

    [Fact]
    public void TestAddMarkerAllConst()
    {
        var pre = IR.F.Math.RangeOfMarker(new[] { 4, 5, 6, 7 }, new[] { 1, 2, 3, 4 });
        CompilerServices.InferenceType(pre);
    }

    [Fact]
    public void TestAddMarkerWithTuple()
    {
        var a = new IR.Tuple((IR.Const)1 * (IR.Const)2);
        var b = new IR.Tuple(a, a, a, a);
        var c = new IR.Tuple(b, b, b, b);
        var d = new IR.Tuple(c, c, c, c);
        var e = new IR.Tuple(d, d, d, d);
        var pre = IR.F.Math.RangeOfMarker(new[] { 4, 5, 6, 7 }, e);
        CompilerServices.InferenceType(pre);
    }

    [Fact]
    public async Task TestAddMarkerWithLstm()
    {
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.Rewrite | DumpFlags.PassIR;
#endif
        var inputSize = 2;
        var hiddenSize = 1;
        var lSTM_direction = LSTMDirection.Forward;
        var numberDirections = lSTM_direction == LSTMDirection.Forward ? 1 : 2;
        var batchSize = 1;
        var seqLength = 3;
        var x = Random.Normal(DataTypes.Float32, new[] { seqLength, batchSize, inputSize });
        var initC = Random.Normal(DataTypes.Float32, new[] { numberDirections, batchSize, hiddenSize });
        var initH = Random.Normal(DataTypes.Float32, new[] { numberDirections, batchSize, hiddenSize });
        var b = DataGenerator.DefaultRandom(new[] { numberDirections, 8 * hiddenSize });
        var w = DataGenerator.DefaultRandom(new[] { numberDirections, 4 * hiddenSize, inputSize });
        var r = DataGenerator.DefaultRandom(new[] { numberDirections, 4 * hiddenSize, hiddenSize });
        var p = new float[numberDirections, 3 * hiddenSize];
        var lstm = IR.F.RNN.LSTM(LSTMDirection.Forward, LSTMLayout.Zero, new[] { "Sigmoid", "Tanh", "Tanh" }, x, w, r, b, new[] { seqLength }, initH, initC, p, 0, 0, float.NaN, hiddenSize, 0, 3);
        var main = new Function(lstm);

        var module = new IRModule(main);
        await TestAddMarkerPasses(module);
        Assert.True(((Function)module.Entry!).Body is Tuple t
                    && CompilerServices.TryMatchRoot(t, IsWrappedLSTM(PatternMatch.F.Tensors.IsLSTM("lstm", "lstmCall", _ => true), (x, _) => IsRangeOfMarker(x, IsWildcard())), out var result)
                    && result["lstmCall"] is Call call
                    && new[] { 0, 1, 2, 5, 6 }.All(i => call.Arguments[i] is Marker));
    }

    [Fact]
    public async Task TestAddMarkerWithLstmInitHEqualsInitC()
    {
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.Rewrite | DumpFlags.PassIR;
#endif
        var inputSize = 2;
        var hiddenSize = 2;
        var lSTM_direction = LSTMDirection.Forward;
        var numberDirections = lSTM_direction == LSTMDirection.Forward ? 1 : 2;
        var batchSize = 1;
        var seqLength = 5;
        var x = Random.Normal(DataTypes.Float32, new[] { seqLength, batchSize, inputSize });
        var initC = Random.Normal(DataTypes.Float32, new[] { numberDirections, batchSize, hiddenSize });
        var initH = initC;
        var b = DataGenerator.DefaultRandom(new[] { numberDirections, 8 * hiddenSize });
        var w = DataGenerator.DefaultRandom(new[] { numberDirections, 4 * hiddenSize, inputSize });
        var r = DataGenerator.DefaultRandom(new[] { numberDirections, 4 * hiddenSize, hiddenSize });
        var p = new float[numberDirections, 3 * hiddenSize];
        var lstm = IR.F.RNN.LSTM(LSTMDirection.Forward, LSTMLayout.Zero, new[] { "Sigmoid", "Tanh", "Tanh" }, x, w, r, b, new[] { seqLength }, initH, initC, p, 0, 0, float.NaN, hiddenSize, 0, 2);
        var main = new Function(lstm);

        var module = new IRModule(main);
        await TestAddMarkerPasses(module);
        Assert.True(((Function)module.Entry!).Body is Tuple t
                    && CompilerServices.TryMatchRoot(t, IsWrappedLSTM(PatternMatch.F.Tensors.IsLSTM("lstm", "lstmCall", _ => true), (x, _) => IsRangeOfMarker(x, IsWildcard())), out var result)
                    && result["lstmCall"] is Call call
                    && new[] { 0, 1, 2, 5, 6 }.All(i => call.Arguments[i] is Marker));
    }

    [Fact]
    public async Task TestAddMarkerOutput()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost;
#endif
        var v121 = new IR.Var("a", new IR.TensorType(DataTypes.Float32, new[] { 1, 2048, 7, 7 }));
        IR.Function main;
        {
            var v122 = IR.F.Tensors.Transpose(v121, new[] { 0, 2, 3, 1 }); // f32[1,7,7,2048]
            var v123 = IR.F.Math.Mul(v122, Testing.Rand<float>(2048)); // f32[1,7,7,2048]
            var v124 = IR.F.Math.Add(v123, Testing.Rand<float>(2048)); // f32[1,7,7,2048]
            var v125 = IR.F.NN.Relu(v124); // f32[1,7,7,2048]
            var v126 = new IR.Tuple(new IR.Expr[] { v125 }); // (f32[1,7,7,2048])
            main = new IR.Function(v126, new[] { v121 });
        }

        var module = new IR.IRModule(main);
        await TestAddMarkerPasses(module);
        Assert.True(((IR.Function)module.Entry!).Body is IR.Tuple tuple && tuple.Fields[0] is IR.Marker);
    }

    [Fact]
    public async Task TestAddMarkerOutputHasRangeOf()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost;
#endif
        var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, new[] { 1, 3, 8, 8 }));
        var main = new IR.Function(new IR.Tuple(Relu(IR.F.Math.RangeOfMarker(a, new[] { -1.0f, 1.0f }))), new[] { a });
        var module = new IR.IRModule(main);
        await TestAddMarkerPasses(module);
        Assert.True(((IR.Function)module.Entry!).Body is IR.Tuple tuple && tuple.Fields[0] is IR.Marker);
    }

    [Fact]
    public async Task TestAddMarkerOutputFalse()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.EGraphCost;
#endif
        var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, new[] { 1, 3, 8, 8 }));
        var main = new IR.Function(new IR.Tuple(IR.F.Math.Unary(UnaryOp.LogicalNot, IR.F.Math.RangeOfMarker(a, new[] { -1.0f, 1.0f }))), new[] { a });
        var module = new IR.IRModule(main);
        await TestAddMarkerPasses(module);
        Assert.True(((IR.Function)module.Entry!).Body is IR.Tuple tuple && tuple.Fields[0] is IR.Call);
    }

    private async Task TestAddMarkerPasses(IR.IRModule module)
    {
        var passManager = CompileSession.CreatePassManager("manager");
        passManager.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p =>
        {
            p.Add<Passes.Rules.Neutral.AddRangeOfAndMarker>();
        });
        await passManager.RunAsync(module);
    }
}
