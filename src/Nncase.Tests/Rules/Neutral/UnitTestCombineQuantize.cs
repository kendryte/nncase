// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using static Nncase.IR.F.NN;
using ITuple = Nncase.IR.ITuple;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCombineQuantize : TestClassBase
{
    public static TheoryData<int[][], int, DataType, QuantParam> CombineQuantizeConcatPositiveData = new()
    {
      { new int [][]{ new int[]{ 1,32,160,160},new int[]{1,32,160,160} }, 1, DataTypes.UInt8, new(2,0.25185302f) },
      { new int [][]{ new int[]{ 1,64,80,80},new int[]{1,64,80,80} }, 1, DataTypes.UInt8, new(20,0.042551044f) }
    };

    [Theory]
    [MemberData(nameof(CombineQuantizeConcatPositiveData))]
    public async Task TestCombineQuantizeConcatPositive(int[][] inShapes, int axis, DataType destType, QuantParam quantParam)
    {
        var parameters = new List<Var>();
        var feedDict = new Dictionary<Var, IValue>();
        for (int i = 0; i < inShapes.Length; i++)
        {
            var v = new Var(i.ToString(), new TensorType(DataTypes.Float32, inShapes[i]));
            parameters.Add(v);
            feedDict.Add(v, IR.F.Random.Uniform(DataTypes.Float32, 1.0f, -1.0f, i, inShapes[i]).Evaluate());
        }
        var rootPre = new IR.Function(IR.F.Math.Quantize(Tensors.Concat(new IR.Tuple(parameters), axis), quantParam, destType), ImmutableArray.CreateRange(parameters));
        Assert.True(CompilerServices.InferenceType(rootPre));
        var pass = new DataflowWithUsdByPass();
        pass.Add<CombineQuantizeConcat>();
        var rootPost = (Function)await pass.RunAsync(rootPre, new());
        Assert.NotEqual(rootPre, rootPost);

        Assert.Equal(CompilerServices.Evaluate(rootPre, feedDict), CompilerServices.Evaluate(rootPost, feedDict));
    }

    [Fact]
    public async Task TestCombineQuantizeConcatNegative()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 256, 20, 20 }));// f32[1,256,20,20]
        var v251 = ReduceWindow2D(ReduceOp.Max, input, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var v253 = ReduceWindow2D(ReduceOp.Max, v251, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var v255 = ReduceWindow2D(ReduceOp.Max, v253, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var body = IR.F.Math.Quantize(IR.F.Tensors.Concat(new IR.Tuple(input, v251, v253, v255), 1), new QuantParam(1, 0.323f), DataTypes.UInt8);

        var feedDict = new Dictionary<Var, IValue>()
        {
          {input,IR.F.Random.Uniform(DataTypes.Float32, 1.0f, -1.0f, 0, new[]{1, 256, 20, 20}).Evaluate()}
        };
        var rootPre = new Function(body, input);
        Assert.True(CompilerServices.InferenceType(rootPre));

        var pass = new DataflowWithUsdByPass();
        pass.Add<CombineQuantizeConcat>();
        var rootPost = (Function)await pass.RunAsync(rootPre, new());
        Assert.Equal(rootPre, rootPost);
    }
}
