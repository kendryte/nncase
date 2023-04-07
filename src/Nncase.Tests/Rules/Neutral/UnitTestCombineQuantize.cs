// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using ITuple = Nncase.IR.ITuple;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCombineQuantize : TransformTestBase
{
    public static TheoryData<int[][], int, DataType, QuantParam> CombineQuantizeConcatPositiveData => new() { { new int[][] { new int[] { 1, 32, 160, 160 }, new int[] { 1, 32, 160, 160 } }, 1, DataTypes.UInt8, new(2, 0.25185302f) }, { new int[][] { new int[] { 1, 64, 80, 80 }, new int[] { 1, 64, 80, 80 } }, 1, DataTypes.UInt8, new(20, 0.042551044f) }, };

    public static TheoryData<int[][], DataType, QuantParam> CombineQuantizeReshapePositiveData => new() { { new int[][] { new int[] { 1, 32, 160, 160 }, new int[] { 1, 160, 32, 160 } }, DataTypes.UInt8, new(2, 0.25185302f) }, { new int[][] { new int[] { 1, 64, 80, 80 }, new int[] { 1, 64, 1, 6400 } }, DataTypes.UInt8, new(20, 0.042551044f) }, };

    public static TheoryData<int[][], DataType, QuantParam> CombineQuantizeTransposePositiveData => new()
    {
        { new int[][] { new int[] { 1, 32, 160, 160 }, new int[] { 0, 3, 1, 2 } }, DataTypes.UInt8, new(2, 0.25185302f) },
        { new int[][] { new int[] { 1, 64, 80, 80 }, new int[] { 3, 2, 1, 0 } }, DataTypes.UInt8, new(20, 0.042551044f) },
    };

    public IAnalyzerManager AnalyzerMananger => CompileSession.GetRequiredService<IAnalyzerManager>();

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

        var rootPre = new IR.Function(IR.F.Math.Quantize(Tensors.Concat(new IR.Tuple(parameters.ToArray()), axis), quantParam, destType), parameters.ToArray());
        TestMatched<CombineQuantizeConcat>(rootPre, feedDict);
    }

    [Fact]
    public async Task TestCombineQuantizeConcatNegative()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 256, 20, 20 })); // f32[1,256,20,20]
        var v251 = ReduceWindow2D(ReduceOp.Max, input, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var v253 = ReduceWindow2D(ReduceOp.Max, v251, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var v255 = ReduceWindow2D(ReduceOp.Max, v253, -3.4028235E+38, new[] { 5L, 5L }, new[] { 1L, 1L }, new[,] { { 2L, 2L }, { 2L, 2L } }, new[] { 1L, 1L }, false, false); // f32[1,256,20,20]
        var body = IR.F.Math.Quantize(IR.F.Tensors.Concat(new IR.Tuple(input, v251, v253, v255), 1), new QuantParam(1, 0.323f), DataTypes.UInt8);
        _ = new Dictionary<Var, IValue>() { { input, IR.F.Random.Uniform(DataTypes.Float32, 1.0f, -1.0f, 0, new[] { 1, 256, 20, 20 }).Evaluate() }, };
        var rootPre = new Function(body, input);
        TestNotMatch<CombineQuantizeConcat>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineQuantizeReshapePositiveData))]
    public async Task TestCombineQuantizeReshapePositive(int[][] shapes, DataType destType, QuantParam quantParam)
    {
        var parameters = new List<Var>();
        var feedDict = new Dictionary<Var, IValue>();
        var v = new Var("input", new TensorType(DataTypes.Float32, shapes[0]));
        parameters.Add(v);
        feedDict.Add(v, IR.F.Random.Uniform(DataTypes.Float32, 1.0f, -1.0f, 0, shapes[0]).Evaluate());

        var rootPre = new IR.Function(Math.Quantize(Tensors.Reshape(v, shapes[1]), quantParam, destType), parameters.ToArray());
        TestMatched<CombineQuantizeReshape>(rootPre, feedDict);
    }

    [Fact]
    public async Task TestCombineQuantizeReshapeNegative()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 256, 20, 20 })); // f32[1,256,20,20]
        var v = Tensors.Reshape(input, new[] { 1, 256, 20, 20 }); // f32[1,256,20,20]
        var body = Math.Add(IR.F.Math.Quantize(v, new QuantParam(1, 0.323f), DataTypes.UInt8), IR.F.Math.Quantize(v, new QuantParam(1, 0.323f), DataTypes.UInt8));
        var rootPre = new Function(body, input);
        TestNotMatch<CombineQuantizeReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineQuantizeTransposePositiveData))]
    public async Task TestCombineQuantizeTransposePositive(int[][] shape_and_perm, DataType destType, QuantParam quantParam)
    {
        var parameters = new List<Var>();
        var feedDict = new Dictionary<Var, IValue>();
        var v = new Var("input", new TensorType(DataTypes.Float32, shape_and_perm[0]));
        parameters.Add(v);
        feedDict.Add(v, Random.Uniform(DataTypes.Float32, 1.0f, -1.0f, 0, shape_and_perm[0]).Evaluate());

        var rootPre = new IR.Function(Math.Quantize(Tensors.Transpose(v, shape_and_perm[1]), quantParam, destType), parameters.ToArray());
        TestMatched<CombineQuantizeTranspose>(rootPre, feedDict);
    }

    [Fact]
    public async Task TestCombineQuantizeTransposeNegative()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 256, 20, 20 })); // f32[1,256,20,20]
        var v = Tensors.Transpose(input, new[] { 0, 3, 2, 1 }); // f32[1,256,20,20]
        var body = Math.Add(Math.Quantize(v, new QuantParam(1, 0.323f), DataTypes.UInt8), IR.F.Math.Quantize(v, new QuantParam(1, 0.323f), DataTypes.UInt8));
        var rootPre = new Function(body, input);
        TestNotMatch<CombineQuantizeTranspose>(rootPre);
    }
}
