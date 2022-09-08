﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Tensorflow.Operations.Initializers;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestFoldBinary: TestFixture.UnitTestFixtrue
{

    public static IEnumerable<object[]> TestFoldNopBinaryNegativeData =>
        new[]
        {
            new object[] {BinaryOp.Add, new[] {3}, 1f},
            new object[] {BinaryOp.Sub, new[] {3, 4}, 1f},
            new object[] {BinaryOp.Mul, new[] {3}, 2f},
            new object[] {BinaryOp.Div, new[] {3}, 2f},
            // new object[] { BinaryOp.Mod ,new[] { 3}, 2f},
            new object[] {BinaryOp.Pow, new[] {3},2f},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopBinaryNegativeData))]
    public void TestFoldNopBinaryNegative(BinaryOp binaryOp, int[] aShape, float bValue, int index)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());

        var rootPre = Math.Binary(binaryOp, Math.Binary(binaryOp, a, bValue), bValue);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldNopBinary(),
        }, caseOptions);
        // rootPre.InferenceType();
        Assert.Equal(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }

    public static IEnumerable<object[]> TestFoldNopBinaryPositiveData =>
        new[]
        {
            new object[] {BinaryOp.Add, new[] {3}, 0f},
            new object[] {BinaryOp.Sub, new[] {3, 4}, 0f},
            new object[] {BinaryOp.Mul, new[] {3}, 1f},
            new object[] {BinaryOp.Div, new[] {3}, 1f},
            // new object[] { BinaryOp.Mod ,new[] { 3}, 1f},
            new object[] {BinaryOp.Pow, new[] {3}, 1f},
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldNopBinaryPositiveData))]
    public void TestFoldNopBinaryPositive(BinaryOp binaryOp, int[] aShape, float bValue, int index)
    {
        var caseOptions = GetPassOptions();
        var a = new Var();
        var Normal = new Dictionary<Var, IValue>();
        Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, aShape).Evaluate());
        var rootPre = Math.Binary(binaryOp, Math.Binary(binaryOp, a, bValue), bValue);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldNopBinary(),
        }, caseOptions);
        // rootPre.InferenceType();
        Assert.NotEqual(rootPre, rootPost);
        Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    }
}