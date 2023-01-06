// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
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
public class UnitTestCombineTranspose : TestClassBase
{
    public static IEnumerable<object[]> TestCombineTransposeBinaryPositiveData =>
        new[]
        {
            new object[] { new[] { 5, 4 }, new[] { 5, 4 }, new[] { 1, 0 } },
            new object[] { new[] { 4, 4 }, new[] { 4, 4 }, new[] { 1, 0 } },
            new object[] { new[] { 4 }, new[] { 4 }, new[] { 0 } },
            new object[] { new[] { 1, 3, 4 }, new[] { 1, 3, 4 }, new[] { 0, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 1, 3, 2, 4 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> TestCombineTransposeConstBinaryNotMatchData =>
        new[]
        {
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 2, 3 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 2, 4, 3 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> TestCombineTransposeRConstBinaryPositiveData =>
        new[]
        {
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 3 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 3 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> TestCombineTransposeLConstBinaryPositiveData =>
        new[]
        {
            new object[] { new[] { 3 }, new[] { 1, 3, 2, 4 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 3 }, new[] { 1, 3, 2, 4 }, new[] { 0, 2, 3, 1 } },
        };

    // public static IEnumerable<object[]> TestCombineTransposeConcatPositiveData =>
    //     new[]
    //     {
    //         // new object[] {new[] {4, 4}, new[] {1, 0}, 1, 2},
    //         new object[] {new[] {1, 3, 4}, new[] {0, 2, 1}, 1, 6},
    //         // new object[] {new[] {1, 3, 2, 4}, new[] {0, 2, 3, 1}, 2, 1},
    //     };
    //
    // [Theory]
    // [MemberData(nameof(TestCombineTransposeConcatPositiveData))]
    // public void TestCombineTransposeConcatPositive(int[] inShape, int[] perm, int axis, int concatNum)
    // {
    //     var caseOptions = GetPassOptions();
    //     var inputList = new List<Var>();
    //     for (int i = 0; i < concatNum; i++)
    //     {
    //         inputList.Add(new Var());
    //     }
    //     var Normal = new Dictionary<Var, IValue>();
    //     var tpList = new List<Call>();
    //     foreach (Var a in inputList)
    //     {
    //         // TODO:  Rely type infer and cse
    //         // Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
    //         // tpList.Add(Tensors.Transpose(a, perm));
    //         var b = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
    //         tpList.Add(Tensors.Transpose(b, perm));
    //     }
    //
    //     var input = Enumerable.Range(0, concatNum).Select(i => tpList[i]);
    //     var rootPre = Tensors.Concat(new IR.Tuple(input), axis);
    //     var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
    //     {
    //         new CombineTransposeConcat(),
    //         // Should not open constant fold.
    //         // new FoldConstCall(),
    //     }, caseOptions);
    //
    //     Assert.NotEqual(rootPre, rootPost);
    //     Assert.Equal(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost));
    //     // Assert.Equal(CompilerServices.Evaluate(rootPre, Normal), CompilerServices.Evaluate(rootPost, Normal));
    // }
    public static IEnumerable<object[]> TestCombineTransposePadPositiveData =>
        new[]
        {
            new object[]
            {
                new[] { 1, 3, 1, 2 }, new[] { 0, 3, 1, 2 },
                new[,]
                {
                    { 0, 0 },
                    { 0, 0 },
                    { 2, 2 },
                    { 1, 1 },
                }, PadMode.Constant, 1.2f,
            },
            new object[]
            {
                new[] { 1, 2, 3, 4 }, new[] { 0, 2, 3, 1 },
                new[,]
                {
                    { 4, 4 },
                    { 3, 3 },
                    { 2, 2 },
                    { 1, 1 },
                }, PadMode.Constant, 1f,
            },
            new object[]
            {
                new[] { 1, 2, 3, 4 }, new[] { 0, 3, 1, 2 },
                new[,]
                {
                    { 1, 1 },
                    { 0, 0 },
                    { 1, 1 },
                    { 1, 1 },
                }, PadMode.Symmetric, 0f,
            },
            new object[]
            {
                new[] { 5, 2, 3, 4 }, new[] { 3, 0, 1, 2 },
                new[,]
                {
                    { 2, 2 },
                    { 0, 0 },
                    { 1, 1 },
                    { 1, 1 },
                }, PadMode.Reflect, 0f,
            },
            new object[]
            {
                new[] { 1, 2, 3, 4 }, new[] { 0, 3, 1, 2 },
                new[,]
                {
                    { 1, 1 },
                    { 0, 0 },
                    { -1, -1 },
                    { 1, 1 },
                }, PadMode.Edge, 0f,
            },
        };

    public static IEnumerable<object[]> TestCombineTransposeReducePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 3, 4 }, new[] { 0, 2, 1 }, 1, 0, false },
            new object[] { new[] { 1, 3, 4, 5 }, new[] { 0, 2, 3, 1 }, 2, 1, true },
        };

    public static IEnumerable<object[]> TestCombineTransposeUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] { 1, 3, 4 }, new[] { 0, 2, 1 } },
            new object[] { UnaryOp.Sqrt, new[] { 1, 3, 4 }, new[] { 0, 2, 1 } },
            new object[] { UnaryOp.Log, new[] { 1, 3, 4, 5 }, new[] { 0, 2, 3, 1 } },
            new object[] { UnaryOp.Abs, new[] { 1, 3, 4, 5 }, new[] { 0, 2, 3, 1 } },
        };

    [Theory]
    [MemberData(nameof(TestCombineTransposeBinaryPositiveData))]
    public void TestCombineTransposeBinaryPositive(int[] lShape, int[] rShape, int[] perm)
    {
        var a = new Var("a", new TensorType(DataTypes.Float32, lShape));
        var b = new Var("b", new TensorType(DataTypes.Float32, rShape));

        var normal = new Dictionary<Var, IValue>()
        {
         { a, Random.Normal(DataTypes.Float32, 0, 1, 0, lShape).Evaluate() },
         { b, Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate() },
        };
        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, permExpr), Tensors.Transpose(b, permExpr));
        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeBinary(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(Comparator.AllEqual(CompilerServices.Evaluate(rootPre, normal), CompilerServices.Evaluate(rootPost, normal)));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeConstBinaryNotMatchData))]
    public void TestCombineTransposeConstNotMatch(int[] lShape, int[] rShape, int[] perm)
    {
        var caseOptions = GetPassOptions();
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, permExpr), b);
        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new[] { new CombineTransposeConstBinary() }, caseOptions);

        Assert.Equal(rootPre, rootPost);
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeRConstBinaryPositiveData))]
    public void TestCombineTransposeRConstBinaryPositive(int[] lShape, int[] rShape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, permExpr), b);
        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeConstBinary(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(Comparator.AllEqual(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost)));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeLConstBinaryPositiveData))]
    public void TestCombineTransposeLConstBinaryPositive(int[] lShape, int[] rShape, int[] perm)
    {
        var a = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, lShape).Evaluate().AsTensor().ToArray<float>(), lShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, a, Tensors.Transpose(b, permExpr));
        CompilerServices.InferenceType(rootPre);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeConstBinary(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(Comparator.AllEqual(CompilerServices.Evaluate(rootPre), CompilerServices.Evaluate(rootPost)));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposePadPositiveData))]
    public void TestCombineTransposePadPositive(int[] inShape, int[] perm, int[,] paddings, PadMode padM, float padValue)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Pad(Tensors.Transpose(a, perm), paddings, padM, padValue);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldConstCall(),
            new CombineTransposePad(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        var vpre = rootPre.Evaluate(normal);
        var vpost = rootPost.Evaluate(normal);
        Assert.True(Comparator.AllEqual(vpre, vpost));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposePadPositiveData))]
    public void TestCombinePadTransposePositive(int[] inShape, int[] perm, int[,] paddings, PadMode padM, float padValue)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Tensors.Transpose(Pad(a, paddings, padM, padValue), perm);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new FoldConstCall(),
            new CombinePadTranspose(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        var vpre = rootPre.Evaluate(normal);
        var vpost = rootPost.Evaluate(normal);
        Assert.True(Comparator.AllEqual(vpre, vpost));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeReducePositiveData))]
    public void TestCombineTransposeReducePositive(int[] inShape, int[] perm, int axis, int initValue, bool keepDims)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Tensors.Reduce(ReduceOp.Mean, Tensors.Transpose(a, perm), axis, initValue, keepDims);
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeReduce(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(Comparator.AllEqual(CompilerServices.Evaluate(rootPre, normal), CompilerServices.Evaluate(rootPost, normal)));
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeUnaryPositiveData))]
    public void TestCombineTransposeUnaryPositive(UnaryOp opType, int[] inShape, int[] perm)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Transpose(a, perm));
        var rootPost = CompilerServices.Rewrite(rootPre, new IRewriteRule[]
        {
            new CombineTransposeUnary(),
        }, new());

        Assert.NotEqual(rootPre, rootPost);
        Assert.True(Comparator.AllEqual(CompilerServices.Evaluate(rootPre, normal), CompilerServices.Evaluate(rootPost, normal)));
    }
}
