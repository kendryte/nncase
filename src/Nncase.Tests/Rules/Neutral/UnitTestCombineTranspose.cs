// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCombineTranspose : TransformTestBase
{
    public static readonly TheoryData<BinaryOp, int[], int[], int[], bool> CombineTransposeConstBinaryPositiveData = new()
    {
        // BinaryOp binaryOp, int[] lShape, int[] rShape, int[] perm, bool leftConst
        { BinaryOp.Add, new[] { 1, 32, 32, 64, }, new[] { 64 }, new[] { 0, 3, 1, 2 }, false },
        { BinaryOp.Add, new[] { 1, 32, 32, 64, }, Array.Empty<int>(), new[] { 0, 3, 1, 2 }, false },
        { BinaryOp.Sub, new[] { 1, 32, 32, 64, }, new[] { 32, 64 }, new[] { 0, 3, 1, 2 }, false },
        { BinaryOp.Mul, new[] { 1, 32, 32, 64, }, new[] { 1, 1, 1, 64 }, new[] { 0, 3, 1, 2 }, false },
        { BinaryOp.Div, new[] { 64 }, new[] { 1, 32, 32, 64, }, new[] { 0, 3, 1, 2 }, true },
        { BinaryOp.Div, Array.Empty<int>(), new[] { 1, 32, 32, 64, }, new[] { 0, 3, 1, 2 }, true },
        { BinaryOp.Sub, new[] { 32, 64 }, new[] { 1, 32, 32, 64, }, new[] { 0, 3, 1, 2 }, true },
        { BinaryOp.Mul, new[] { 1, 1, 1, 64 }, new[] { 1, 32, 32, 64, }, new[] { 0, 3, 1, 2 }, true },
    };

    public static IEnumerable<object[]> CombineBinaryTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 5, 4 }, new[] { 5, 4 }, new[] { 1, 0 } },
            new object[] { new[] { 4, 4 }, new[] { 4, 4 }, new[] { 1, 0 } },
            new object[] { new[] { 4 }, new[] { 4 }, new[] { 0 } },
            new object[] { new[] { 1, 3, 4 }, new[] { 1, 3, 4 }, new[] { 0, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 1, 3, 2, 4 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> CombineConstBinaryTransposeNotMatchData =>
        new[]
        {
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 2, 3 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 2, 4, 3 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> CombineRConstBinaryTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 3 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 3 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> CombineLConstBinaryTransposePositiveData =>
        new[]
        {
            new object[] { new[] { 3 }, new[] { 1, 3, 2, 4 }, new[] { 0, 3, 2, 1 } },
            new object[] { new[] { 3 }, new[] { 1, 3, 2, 4 }, new[] { 0, 2, 3, 1 } },
        };

    public static IEnumerable<object[]> TestCombineTransposeConcatPositiveData =>
        new[]
        {
            new object[] { new[] { 4, 4 }, new[] { 1, 0 }, 1, 2 },
            new object[] { new[] { 1, 3, 4 }, new[] { 0, 2, 1 }, 1, 6 },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { 0, 2, 3, 1 }, 2, 2 },
        };

    public static IEnumerable<object[]> TestCombineTransposeConcatNegativeData =>
        new[]
        {
            new object[] { new[] { 4, 4 }, new[] { new[] { 1, 0 }, new[] { 0, 1 } }, 1, 2, true },
            new object[] { new[] { 1, 3, 2, 4 }, new[] { new[] { 0, 2, 3, 1 }, new[] { 0, 2, 3, 1 } }, 2, 2, false },
        };

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
    [MemberData(nameof(TestCombineTransposeConcatPositiveData))]
    public void TestCombineTransposeConcatPositive(int[] inShape, int[] perm, int axis, int concatNum)
    {
        var inputList = new List<Var>();
        for (int i = 0; i < concatNum; i++)
        {
            inputList.Add(new Var());
        }

        var normal = new Dictionary<Var, IValue>();
        var tpList = new List<Call>();
        foreach (Var a in inputList)
        {
            // TODO:  Rely type infer and cse
            // Normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
            // tpList.Add(Tensors.Transpose(a, perm));
            var b = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
            tpList.Add(Tensors.Transpose(b, perm));
        }

        var input = Enumerable.Range(0, concatNum).Select(i => tpList[i]).ToArray();
        var rootPre = Tensors.Concat(new IR.Tuple(input), axis);
        TestMatched<CombineTransposeConcat>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeConcatNegativeData))]
    public void TestCombineTransposeConcatNegative(int[] inShape, int[][] perm, int axis, int concatNum, bool lastInputIsTp)
    {
        var inputList = new List<Call>();
        foreach (var i in Enumerable.Range(0, concatNum - 1))
        {
            var b = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
            inputList.Add(Tensors.Transpose(b, perm[i]));
        }

        if (lastInputIsTp)
        {
            var b = Random.Normal(DataTypes.Float32, 0, 1, 0, inShape);
            inputList.Add(Tensors.Transpose(b, perm[concatNum - 1]));
        }
        else
        {
            var b = Random.Normal(DataTypes.Float32, 0, 1, 0, perm[concatNum - 1].Select(p => inShape[p]).ToArray());
            inputList.Add(Math.Unary(UnaryOp.Neg, b));
        }

        var input = Enumerable.Range(0, concatNum).Select(i => inputList[i]).ToArray();
        var rootPre = Tensors.Concat(new IR.Tuple(input), axis);
        TestNotMatch<CombineTransposeConcat>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineTransposeConstBinaryPositiveData))]
    public void TestCombineTransposeConstBinaryPositive(BinaryOp binaryOp, int[] lShape, int[] rShape, int[] perm, bool leftConst)
    {
        Expr lhs = leftConst ?
          lShape.Length == 0 ? 0.5f : Const.FromValue(Random.Normal(DataTypes.Float32, 0, 1, 3, lShape).Evaluate()) :
          new Var("lhs", new TensorType(DataTypes.Float32, lShape));
        Expr rhs = leftConst ? new Var("b", new TensorType(DataTypes.Float32, rShape)) :
          rShape.Length == 0 ? 0.2f : Const.FromValue(Random.Normal(DataTypes.Float32, 0, 1, 4, rShape).Evaluate());

        var feedDict = new Dictionary<Var, IValue>();
        if (leftConst)
        {
            feedDict.Add((Var)rhs, Random.Normal(DataTypes.Float32, 0, 1, 1, rShape).Evaluate());
        }
        else
        {
            feedDict.Add((Var)lhs, Random.Normal(DataTypes.Float32, 0, 1, 2, lShape).Evaluate());
        }

        var rootPre = Tensors.Transpose(Math.Binary(BinaryOp.Add, lhs, rhs), perm);
        TestMatched<CombineTransposeConstBinary>(rootPre, feedDict);
    }

    [Theory]
    [MemberData(nameof(CombineBinaryTransposePositiveData))]
    public void TestCombineBinaryTransposePositive(int[] lShape, int[] rShape, int[] perm)
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
        TestMatched<CombineBinaryTranspose>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(CombineConstBinaryTransposeNotMatchData))]
    public void TestCombineConstTransposeNotMatch(int[] lShape, int[] rShape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, permExpr), b);
        TestNotMatch<CombineBinaryTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineRConstBinaryTransposePositiveData))]
    public void TestCombineTransposeRConstBinaryPositive(int[] lShape, int[] rShape, int[] perm)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Transpose(a, permExpr), b);
        TestMatched<CombineConstBinaryTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineLConstBinaryTransposePositiveData))]
    public void TestCombineLConstBinaryTransposePositive(int[] lShape, int[] rShape, int[] perm)
    {
        var a = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, lShape).Evaluate().AsTensor().ToArray<float>(), lShape);
        var b = Random.Normal(DataTypes.Float32, 0, 1, 0, rShape);

        Expr permExpr = perm;
        var rootPre = Math.Binary(BinaryOp.Add, a, Tensors.Transpose(b, permExpr));
        TestMatched<CombineConstBinaryTranspose>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposePadPositiveData))]
    public void TestCombineTransposePadPositive(int[] inShape, int[] perm, int[,] paddings, PadMode padM, float padValue)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Pad(Tensors.Transpose(a, perm), paddings, padM, padValue);
        TestMatchedCore(
            rootPre,
            normal,
            new IRewriteRule[]
            {
                new FoldConstCall(),
                new CombineTransposePad(),
            });
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposePadPositiveData))]
    public void TestCombinePadTransposePositive(int[] inShape, int[] perm, int[,] paddings, PadMode padM, float padValue)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Tensors.Transpose(Pad(a, paddings, padM, padValue), perm);
        TestMatchedCore(
            rootPre,
            normal,
            new IRewriteRule[]
            {
                new FoldConstCall(),
                new CombinePadTranspose(),
            });
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeReducePositiveData))]
    public void TestCombineTransposeReducePositive(int[] inShape, int[] perm, int axis, int initValue, bool keepDims)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Tensors.Reduce(ReduceOp.Mean, Tensors.Transpose(a, perm), axis, initValue, keepDims);
        TestMatched<CombineTransposeReduce>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCombineTransposeUnaryPositiveData))]
    public void TestCombineTransposeUnaryPositive(UnaryOp opType, int[] inShape, int[] perm)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Transpose(a, perm));
        TestMatched<CombineTransposeUnary>(rootPre, normal);
    }
}
