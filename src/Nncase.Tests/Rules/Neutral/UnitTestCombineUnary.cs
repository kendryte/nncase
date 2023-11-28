// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
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
public class UnitTestCombineUnary : TransformTestBase
{
    // TODO: CombinePadUnary
    public static IEnumerable<object[]> TestCombinePadUnaryPositiveData =>
        new[]
        {
            new object[]
            {
                UnaryOp.Exp, new[] { 1, 3, 4, 5 },  new[,]
            {
                { 1, 1 },
                { 2, 2 },
                { 1, 1 },
                { 3, 3 },
            }, PadMode.Symmetric, 0f,
            },

            // new object[]
            // {
            //     UnaryOp.Abs, new[] { 1, 3, 4, 5 },  new[,]
            // {
            //     { 1, 1 },
            //     { -1, -1 },
            //     { 1, 1 },
            //     { 3, 3 },
            // }, PadMode.Reflect, 0f,
            // },
            new object[]
            {
                UnaryOp.Floor, new[] { 1, 3, 4, 5 },  new[,]
            {
                { 1, 1 },
                { 0, 0 },
                { 1, 1 },
                { 0, 0 },
            }, PadMode.Constant, 2f,
            },
            new object[]
            {
                UnaryOp.Floor, new[] { 1, 3, 4, 5 },  new[,]
            {
                { 1, 1 },
                { 0, 0 },
                { 1, 3 },
                { 6, 0 },
            }, PadMode.Edge, 2f,
            },
        };

    public static IEnumerable<object[]> TestCombineSliceUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] { 6 },  new[] { 0 }, new[] { 4 }, new[] { 0 }, new[] { 1 } },
            new object[] { UnaryOp.Abs, new[] { 4, 5 },  new[] { 0, 0 }, new[] { 2, 3 }, new[] { 0, 1 }, new[] { 1, 2 } },
            new object[] { UnaryOp.Sqrt, new[] { 3, 4, 5 },  new[] { 0, 0, 1 }, new[] { 2, 3, 4 }, new[] { 0, 1, 2 }, new[] { 1, 2, 3 } },
            new object[] { UnaryOp.Square, new[] { 3, 2, 4, 5 },  new[] { 0, 0, 1, 1 }, new[] { -1, 2, 3, 4 }, new[] { -4, 1, 2, 3 }, new[] { 1, 2, 3, 2 } },
        };

    // TODO: CombineReshapeUnary
    public static IEnumerable<object[]> TestCombineReshapeUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] { 6 }, new[] { 2, 3 } },
            new object[] { UnaryOp.Abs, new[] { 4, 5 }, new[] { 2, 2, 5 }, },
            new object[] { UnaryOp.Sqrt, new[] { 3, 4, 5 }, new[] { 3, 2, 2, 5 } },
            new object[] { UnaryOp.Square, new[] { 3, 2, 4, 5 }, new[] { 3, 4, 1, 10 } },
        };

    public static IEnumerable<object[]> TestCombineTranposeUnaryPositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new[] { 6 }, new[] { 0 } },
            new object[] { UnaryOp.Abs, new[] { 4, 5 }, new[] { 1, 0 }, },
            new object[] { UnaryOp.Sqrt, new[] { 3, 4, 5 }, new[] { 2, 0, 1 } },
            new object[] { UnaryOp.Square, new[] { 3, 2, 4, 5 }, new[] { 3, 1, 2, 0 } },
        };

    [Theory]
    [MemberData(nameof(TestCombinePadUnaryPositiveData))]
    public void TestCombinePadUnaryPositive(UnaryOp opType, int[] inShape, int[,] paddings, PadMode padM, float padValue)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Pad(a, paddings, padM, padValue));
        TestMatched<CombinePadUnary>(rootPre, normal);
    }

    [Fact(Skip = "Bug")]
    public void TestCombinePadAbs()
    {
        var a = new Var();
        var feeds = new Dictionary<Var, IValue>();
        feeds.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, new[] { 1, 3, 4, 5 }).Evaluate());
        var pre = IR.F.Math.Unary(UnaryOp.Abs, Pad(a, new int[,] { { 1, 1 }, { -1, -1 }, { 1, 1 }, { 3, 3 } }, PadMode.Reflect, 0.0f));
        var rules = new[] { new CombinePadUnary() };
        Assert.True(pre.InferenceType(), "TestInferFailed:" + pre.CheckedType);
        if (rules.Length == 0)
        {
            throw new InvalidOperationException("Rules should not be empty");
        }

        var preHashCode = pre.GetHashCode();
        var v1 = pre.Evaluate(feeds);
        var post = CompilerServices.Rewrite(pre, rules, new());
        Assert.NotEqual(preHashCode, post.GetHashCode());
        var v2 = post.Evaluate(feeds);
        if (!Comparator.AllEqual(v1, v2))
        {
            Comparator.Compare(v1, v2);
        }
    }

    [Theory]
    [MemberData(nameof(TestCombineSliceUnaryPositiveData))]
    public void TestCombineSliceUnaryPositive(UnaryOp opType, int[] inShape, int[] begins, int[] ends, int[] axes, int[] strides)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Slice(a, begins, ends, axes, strides));
        TestMatched<CombineSliceUnary>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCombineReshapeUnaryPositiveData))]
    public void TestCombineReshapeUnaryPositive(UnaryOp opType, int[] inShape, int[] outShape)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Reshape(a, outShape));
        TestMatched<CombineReshapeUnary>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCombineTranposeUnaryPositiveData))]
    public void TestCombineTranposeUnaryPositive(UnaryOp opType, int[] inShape, int[] perm)
    {
        var a = new Var();
        var normal = new Dictionary<Var, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = IR.F.Math.Unary(opType, Tensors.Transpose(a, perm));
        TestMatched<CombineTranposeUnary>(rootPre, normal);
    }
}
