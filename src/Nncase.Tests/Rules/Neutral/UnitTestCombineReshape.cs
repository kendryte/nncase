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
public class UnitTestCombineReshape : TransformTestBase
{
    public static readonly TheoryData<BinaryOp, long[], long[], long[], bool> CombineConstBinaryReshapePositiveData = new()
    {
        // BinaryOp binaryOp, long[] lShape, long[] rShape, long[] shape, bool leftConst
        { BinaryOp.Add, new long[] { 32, 1, 32, 64 }, new long[] { 64 }, new long[] { 1, 32, 32, 64 }, false },
        { BinaryOp.Sub, new long[] { 1, 32, 32, 64 }, new long[] { 1 }, new long[] { 1, 1024, 1, 64 }, false },
        { BinaryOp.Div, new long[] { 64 }, new long[] { 32, 1, 32, 64, }, new long[] { 1, 32, 32, 64 }, true },
        { BinaryOp.Mul, new long[] { 1 }, new long[] { 1, 32, 32, 64, }, new long[] { 1, 1024, 64, 1 }, true },
        { BinaryOp.Sub, new long[] { 1 }, new long[] { 1, 32, 32, 64, }, new long[] { 1, 1024, 64, 1 }, true },
    };

    public static readonly TheoryData<long[], long[], long[]> TestCombineReshapeTransposeNegativeData =
    new()
    {
        { new long[] { 1, 77, 1, 64 }, new long[] { 2, 1, 3, 0 }, new long[] { 77, 64, 1 } },
        { new long[] { 1, 77, 12, 64 }, new long[] { 1, 0, 2, 3 }, new long[] { 1, 77, 768 } },
    };

    public static IEnumerable<object[]> CombineBinaryReshapePositiveData =>
        new[]
        {
            new object[] { new long[] { 5, 4 }, new long[] { 5, 4 }, new long[] { 1, 20 } },
            new object[] { new long[] { 4, 4 }, new long[] { 4, 4 }, new long[] { 2, 8 } },
            new object[] { new long[] { 4 }, new long[] { 4 }, new long[] { 4 } },
            new object[] { new long[] { 1, 3, 4 }, new long[] { 1, 3, 4 }, new long[] { 1, 4, 3 } },
            new object[] { new long[] { 1, 3, 2, 4 }, new long[] { 1, 3, 2, 4 }, new long[] { 1, 1, 6, 4 } },
        };

    public static IEnumerable<object[]> CombineConstBinaryReshapeNegativeData =>
        new[]
        {
            new object[] { new long[] { 1, 32, 32, 64, }, new long[] { 32, 64 }, new long[] { 1, 16, 64, 64 } },
        };

    public static IEnumerable<object[]> CombineBinaryReshapeNegativeData =>
        new[]
        {
            new object[] { new long[] { 5, 4 }, new long[] { 4, 5 }, new long[] { 1, 20 } },
        };

    public static IEnumerable<object[]> TestCombineUnaryReshapePositiveData =>
        new[]
        {
            new object[] { UnaryOp.Exp, new long[] { 1, 3, 4 }, new long[] { 1, 4, 3 } },
            new object[] { UnaryOp.Sqrt, new long[] { 1, 3, 4 }, new long[] { 3, 4, 1 } },
            new object[] { UnaryOp.Log, new long[] { 1, 3, 4, 5 }, new long[] { 3, 1, 1, 20 } },
            new object[] { UnaryOp.Abs, new long[] { 1, 3, 4, 5 }, new long[] { 1, 12, 5, 1 } },
        };

    public static IEnumerable<object[]> TestCombineReshapePadPositiveData =>
        new[]
        {
            new object[] { new long[] { 1, 3, 4 }, new long[] { 1, 5, 8 }, new int[] { 0, 0, 1, 1, 2, 2 } },
            new object[] { new long[] { 1, 3, 4 }, new long[] { 1, 4, 5, 7 }, new int[] { 1, 2, 1, 1, 2, 1 } },
        };

    public static IEnumerable<object[]> TestCombineReshapePadNegativeData =>
        new[]
        {
            new object[] { new long[] { 1, 3, 4 }, new long[] { 5, 8 }, new int[] { 0, 0, 1, 1, 2, 2 } },
            new object[] { new long[] { 1, 3, 4 }, new long[] { 1, 4, 1, 35 }, new int[] { 1, 2, 1, 1, 2, 1 } },
        };

    public static TheoryData<(int Count, IR.Expr Act)> TestCombineActivationsReshapePositiveData => new()
    {
        (1, Relu(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 4 }), new int[] { 1, 1, 8 }))),
        (2, Celu(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), new int[] { 2, 4, 1 }), 0.6f)),
        (3, HardSigmoid(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), new int[] { 8, 1, 1 }), 0.6f, 0.3f)),
        (3, Erf(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), new int[] { 2, 2, 2 }))),
        (3, Gelu(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), new int[] { 4, 1, 2 }), 1f)),
    };

    public static TheoryData<(int Count, IR.Expr Act)> TestCombineActivationsReshapeNegativeData => new()
    {
        (1, Softplus(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 1, 2, 4 }), new int[] { 4, 2, 1 }))),
        (2, Softsign(Tensors.Reshape(Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 4, 2, 1 }), new int[] { 1, 1, 8 }))),
    };

    [Theory]
    [MemberData(nameof(CombineConstBinaryReshapePositiveData))]
    public void TestCombineConstBinaryReshapePositive(BinaryOp binaryOp, long[] lShape, long[] rShape, long[] shape, bool leftConst)
    {
        Expr lhs = leftConst ? lShape.Sum() == 1 ? 0.5f : Const.FromValue(Random.Normal(DataTypes.Float32, 0, 1, 3, lShape).Evaluate()) : new Var("lhs", new TensorType(DataTypes.Float32, lShape));
        Expr rhs = leftConst ? new Var("b", new TensorType(DataTypes.Float32, rShape)) :
            rShape.Sum() == 1 ? 0.2f : Const.FromValue(Random.Normal(DataTypes.Float32, 0, 1, 4, rShape).Evaluate());

        var feedDict = new Dictionary<IVar, IValue>();
        if (leftConst)
        {
            feedDict.Add((Var)rhs, Random.Normal(DataTypes.Float32, 0, 1, 1, rShape).Evaluate());
        }
        else
        {
            feedDict.Add((Var)lhs, Random.Normal(DataTypes.Float32, 0, 1, 2, lShape).Evaluate());
        }

        var rootPre = Math.Binary(binaryOp, leftConst ? lhs : Tensors.Reshape(lhs, shape), leftConst ? Tensors.Reshape(rhs, shape) : rhs);
        TestMatched<CombineConstBinaryReshape>(rootPre, feedDict);
    }

    [Theory]
    [MemberData(nameof(CombineBinaryReshapePositiveData))]
    public void TestCombineBinaryReshapePositive(long[] lShape, long[] rShape, long[] shape)
    {
        var a = new Var("a", new TensorType(DataTypes.Float32, lShape));
        var b = new Var("b", new TensorType(DataTypes.Float32, rShape));

        var normal = new Dictionary<IVar, IValue>() { { a, Random.Normal(DataTypes.Float32, 0, 1, 0, lShape).Evaluate() }, { b, Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate() }, };

        Shape s = shape;
        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Reshape(a, s), Tensors.Reshape(b, s));
        TestMatched<CombineBinaryReshape>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(CombineBinaryReshapeNegativeData))]
    public void TestCombineBinaryReshapeNegative(long[] lShape, long[] rShape, long[] shape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Reshape(a, shape), b);
        TestNotMatch<CombineBinaryReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(CombineConstBinaryReshapeNegativeData))]
    public void TestCombineConstBinaryReshapeNegative(long[] lShape, long[] rShape, long[] shape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, lShape);
        var b = Tensor.From<float>(Random.Normal(DataTypes.Float32, 0, 1, 0, rShape).Evaluate().AsTensor().ToArray<float>(), rShape);

        var rootPre = Math.Binary(BinaryOp.Add, Tensors.Reshape(a, shape), b);
        TestNotMatch<CombineBinaryReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestCombineUnaryReshapePositiveData))]
    public void TestCombineUnaryReshapePositive(UnaryOp opType, long[] inShape, long[] shape)
    {
        var a = new Var();
        var normal = new Dictionary<IVar, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Math.Unary(opType, Tensors.Reshape(a, shape));
        TestMatched<CombineUnaryReshape>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCombineActivationsReshapePositiveData))]
    public void TestCombineActivationsReshapePositive((int Count, IR.Expr Act) param)
    {
        TestMatched<CombineActivationsReshape>(param.Act);
    }

    [Theory]
    [MemberData(nameof(TestCombineActivationsReshapeNegativeData))]
    public void TestCombineActivationsReshapeNegative((int Count, IR.Expr Act) param)
    {
        TestNotMatch<CombineActivationsReshape>(param.Act);
    }

    [Theory]
    [MemberData(nameof(TestCombineReshapePadPositiveData))]
    public void TestCombineReshapePadPositive(long[] inShape, long[] shape, int[] pads)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var normal = new Dictionary<IVar, IValue>();
        normal.Add(a, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate());
        var rootPre = Tensors.Reshape(NN.Pad(a, Tensor.From(pads, [pads.Length / 2, 2]), PadMode.Constant, 0f), shape);
        TestMatched<CombineReshapePad>(rootPre, normal);
    }

    [Theory]
    [MemberData(nameof(TestCombineReshapePadNegativeData))]
    public void TestCombineReshapePadNegative(long[] inShape, long[] shape, int[] pads)
    {
        var a = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var rootPre = Tensors.Reshape(NN.Pad(a, Tensor.From(pads, [pads.Length / 2, 2]), PadMode.Constant, 0f), shape);
        TestNotMatch<CombineReshapePad>(rootPre);
    }

    [Theory]
    [ClassData(typeof(CombineReshapeTransposePostiveData))]
    public void TestCombineReshapeTransposePostive(long[] inShape, long[] perm, long[] newshape)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var feed_dict = new Dictionary<IVar, IValue>
        {
            { input, Random.Normal(DataTypes.Float32, 0, 1, 0, inShape).Evaluate() },
        };
        var rootPre = Tensors.Reshape(Tensors.Transpose(input, perm), newshape);
        TestMatched<CombineReshapeTranspose>(rootPre, feed_dict);
    }

    [Theory]
    [MemberData(nameof(TestCombineReshapeTransposeNegativeData))]
    public void TestCombineReshapeTransposeNegative(long[] inShape, long[] perm, long[] newshape)
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, inShape));
        var rootPre = Tensors.Reshape(Tensors.Transpose(input, perm), newshape);
        TestNotMatch<CombineReshapeTranspose>(rootPre);
    }

    private sealed class CombineReshapeTransposePostiveData : TheoryData<long[], long[], long[]>
    {
        public CombineReshapeTransposePostiveData()
        {
            var inshapes = new[] {
                new long[] { 1, 77, 12, 64 },
                new long[] { 77, 1, 12, 64 },
                new long[] { 77, 12, 1, 64 },
                new long[] { 77, 12, 64, 1 },
             };

            var perms = new long[] { 0, 1, 2, 3 }.Permutate().ToArray();

            foreach (var (inshape, perm) in new[] { inshapes, perms }.CartesianProduct().Select(i => i.ToArray()).Select(i => (i[0], i[1])))
            {
                var newshape = perm.Select(i => inshape[i]).ToList();
                newshape.Remove(1);
                Add(inshape, perm, newshape.ToArray());
            }
        }
    }
}
