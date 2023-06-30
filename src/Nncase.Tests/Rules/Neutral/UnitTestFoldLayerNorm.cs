// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Random;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldLayerNorm : TransformTestBase
{
    public static TheoryData<int[]> FoldLayerNormData => new()
    {
        new[] { 1, 3, 16 },
        new[] { 1, 2, 4 },
        new[] { 1, 1, 5 },
    };

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormPositive1(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Tensors.Reshape(v0, shape);
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v1, axes, initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v1, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Pow, v3, 1f);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, axes, initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Sqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Div, v3, v7);
            var v9 = IR.F.Tensors.Reshape(v8, shape);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v9, 1f);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, 1f);
            rootPre = v11;
        }

        TestMatched<FoldLayerNormPattern1>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormNegative1(int[] shape)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v1 = IR.F.Tensors.Reshape(v0, shape);
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v1, axes, initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v0, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Pow, v3, 2f);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, axes, initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Sqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Div, v3, v7);
            var v9 = IR.F.Tensors.Reshape(v8, shape);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v9, new[] { -0.1f });
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, new[] { 0.5f });
            rootPre = v11;
        }

        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
                new FoldLayerNormPattern1(),
                new FoldConstCall(),
            });
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormPositive2(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v0, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Pow, v3, 2f);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Sqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Div, v3, v7);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v8, -0.1f);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, 0.5f);
            rootPre = v11;
        }

        TestMatched<FoldLayerNormPattern2>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormNegative2(int[] shape)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v0, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Pow, v3, 2f);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Sqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Div, v4, v7);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v8, -0.1f);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, 0.5f);
            rootPre = v11;
        }

        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
                new FoldLayerNormPattern2(),
                new FoldConstCall(),
            });
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormPositive3(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v3 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            var v4 = IR.F.Math.Binary(BinaryOp.Sub, v0, v3);
            var v5 = IR.F.Math.Unary(UnaryOp.Square, v4);
            var v6 = IR.F.Tensors.Reduce(ReduceOp.Mean, v5, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v7 = IR.F.Math.Binary(BinaryOp.Add, v6, 1e-05f);
            var v8 = IR.F.Math.Unary(UnaryOp.Rsqrt, v7);
            var v9 = IR.F.Math.Binary(BinaryOp.Mul, v8, 0.5f);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v0, v9);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v3, v9);
            var v1 = IR.F.Math.Binary(BinaryOp.Sub, 0.5f, v2);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, v1);
            rootPre = v11;
        }

        TestMatched<FoldLayerNormPattern3>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormNegative3(int[] shape)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v3 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            var v4 = IR.F.Math.Binary(BinaryOp.Sub, v0, v3);
            var v5 = IR.F.Math.Unary(UnaryOp.Sqrt, v4);
            var v6 = IR.F.Tensors.Reduce(ReduceOp.Mean, v5, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v7 = IR.F.Math.Binary(BinaryOp.Add, v6, 1e-05f);
            var v8 = IR.F.Math.Unary(UnaryOp.Rsqrt, v7);
            var v9 = IR.F.Math.Binary(BinaryOp.Mul, v8, 0.5f);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v0, v9);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v3, 1e-05f);
            var v1 = IR.F.Math.Binary(BinaryOp.Sub, 0.5f, v2);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, v1);
            rootPre = v11;
        }

        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
                new FoldLayerNormPattern3(),
                new FoldConstCall(),
            });
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormPositive4(int[] shape)
    {
        // note shape is nchw
        var input = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        long[] axes = { -1 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v1 = input;
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v1, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v1, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v3, v3);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Rsqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Mul, v7, 0.05f);
            var v9 = IR.F.Math.Binary(BinaryOp.Mul, v1, v8);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v2, v8);
            var v11 = IR.F.Math.Binary(BinaryOp.Sub, 0.5f, v10);
            var v12 = IR.F.Math.Binary(BinaryOp.Add, v9, v11);
            rootPre = v12;
        }

        TestMatched<FoldLayerNormPattern4>(rootPre);

        var input1 = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape);
        var gamma = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape[^1]);
        var beta = IR.F.Random.Normal(DataTypes.Float32, 0, 1, 4, shape[^1]);
        {
            var v1 = input1;
            var v2 = IR.F.Tensors.Reduce(ReduceOp.Mean, v1, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v3 = IR.F.Math.Binary(BinaryOp.Sub, v1, v2);
            var v4 = IR.F.Math.Binary(BinaryOp.Mul, v3, v3);
            var v5 = IR.F.Tensors.Reduce(ReduceOp.Mean, v4, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v6 = IR.F.Math.Binary(BinaryOp.Add, v5, 1e-05f);
            var v7 = IR.F.Math.Unary(UnaryOp.Rsqrt, v6);
            var v8 = IR.F.Math.Binary(BinaryOp.Mul, v7, gamma.Evaluate().AsTensor());
            var v9 = IR.F.Math.Binary(BinaryOp.Mul, v1, v8);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v2, v8);
            var v11 = IR.F.Math.Binary(BinaryOp.Sub, beta.Evaluate().AsTensor(), v10);
            var v12 = IR.F.Math.Binary(BinaryOp.Add, v9, v11);
            rootPre = v12;
        }

        TestMatched<FoldLayerNormPattern4>(rootPre);
    }

    [Theory]
    [MemberData(nameof(FoldLayerNormData))]
    public void TestFoldLayerNormNegative4(int[] shape)
    {
        // note shape is nchw
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        Expr rootPre;
        {
            var v0 = input;
            var v3 = IR.F.Tensors.Reduce(ReduceOp.Mean, v0, axes, initValue, keepDims);
            var v4 = IR.F.Math.Binary(BinaryOp.Sub, v0, v3);
            var v5 = IR.F.Math.Unary(UnaryOp.Sqrt, v4);
            var v6 = IR.F.Tensors.Reduce(ReduceOp.Mean, v5, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            var v7 = IR.F.Math.Binary(BinaryOp.Add, v6, 1e-05f);
            var v8 = IR.F.Math.Unary(UnaryOp.Rsqrt, v7);
            var v9 = IR.F.Math.Binary(BinaryOp.Mul, v8, 0.5f);
            var v10 = IR.F.Math.Binary(BinaryOp.Mul, v0, v9);
            var v2 = IR.F.Math.Binary(BinaryOp.Mul, v3, 1e-05f);
            var v1 = IR.F.Math.Binary(BinaryOp.Sub, 0.5f, v2);
            var v11 = IR.F.Math.Binary(BinaryOp.Add, v10, v1);
            rootPre = v11;
        }

        TestNotMatch(
            rootPre,
            new IRewriteRule[]
            {
                new FoldLayerNormPattern4(),
                new FoldConstCall(),
            });
    }
}
