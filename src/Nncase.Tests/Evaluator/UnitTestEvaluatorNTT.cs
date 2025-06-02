// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.HighPerformance;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;

namespace Nncase.Tests.EvaluatorTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestEvaluatorNTT : TestClassBase
{
    public const int Lanes = 32;

    public static TheoryData<long[][], int[][], int> PackedConcatData { get; } = new()
    {
        { new[] { new long[] { 1, 64, 384, 64 }, new long[] { 1, 64, 384, 64 } }, new[] { new[] { 2, 3 }, new[] { 2, 3 } }, 1 },
    };

    [Theory]
    [InlineData(new object[] { false, new long[] { 1, 1, 4, 4 }, new long[] { 8, 1, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 } })]
    [InlineData(new object[] { false, new long[] { 3, 2, 4, 4 }, new long[] { 8, 2, 3, 3 }, new int[] { 0, 0, 1, 1 }, new int[] { 1, 2 } })]
    [InlineData(new object[] { false, new long[] { 3, 2, 4, 4 }, new long[] { 8, 2, 3, 3 }, new int[] { 1, 0, 1, 1 }, new int[] { 2, 1 } })]
    [InlineData(new object[] { true, new long[] { 1, 4, 4, 4 }, new long[] { 8, 4, 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 } })]
    [InlineData(new object[] { true, new long[] { 3, 8, 4, 4 }, new long[] { 8, 8, 3, 3 }, new int[] { 0, 0, 1, 1 }, new int[] { 1, 2 } })]
    [InlineData(new object[] { true, new long[] { 3, 8, 4, 4 }, new long[] { 8, 8, 3, 3 }, new int[] { 1, 0, 1, 1 }, new int[] { 2, 1 } })]
    public void TestIm2colConv(bool pack, long[] inputShape, long[] weightShape, int[] padding, int[] strides)
    {
        var dilation = new[] { 1, 1 };
        var groups = 1;
        var input = new Var(new TensorType(DataTypes.Float32, inputShape));
        var weights = new Var(new TensorType(DataTypes.Float32, weightShape));
        var bias = IR.F.Random.Normal(DataTypes.Float32, new[] { weightShape[0] });
        var pre = IR.F.NN.Conv2D(input, weights, bias, strides, new[,] { { padding[0], padding[1] }, { padding[2], padding[3] } }, dilation, PadMode.Constant, groups);
        var outShape = pre.CheckedShape.ToValueArray();

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate() },
            { weights, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, weightShape).Evaluate() },
        };

        Expr post;
        {
            if (pack)
            {
                var col = IR.F.NTT.Im2col(IR.F.Tensors.Pack(input, new[] { 4 }, new[] { 1 }), new[] { weightShape[2], weightShape[3] }, strides, padding, new[] { 1 }, new[] { 0 });
                var newW = IR.F.Tensors.Reshape(IR.F.Tensors.Pack(weights, new[] { 4 }, new[] { 1 }), new[] { weightShape[0], weightShape[1] / 4 * weightShape[2] * weightShape[3] });
                var matmul = IR.F.NTT.PackedMatMul(newW, col, new[] { 1 }, new[] { 0 }, false, false, false); // [oc, b*oh*ow]
                var newBias = IR.F.Tensors.Reshape(bias, new[] { weightShape[0], 1 });
                var add = IR.F.Tensors.Reshape(matmul + newBias, new[] { outShape[1], outShape[0], outShape[2], outShape[3] });
                post = IR.F.Tensors.Transpose(add, new[] { 1, 0, 2, 3 });
            }
            else
            {
                var col = IR.F.NTT.Im2col(input, new[] { weightShape[2], weightShape[3] }, strides, padding);
                var newW = IR.F.Tensors.Reshape(weights, new[] { weightShape[0], weightShape[1] * weightShape[2] * weightShape[3] });
                var matmul = IR.F.Tensors.MatMul(newW, col); // [oc, b*oh*ow]
                var newBias = IR.F.Tensors.Reshape(bias, new[] { weightShape[0], 1 });
                var add = IR.F.Tensors.Reshape(matmul + newBias, new[] { outShape[1], outShape[0], outShape[2], outShape[3] });
                post = IR.F.Tensors.Transpose(add, new[] { 1, 0, 2, 3 });
            }
        }

        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

#if false
    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 2 } })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 0 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 0 } })] // packed on axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 2 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 0, new[] { 2 } })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 1, new[] { 0 } })]
    [InlineData(new object[] { new[] { 32, 64, 135 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 0, new[] { 0 } })] // padded and packed on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 135 }, 2, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 0, 1 } })]
    public void TestPackedSoftmax(long[] shape, int axis, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Softmax(input, axis);

        Expr post;
        {
            var lanes = Enumerable.Repeat(Lanes, packedAxes.Length).ToArray();
            var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, shape, packedAxes, lanes, float.NegativeInfinity, out var pads), lanes, packedAxes);
            var softmax = IR.F.Tensors.PackedSoftmax(packed, axis, packedAxes);
            post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(softmax, packedAxes), shape, pads);
        }

        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0 })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1 })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2 })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 0 })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 1 })]
    [InlineData(new object[] { new[] { 32, 64, 135 }, 2 })]
    public void TestPackSoftmaxRule(long[] shape, int axis)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Softmax(input, axis);
        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        var rule = new Passes.Rules.NTT.PackSoftmax();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 0 } })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 0 } })] // packed on axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2, new[] { 2 } })]
    [InlineData(new object[] { new[] { 36, 64, 128 }, 1, new[] { 0 } })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 2, new[] { 1 } })]
    [InlineData(new object[] { new[] { 35, 64, 128 }, 0, new[] { 0 } })]// padded and packed on axis
    [InlineData(new object[] { new[] { 32, 60, 128 }, 0, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 199 }, 0, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 57, 128 }, 1, new[] { 1 } })]
    [InlineData(new object[] { new[] { 32, 64, 81 }, 1, new[] { 2 } })]
    [InlineData(new object[] { new[] { 32, 64, 99 }, 2, new[] { 2 } })]
    public void TestPackedLayerNorm(long[] shape, int axis, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = shape.Skip(axis).ToArray();
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.LayerNorm(axis, 1e-6f, input, scale, bias, false);

        Expr post;
        {
            var lanes = Enumerable.Repeat(Lanes, packedAxes.Length).ToArray();
            var packedInput = IR.F.Tensors.Pack(PackUtility.PadForPack(input, shape, packedAxes, lanes, 0f, out var padsInput), lanes, packedAxes);

            var pAxes = packedAxes.Where(i => i >= axis).Select(i => i - axis).ToArray();
            var packedScale = PackUtility.PadForPack(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                packedScale = IR.F.Tensors.Pack(packedScale, Enumerable.Repeat(Lanes, pAxes.Length).ToArray(), pAxes);
            }

            var packedBias = PackUtility.PadForPack(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                packedBias = IR.F.Tensors.Pack(packedBias, Enumerable.Repeat(Lanes, pAxes.Length).ToArray(), pAxes);
            }

            var layernorm = IR.F.Tensors.PackedLayerNorm(packedInput, packedScale, packedBias, axis, 1e-6f, false, packedAxes, padsInput);

            post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(layernorm, packedAxes), shape, padsInput);
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 1 })] // unrelated with axis
    [InlineData(new object[] { new[] { 32, 64, 128 }, 2 })]
    [InlineData(new object[] { new[] { 32, 64, 128 }, 0 })] // packed on axis
    [InlineData(new object[] { new[] { 36, 64, 128 }, 1 })] // padded but packed not on axis
    [InlineData(new object[] { new[] { 32, 69, 128 }, 2 })]
    [InlineData(new object[] { new[] { 35, 64, 128 }, 0 })]// padded and packed on axis
    [InlineData(new object[] { new[] { 32, 60, 128 }, 0 })]
    [InlineData(new object[] { new[] { 32, 64, 199 }, 0 })]
    [InlineData(new object[] { new[] { 32, 57, 128 }, 1 })]
    [InlineData(new object[] { new[] { 32, 64, 81 }, 1 })]
    [InlineData(new object[] { new[] { 32, 64, 99 }, 2 })]
    public void TestPackLayerNormRule(long[] shape, int axis)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pshape = shape.Skip(axis).ToArray();
        var scale = new Var(new TensorType(DataTypes.Float32, pshape));
        var bias = new Var(new TensorType(DataTypes.Float32, pshape));
        var pre = IR.F.NN.LayerNorm(axis, 1e-6f, input, scale, bias, false);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
            { scale, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
            { bias, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, pshape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackLayerNorm();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 12, 128, 768 }, new[] { 12, 768, 64 }, new[] { 2 }, new[] { 1 } })] // no broadcast, no pad
    [InlineData(new object[] { new[] { 12, 128, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // no broadcast, no pad
    [InlineData(new object[] { new[] { 1, 128, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, no pad
    [InlineData(new object[] { new[] { 1, 129, 768 }, new[] { 12, 768, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 128, 777 }, new[] { 12, 777, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 64 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 58 }, new[] { 1, 2 }, new[] { 1, 2 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 1, 12 * 32, 256 * 32 }, new[] { 64, 256 * 32, 4 * 32 }, new[] { 2, 3 }, new[] { 1, 2 } })] // onnx bug
    public void TestPackedMatMul(int[] lhsShape, int[] rhsShape, int[] lhsPackedAxes, int[] rhsPackedAxes)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        Expr post;
        {
            var lLanes = Enumerable.Repeat(Lanes, lhsPackedAxes.Length).ToArray();
            var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lLanes, 0f, out var lhsPadNums), lLanes, lhsPackedAxes);
            var rLanes = Enumerable.Repeat(Lanes, rhsPackedAxes.Length).ToArray();
            var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rLanes, 0f, out var rhsPadNums), rLanes, rhsPackedAxes);

            var matmul = IR.F.NTT.PackedMatMul(packedLhs, packedRhs, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);
            var lhsAlign = System.Math.Max(lhsShape.Length, rhsShape.Length) - lhsShape.Length;
            var rhsAlign = System.Math.Max(lhsShape.Length, rhsShape.Length) - rhsShape.Length;
            post = matmul;
            if (lhsPackedAxes.Length == 2 && rhsPackedAxes.Length == 2)
            {
                post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(matmul, new[] { lhsAlign + lhsPackedAxes[0], rhsAlign + rhsPackedAxes[1] }), pre.CheckedShape.ToValueArray(), new[] { lhsPadNums[0], rhsPadNums[1] });
            }
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 12, 128, 768 }, new[] { 12, 768, 64 } })] // no broadcast, no pad
    [InlineData(new object[] { new[] { 1, 128, 768 }, new[] { 12, 768, 64 } })] // broadcast, no pad
    [InlineData(new object[] { new[] { 1, 129, 768 }, new[] { 12, 768, 64 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 128, 777 }, new[] { 12, 777, 64 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 64 } })] // broadcast, pad
    [InlineData(new object[] { new[] { 1, 131, 776 }, new[] { 12, 776, 58 } })] // broadcast, pad
    public void TestPackMatMulRule(int[] lhsShape, int[] rhsShape)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Tensors.MatMul(lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };
        var rule = new Passes.Rules.NTT.PackMatMul();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 12, 128, 768 } })]
    [InlineData(new object[] { new[] { 1, 128, 768 } })]
    public void TestPackUnaryRule(long[] shape)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Math.Unary(UnaryOp.Neg, input);

        var feedDict = new Dictionary<IVar, IValue>() {
            { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() },
        };
        var rule = new Passes.Rules.NTT.PackUnary();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 768 }, new int[] { 1, 77, 768 }, new[] { 1, 2 }, new int[] { 1, 2 } })] // normal
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 }, new[] { 0 }, new[] { 1, 2 }, false })] // packed on broadcast axis, invalid
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 }, new[] { 1, 2 }, new[] { 1, 2 }, false })] // packed on broadcast axis, invalid
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 }, new[] { 0, 2 }, new[] { 2 }, false })] // packed on broadcast axis, invalid
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 }, new[] { 1, 2 }, new[] { 2 } })] // packed on no broadcast axis, 2d simd with 1d simd.
    [InlineData(new object[] { BinaryOp.Mul, new[] { 12, 77, 64 }, new int[] { }, new[] { 1, 2 }, new int[] { } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 1 }, new int[] { 1 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 2 }, new int[] { 2 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 }, new[] { 1, 2 }, new int[] { 1, 2 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 768 }, new int[] { 768 }, new[] { 1, 2 }, new int[] { 0 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 3072 }, new int[] { 3072 }, new[] { 1, 2 }, new int[] { 0 } })]
    [InlineData(new object[] { BinaryOp.Div, new[] { 1, 64, 384, 384 }, new int[] { 1 }, new[] { 2, 3 }, new int[] { } })]
    public void TestPackedBinary(BinaryOp op, int[] lhsShape, int[] rhsShape, int[] lhsPackedAxes, int[] rhsPackedAxes, bool valid = true)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        Expr post;
        {
            var lhsLanes = Enumerable.Repeat(Lanes, lhsPackedAxes.Length).ToArray();
            var packedLhs = IR.F.Tensors.Pack(PackUtility.PadForPack(lhs, lhsShape, lhsPackedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsPackedAxes);
            var rhsLanes = Enumerable.Repeat(Lanes, rhsPackedAxes.Length).ToArray();
            var packedRhs = IR.F.Tensors.Pack(PackUtility.PadForPack(rhs, rhsShape, rhsPackedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsPackedAxes);

            var binary = IR.F.NTT.PackedBinary(packedLhs, packedRhs, op, lhsPackedAxes, lhsPadNums, rhsPackedAxes, rhsPadNums);

            post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(binary, lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPackedAxes : rhsPackedAxes), pre.CheckedShape.ToValueArray(), lhsPackedAxes.Length >= rhsPackedAxes.Length ? lhsPadNums : rhsPadNums);
        }

        if (!valid)
        {
            Assert.IsType<InvalidType>(post.CheckedType);
            return;
        }

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 77, 768 }, new int[] { 1, 77, 768 } })] // normal
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 64 }, new[] { 12, 1, 64 } })] // packed on broadcast axis, invalid
    [InlineData(new object[] { BinaryOp.Mul, new[] { 12, 77, 64 }, new int[] { } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 12, 77, 77 }, new int[] { 1, 77, 77 } })]
    [InlineData(new object[] { BinaryOp.Mul, new[] { 1, 77, 3072 }, new int[] { 3072 } })]
    [InlineData(new object[] { BinaryOp.Add, new[] { 1, 64, 96, 128 }, new int[] { 1 } })] // normal
    public void TestPackBinaryRule(BinaryOp op, int[] lhsShape, int[] rhsShape)
    {
        var lhs = new Var(new TensorType(DataTypes.Float32, lhsShape));
        var rhs = new Var(new TensorType(DataTypes.Float32, rhsShape));
        var pre = IR.F.Math.Binary(op, lhs, rhs);

        var feedDict = new Dictionary<IVar, IValue>() {
            { lhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, lhsShape).Evaluate() },
            { rhs, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 3, rhsShape).Evaluate() },
        };

        var rule = new Passes.Rules.NTT.PackBinary();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 77, 768 }, new[] { 2 } })]
    [InlineData(new object[] { new[] { 1, 77, 768 }, new[] { 1 } })]
    public void TestPackedSwish(long[] shape, int[] packedAxes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Swish(input, 1.23f);

        Expr post;
        {
            var lanes = Enumerable.Repeat(Lanes, packedAxes.Length).ToArray();
            var packed = IR.F.Tensors.Pack(PackUtility.PadForPack(input, shape, packedAxes, lanes, 0f, out var pads), lanes, packedAxes);
            var swish = IR.F.NN.Swish(packed, 1.23f);
            post = PackUtility.SliceForPack(IR.F.Tensors.Unpack(swish, packedAxes), shape, pads);
        }

        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 77, 768 } })]
    public void TestPackSwishRule(long[] shape)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.NN.Swish(input, 1.23f);

        var rule = new Passes.Rules.NTT.PackSwish();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 0, 1, 3, 2 } })]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 0, 3, 1, 2 } })]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 3, 0, 1, 2 } })]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 1, 0, 3, 2 } })]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 0, 3, 2, 1 } })]
    [InlineData(new object[] { new[] { 1, 32, 64, 96 }, new[] { 3, 0, 2, 1 } })]
    public void TestPackTransposeRule(long[] shape, int[] perm)
    {
        // NOTE the big shape will make ortki crash
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Tensors.Transpose(input, perm);

        var rule = new Passes.Rules.NTT.PackTranspose();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        foreach (var post in posts)
        {
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 384, 4096 }, new[] { 1 } })]
    public void TestPackUnsqueezeRule(long[] shape, int[] axes)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Tensors.Unsqueeze(input, axes);

        var rule = new Passes.Rules.NTT.PackUnsqueeze();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        foreach (var post in posts)
        {
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 384, 128 }, new[] { 1, 1, 384, 128 } })]
    [InlineData(new object[] { new[] { 1, 384, 32, 128 }, new[] { 1, 384, 4096 } })]
    [InlineData(new object[] { new[] { 1, 384, 64, 128 }, new[] { 1, 384, 8192 } })]
    public void TestPackReshapeRule(long[] shape, long[] newShape)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Tensors.Reshape(input, newShape);

        var rule = new Passes.Rules.NTT.PackReshape();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [InlineData(new object[] { new[] { 1, 32, 384, 128 }, new[] { 64L }, new[] { long.MaxValue }, 3 })]
    [InlineData(new object[] { new[] { 1, 32, 384, 128 }, new[] { 0L }, new[] { 64L }, 3 })]
    public void TestPackSliceRule(long[] shape, long[] start, long[] stop, long axis)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var pre = IR.F.Tensors.Slice(input, start, stop, new[] { axis }, new[] { 1 });

        var feedDict = new Dictionary<IVar, IValue>() { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, shape).Evaluate() } };
        var rule = new Passes.Rules.NTT.PackSlice();
        CompilerServices.TryMatch(pre, rule.Pattern, out var result);
        var posts = rule.GetReplaceCandidates(result!, new Passes.RunPassContext());
        foreach (var post in posts)
        {
#if DEBUG
            System.Console.WriteLine(CompilerServices.Print(post));
#endif
            Comparator.Compare(pre.Evaluate(feedDict), post.Evaluate(feedDict), 0.999f);
        }
    }

    [Theory]
    [MemberData(nameof(PackedConcatData))]
    public void TestPackedConcat(int[][] shapes, int[][] packedAxes, int axis)
    {
        var inputs = shapes.Select(shape => new Var(new TensorType(DataTypes.Float32, shape))).ToArray();
        var pre = IR.F.Tensors.Concat(new IR.Tuple(inputs), axis);
        int count = 1;
        var feedDict = shapes.Zip(inputs).ToDictionary(kv => kv.Second, kv => IR.F.Random.Normal(DataTypes.Float32, 0, 1, count++, kv.First).Evaluate());
        var post = IR.F.Tensors.Concat(new IR.Tuple(inputs.Zip(packedAxes).Select(p => IR.F.Tensors.Pack(p.First, Enumerable.Repeat(Lanes, p.Second.Length).ToArray(), p.Second)).ToArray()), axis);
        post.Evaluate(feedDict);
    }
#endif
}
