// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Hosting;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.CoreTest;

public class UnitTypeInferBase : TestClassBase
{
    public void CheckInferShape(Expr expr, params int[] shapeDimensions)
    {
        CheckInferShape(expr, new Shape(shapeDimensions));
    }

    public void CheckInferShape(Expr expr, Shape expectShape)
    {
        Assert.True(CompilerServices.InferenceType(expr));
        Assert.Equal(expectShape, expr.CheckedShape);
    }

    public void CheckInferType(Expr expr, DataType dt, Shape shape)
    {
        Assert.True(CompilerServices.InferenceType(expr));
        Assert.Equal(new TensorType(dt, shape), expr.CheckedType);
    }

    public Var Var(Shape shape, DataType dt) => new Var(new TensorType(dt, shape));

    public Var Var(Shape shape) => Var(shape, DataTypes.Float32);
}

public class UnitTestTypeInfer : UnitTypeInferBase
{
    public UnitTestTypeInfer()
        : base()
    {
    }

    public static IEnumerable<object[]> TestMatMulData =>
        new[]
        {
            new object[] { new[] { 3, 10, 128 }, new[] { 128, 128 }, new[] { 3, 10, 128 } },
            new object[] { new[] { 10, 128 }, new[] { 2, 128, 128 }, new[] { 2, 10, 128 } },
            new object[] { new[] { 4, 10, 128 }, new[] { 1, 128, 128 }, new[] { 4, 10, 128 } },
            new object[] { new[] { 1, 10, 128 }, new[] { 4, 128, 128 }, new[] { 4, 10, 128 } },
        };

    [Fact]
    public void TestInferBinary()
    {
        var a = new Var(new TensorType(DataTypes.Float32, new[] { 1, 5, 1 }));
        var b = Tensor.FromScalar(1.0f, new[] { 1, 5, 3 });
        var c = a + b;
        _ = CompilerServices.InferenceType(c);

        Assert.True(HasShape(new[] { 1, 5, 3 }).MatchLeaf(c.CheckedType!));
    }

    [Fact]
    public void TestInferUnary()
    {
        var a = new Var(AnyType.Default);
        var c = Square(a);
        CompilerServices.InferenceType(c);
        Assert.IsType<AnyType>(c.CheckedType);
    }

    [Fact]
    public void TestInferPad()
    {
        var a = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var pads = Tensor.From(new[] { 0, 0, 1, 1, 2, 2, 3, 3 }, new Shape(4, 2));
        var pad = Pad(a, pads, PadMode.Constant, 1f);
        Assert.True(CompilerServices.InferenceType(pad));
        Assert.Equal(pad.CheckedShape, new Shape(1, 5, 228, 230));
    }

    [Fact]
    public void TestSlice()
    {
        var input = Tensor.From<int>(new[] { 1, 7, 7, 75 });
        var begin = Tensor.From<int>(new[] { 0 });
        var end = Tensor.From<int>(new[] { 1 });
        var stride = Tensor.From<int>(new[] { 1 });
        var axis = Tensor.From<int>(new[] { 0 });
        var s = Slice(input, begin, end, axis, stride);
        Assert.True(CompilerServices.InferenceType(s));
        var post = s.Evaluate();
        Assert.Equal(s.CheckedShape, ((TensorType)post.Type).Shape);
    }

    [Fact]
    public void TestSlice2()
    {
        var input_a = new Var("input_a", new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, Dimension.Unknown, Dimension.Unknown }));
        var repeats = IR.F.Tensors.Slice(IR.F.Tensors.ShapeOf(input_a), new[] { -2 }, new[] { -1 }, 1);
        Assert.True(CompilerServices.InferenceType(repeats));
        Assert.True(repeats.CheckedShape.Rank == 1);
    }

    [Fact]
    public void TestSliceShapeOp()
    {
        var begin = new[] { 1 };
        var end = new[] { 3 };
        var stride = new[] { 1 };
        var axes = new[] { 0 };
        var slice = Slice(new Shape(1, 7, 7, 768), begin, end, axes, stride);
        CompilerServices.InferenceType(slice);
        var post = slice.Evaluate();
        Assert.Equal(new Shape(2), ((TensorType)post.Type).Shape);
    }

    [Fact]
    public void TestStack()
    {
        var a = (Const)1;
        var b = (Const)1;
        var c = (Const)1;
        var s = Stack(new Tuple(a, b, c), 0);
        CompilerServices.InferenceType(s);
        Assert.Equal(new Shape(3), s.CheckedShape);

        var x = Tensor.From<int>(new[] { 1, 2 });
        var y = Tensor.From<int>(new[] { 1, 2 });
        var z = Tensor.From<int>(new[] { 1, 2 });
        var ss = Stack(new Tuple(x, y, z), 1);
        CompilerServices.InferenceType(ss);
        Assert.Equal(new Shape(2, 3), ss.CheckedShape);
    }

    [Fact]
    public void TestReduceArgTypeInfer()
    {
        var input = new Var("v", new TensorType(DataTypes.Float32, new Shape(4, 5, 6, 7)));
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 0, false, false), 5, 6, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 1, false, false), 4, 6, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 2, false, false), 4, 5, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 3, false, false), 4, 5, 6);

        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 0, true, false), 1, 5, 6, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 1, true, false), 4, 1, 6, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 2, true, false), 4, 5, 1, 7);
        CheckInferShape(ReduceArg(ReduceArgOp.ArgMax, DataTypes.Int64, input, 3, true, false), 4, 5, 6, 1);
    }

    [Fact]
    public void TestInferReshape()
    {
        var input = new Var("v", new TensorType(DataTypes.Float32, new Shape(4, 5, 6, 7)));
        CheckReshape(input, new[] { 8, 5, 3, 7 }, new[] { 8, 5, 3, 7 });
        CheckReshape(input, new[] { -1, 5, 6, 7 }, new[] { 4, 5, 6, 7 });
        CheckReshape(input, new[] { -1, 5, 3, 7 }, new[] { 8, 5, 3, 7 });
        CheckReshape(input, new[] { -1 }, new[] { 4 * 5 * 6 * 7 });
    }

    [Fact]
    public void TestReInference()
    {
        // 1. before the transform the dag is invalid type
        Var x = new("x");
        Const b = 2;
        Function f = new("f", x + b, new[] { x });
        CompilerServices.InferenceType(f);
        Assert.IsType<AnyType>(f.Body.CheckedType);

        // 2. after the  transfrom the dag is valid type
        var y = x.With(typeAnnotation: TensorType.Scalar(DataTypes.Int32));
        var new_f = f.With(body: y + b, parameters: new[] { y });
        Assert.True(CompilerServices.InferenceType(new_f));
    }

    [Fact]
    public void TestResize()
    {
        var resize = IR.F.Imaging.ResizeImage(
            ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 1, 3, 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize));
        Assert.True(HasShape(new[] { 1, 3, 32, 48 }).MatchLeaf(resize.CheckedType!));

        var resize2 = IR.F.Imaging.ResizeImage(
            ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 3, 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize2));
        Assert.True(HasShape(new[] { 32, 48, 67 }).MatchLeaf(resize2.CheckedType!));

        var resize3 = IR.F.Imaging.ResizeImage(
            ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize3));
        Assert.True(HasShape(new[] { 32, 48 }).MatchLeaf(resize3.CheckedType!));
    }

    [Fact]
    public void TestGather0()
    {
        var input = new Var("a", new TensorType(DataTypes.Float32, new Shape(32, 256)));
        var indices = Tensor.From<int>(new[] { 1, 10 });
        var g = Gather(input, 0, indices);
        CheckInferType(g, DataTypes.Float32, new Shape(2, 256));
    }

    [Fact]
    public void TestSlice1()
    {
        var input = new Var("a", new TensorType(DataTypes.Float32, new Shape(1, 10, 256)));
        var ones = Tensor.From<long>(new[] { 1L });
        var begins = ones;
        var ends = Tensor.From<long>(new[] { 9223372036854775807 });
        var axes = ones;
        var s = Slice(input, begins, ends, axes, ones);
        CheckInferType(s, DataTypes.Float32, new Shape(1, 9, 256));
    }

    [Theory]
    [MemberData(nameof(TestMatMulData))]
    public void TestMatMul(int[] lhsShape, int[] rhsShape, int[] expectShape)
    {
        var lhs = Var(lhsShape);
        var rhs = Var(rhsShape);
        CheckInferShape(IR.F.Math.MatMul(lhs, rhs), expectShape);
    }

    [Fact]
    public void TestConv2DInvalidWeights()
    {
        var v30 = new Var("serving_default_input_1:0", new TensorType(DataTypes.Float32, new[] { 1, 256, 56, 56 }));
        var v30_1 = new Marker("RangeOf", Testing.Rand<float>(256, 64, 1, 1), new float[] { -0.3903954f, 0.46443018f }); // f32[256,64,1,1]
        var v30_2 = new Call(new IR.NN.Conv2D(PadMode.Constant), new Expr[] { v30, v30_1, Testing.Rand<float>(256), new int[] { 1, 1 }, new int[,] { { 0, 0 }, { 0, 0 } }, new int[] { 1, 1 }, 1, new float[] { -float.PositiveInfinity, float.PositiveInfinity } }); // f32[1,256,56,56]
        CompilerServices.InferenceType(v30_2);
        Assert.IsType<InvalidType>(v30_2.CheckedType);
    }

    [Fact]
    public void TestConcat()
    {
        var v1 = Var(new[] { 1, 3, 16 });
        var v2 = Var(new[] { 1, 3, 16 });
        var cat = Concat(new Tuple(v1, v2), -1);
        CheckInferShape(cat, new[] { 1, 3, 32 });
    }

    [Fact]
    public void TestUnsqueeze()
    {
        var v1 = Var(new[] { 3 });
        var us1 = Unsqueeze(v1, new[] { -1 });
        CheckInferShape(us1, new[] { 3, 1 });
        var us2 = Unsqueeze(v1, new[] { 0 });
        CheckInferShape(us2, new[] { 1, 3 });
    }

    private void CheckReshape(Expr input, int[] reshapeArgs, int[] expectShape)
    {
        CheckInferShape(Reshape(input, reshapeArgs), expectShape);
    }
}

public class UnitTestDynamicTypeInfer : UnitTypeInferBase
{
    public UnitTestDynamicTypeInfer()
        : base()
    {
    }

    [Fact]
    public void TestRange()
    {
        var begin = Var(Shape.Scalar);
        var end = Var(Shape.Scalar);
        var step = Var(Shape.Scalar);
        var r = Range(begin, end, step);
        CheckInferShape(r, Dimension.Unknown);
    }

    [Fact]
    public void TestConcat()
    {
        var in0 = Var(Shape.Unranked);
        var in1 = Var(Shape.Unranked);
        var cat = Concat(new IR.Tuple(in0, in1), 0);
        CheckInferShape(cat, Shape.Unranked);
    }

    [Fact]
    public void TestBroadcastInfer()
    {
        // appear in where
        var a = new TensorType(DataTypes.Int32, new Shape(new[] { 1, 3, 224, 224 }));
        var b = new TensorType(DataTypes.Float32, Shape.Unknown(4));
        var c = new TensorType(DataTypes.Float32, Shape.Unknown(4));
        var result = (TensorType)TypeInference.BroadcastType(b.DType, a, b, c);
        Assert.Equal(result.DType, DataTypes.Float32);
    }

    private void CheckInferShape(Expr expr, params Dimension[] shapeDimensions)
    {
        CheckInferShape(expr, new Shape(shapeDimensions));
    }
}
