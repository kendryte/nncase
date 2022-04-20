using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
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

public class UnitTestTypeInfer : IHostFixtrue
{

    public UnitTestTypeInfer(IHost host) : base(host)
    {

    }

    [Fact]
    public void TestInferBinary()
    {
        Var a = new Var(new TensorType(DataTypes.Float32, new[] { 1, 5, 1 }));
        var b = Tensor.FromScalar(1.0f, new[] { 1, 5, 3 });
        var c = a + b;
        var ctype = CompilerServices.InferenceType(c);

        Assert.True(HasShape(new[] { 1, 5, 3 }).MatchLeaf(c.CheckedType));
    }

    [Fact]
    public void TestInferUnary()
    {
        Var a = new Var(AnyType.Default);
        var c = Square(a);
        Assert.False(CompilerServices.InferenceType(c));
    }

    [Fact]
    public void TestInferPad()
    {
        var a = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var pads = Tensor.FromSpan(new[] { 0, 0, 1, 1, 2, 2, 3, 3 }, new Shape(4, 2));
        var pad = Pad(a, pads, PadMode.Constant, 1f);
        Assert.True(CompilerServices.InferenceType(pad));
        Assert.Equal(pad.CheckedShape, new Shape(1, 5, 228, 230));
    }

    [Fact]
    public void TestSlice()
    {
        var input = Tensor.FromSpan<int>(new[] { 1, 7, 7, 75 });
        var begin = Tensor.FromSpan<int>(new[] { 0 });
        var end = Tensor.FromSpan<int>(new[] { 1 });
        var stride = Tensor.FromSpan<int>(new[] { 1 });
        var axis = Tensor.FromSpan<int>(new[] { 0 });
        var s = Slice(input, begin, end, axis, stride);
        Assert.True(CompilerServices.InferenceType(s));
        var post = s.Evaluate();
        Assert.Equal(s.CheckedShape, ((TensorType)post.Type).Shape);
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

        var x = Tensor.FromSpan<int>(new[] { 1, 2 });
        var y = Tensor.FromSpan<int>(new[] { 1, 2 });
        var z = Tensor.FromSpan<int>(new[] { 1, 2 });
        var ss = Stack(new Tuple(x, y, z), 1);
        CompilerServices.InferenceType(ss);
        Assert.Equal(new Shape(2, 3), ss.CheckedShape);
    }

    void CheckInferShape(Expr expr, params int[] shapeDimensions)
    {
        CheckInferShape(expr, new Shape(shapeDimensions));
    }

    void CheckInferShape(Expr expr, Shape shape)
    {
        Assert.True(CompilerServices.InferenceType(expr));
        Assert.Equal(expr.CheckedShape, shape);
    }

    [Fact]
    public void TestReduceArgTypeInfer()
    {
        var input = new Var("v", new TensorType(DataTypes.Float32, new Shape(4, 5, 6, 7)));
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 0, false, false),
            5, 6, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 1, false, false),
            4, 6, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 2, false, false),
            4, 5, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 3, false, false),
            4, 5, 6);

        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 0, true, false),
            1, 5, 6, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 1, true, false),
            4, 1, 6, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 2, true, false),
            4, 5, 1, 7);
        CheckInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 3, true, false),
            4, 5, 6, 1);
    }


    void CheckReshape(Expr input, int[] reshapeArgs, int[] expectShape)
    {
        CheckInferShape(Reshape(input, reshapeArgs), expectShape);
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
        Assert.False(CompilerServices.InferenceType(f));

        // 2. after the  transfrom the dag is valid type
        var y = x with { TypeAnnotation = TensorType.Scalar(DataTypes.Int32) };
        var new_f = f with { Body = y + b, Parameters = new[] { y } };
        Assert.True(CompilerServices.InferenceType(new_f));
    }

    [Fact]
    public void TestResize()
    {
        var resize = IR.F.Imaging.ResizeImage(ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 1, 3, 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize));
        Assert.True(HasShape(new[] { 1, 3, 32, 48 }).MatchLeaf(resize.CheckedType!));

        var resize2 = IR.F.Imaging.ResizeImage(ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 3, 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize2));
        Assert.True(HasShape(new[] { 32, 48, 67 }).MatchLeaf(resize2.CheckedType!));

        var resize3 = IR.F.Imaging.ResizeImage(ImageResizeMode.NearestNeighbor,
            IR.F.Random.Uniform(DataTypes.Float32, 0, 2, 1, new[] { 34, 67 }),
            float.NaN,
            Const.FromShape(new[] { 32, 48 }));
        Assert.True(CompilerServices.InferenceType(resize3));
        Assert.True(HasShape(new[] { 32, 48 }).MatchLeaf(resize3.CheckedType!));
    }
}