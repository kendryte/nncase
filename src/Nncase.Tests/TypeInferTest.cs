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
using static Nncase.Pattern.Utility;

namespace Nncase.Tests;

public class UnitTestTypeInfer : IHostFixtrue
{

    public UnitTestTypeInfer(IHost host) : base(host)
    {

    }

    [Fact]
    public void TestInferBinary()
    {
        Var a = new Var(new TensorType(DataType.Float32, new[] { 1, 5, 1 }));
        Const b = (Const)(new DenseTensor<float>(Enumerable.Repeat(1.0f, 15).ToArray(), new[] { 1, 5, 3 }));
        var c = a + b;
        var ctype = TypeInference.InferenceType(c);

        Assert.True(IsShape(new[] { 1, 5, 3 }).MatchLeaf(c.CheckedType));
    }

    [Fact]
    public void TestInferUnary()
    {
        Var a = new Var(AnyType.Default);
        var c = Square(a);
        Assert.False(TypeInference.InferenceType(c));
    }

    [Fact]
    public void TestInferPad()
    {
        var a = new Var(new TensorType(DataType.Float32, new Shape(1, 3, 224, 224)));
        var pads = Const.FromSpan<int>(new[] { 0, 0, 1, 1, 2, 2, 3, 3 }, new Shape(4, 2));
        var pad = Pad(a, pads, PadMode.Constant, 1);
        Assert.True(TypeInference.InferenceType(pad));
        Assert.Equal(pad.CheckedShape, new Shape(1, 5, 228, 230));
    }

    [Fact]
    public void TestSlice()
    {
        var input = Const.FromSpan<int>(new[] { 1, 7, 7, 75 });
        var begin = Const.FromSpan<int>(new[] { 0 });
        var end = Const.FromSpan<int>(new[] { 1 });
        var stride = Const.FromSpan<int>(new[] { 1 });
        var axis = Const.FromSpan<int>(new[] { 0 });
        var s = Slice(input, begin, end, axis, stride);
        Assert.True(TypeInference.InferenceType(s));
        var post = s.Eval().ToConst();
        Assert.True(post.InferenceType());
        Assert.Equal(s.CheckedShape, post.CheckedShape);
    }

    [Fact]
    public void TestSliceShapeOp()
    {
        var begin = new[] { 1 };
        var end = new[] { 3 };
        var stride = new[] { 1 };
        var axes = new[] { 0 };
        var slice = Slice(new Shape(1, 7, 7, 768), begin, end, axes, stride);
        TypeInference.InferenceType(slice);
        var post = slice.Eval().ToConst();
        Assert.True(post.InferenceType());
        Assert.Equal(new Shape(2), post.CheckedShape);
    }

    [Fact]
    public void TestStack()
    {
        var a = (Const)1;
        var b = (Const)1;
        var c = (Const)1;
        var s = Stack(new Tuple(a, b, c), 0);
        TypeInference.InferenceType(s);
        Assert.Equal(new Shape(3), s.CheckedShape);

        var x = Const.FromSpan<int>(new[] { 1, 2 });
        var y = Const.FromSpan<int>(new[] { 1, 2 });
        var z = Const.FromSpan<int>(new[] { 1, 2 });
        var ss = Stack(new Tuple(x, y, z), 1);
        TypeInference.InferenceType(ss);
        Assert.Equal(new Shape(2, 3), ss.CheckedShape);
    }

    void AssertInferShape(Expr expr, params int[] shapeDimensions)
    {
        AssertInferShape(expr, new Shape(shapeDimensions));
    }

    void AssertInferShape(Expr expr, Shape shape)
    {
        Assert.True(TypeInference.InferenceType(expr));
        Assert.Equal(expr.CheckedShape, shape);
    }

    [Fact]
    public void TestReduceArgTypeInfer()
    {
        var input = new Var("v", new TensorType(DataType.Float32, new Shape(4, 5, 6, 7)));
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 0, false, false),
            5, 6, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 1, false, false),
            4, 6, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 2, false, false),
            4, 5, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 3, false, false),
            4, 5, 6);

        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 0, true, false),
            1, 5, 6, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 1, true, false),
            4, 1, 6, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 2, true, false),
            4, 5, 1, 7);
        AssertInferShape(
            ReduceArg(ReduceArgOp.ArgMax, input, 3, true, false),
            4, 5, 6, 1);
    }
}