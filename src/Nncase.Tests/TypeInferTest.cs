using Xunit;
using Nncase;
using Nncase.IR;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.IR.Utility;

public class UnitTestTypeInfer
{
    [Fact]
    public void TestInferBinary()
    {
        Var a = new Var(new TensorType(DataType.Float32, new[] { 1, 5, 1 }));
        Const b = (Const)(new DenseTensor<float>(Enumerable.Repeat(1.0f, 15).ToArray(), new[] { 1, 5, 3 }));
        var c = a + b;
        var ctype = TypeInference.InferenceType(c);

        Assert.True(HasShape(new[] { 1, 5, 3 }).MatchLeaf(c.CheckedType));
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
        var pads = Const.FromSpan<int>(new[] {0, 0, 1, 1, 2, 2, 3, 3}, new Shape(4, 2));
        var pad = Pad(a, pads, PadMode.Constant, 1);
        Assert.True(TypeInference.InferenceType(pad));
        Assert.Equal(pad.CheckedShape, new Shape(1, 5, 228, 230));
    }

    [Fact]
    public void TestSlice()
    {
        var input = Const.FromSpan<int>(new[] {1, 7, 7, 75});
        var begin = Const.FromSpan<int>(new[] {0});
        var end = Const.FromSpan<int>(new[] {1});
        var stride = Const.FromSpan<int>(new[] {1});
        var axis = Const.FromSpan<int>(new[] {0});
        var s = Slice(input, begin, end, axis, stride);
        Assert.True(TypeInference.InferenceType(s));
        var post = s.Eval().ToConst();
        Assert.Equal(s.CheckedShape, post.CheckedShape);
    }

    [Fact]
    public void SliceShapeOp()
    {
        var begin = new[] { 1 };
        var end = new[] { 3 };
        var stride = new[] { 1 };
        var axes = new[] { 0 };
        var slice = Slice(new Shape(1, 7, 7, 768), begin, end, axes, stride);
        TypeInference.InferenceType(slice);
        var post = slice.Eval().ToConst();
        Assert.Equal(new Shape(2), post.CheckedShape);
    }
}