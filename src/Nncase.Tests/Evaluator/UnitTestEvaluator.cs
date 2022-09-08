using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Autofac;
using Autofac.Extras.CommonServiceLocator;
using CommonServiceLocator;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.TestFixture;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static OrtKISharp.TensorHelper;
using RangeOf = Nncase.IR.Math.RangeOf;
using Tuple = Nncase.IR.Tuple;
using static Nncase.Utilities.DumpUtility;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluator : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestEvalFuncCall()
    {
        var cfunc = (int x) => ((x * 10) - 200) / 5;

        var x = new Var("x", TensorType.Scalar(DataTypes.Int32));
        var func = new Function("main", ((x * 10) - 200) / 5, new(new[] { x }));

        Assert.Equal(cfunc(10), new Call(func, 10).Evaluate().AsTensor().ToScalar<int>());

        Assert.Equal(cfunc(10) + cfunc(12), (new Call(func, 10) + new Call(func, 12)).Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestOrtKI()
    {
        var a = Const.FromTensor(Tensor.FromSpan<int>(new[] { 1, 2, 3 }));
        var b = Const.FromTensor(Tensor.FromSpan<int>(new[] { 1, 2, 3 }));
        // var b = (Const) 2;
        a.InferenceType();
        b.InferenceType();
        var na = a.Value.ToOrtTensor();
        var nb = b.Value.ToOrtTensor();
        Assert.Equal(new[] { 1, 2, 3 }, na.ToDense<int>().ToArray());
        var v = na.ToType(OrtDataType.Float16).ToValue();
        var f = na.ToType(OrtDataType.Float16).ToType(OrtDataType.Float);

        var c = na + nb;
        Assert.Equal(new[] { 2, 4, 6 }, c.ToTensor().ToArray<int>());
    }

    [Fact]
    public void TestUnary()
    {
        var a = (Const)7f;
        var tA = OrtTensorFromScalar(7f);
        var expr = -a;
        CompilerServices.InferenceType(expr);
        Assert.Equal(
            -tA,
            expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBinary()
    {
        var tA = OrtTensorFromScalar(1f);
        var tB = tA * 2f;

        var a = (Const)1f;
        var b = (Const)2f;
        var expr = a * b + a;
        CompilerServices.InferenceType(expr);
        Assert.Equal(
            tA * tB + tA,
            expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBinarySub()
    {
        var tA = OrtTensorFromScalar((int)4);
        var tB = OrtTensorFromScalar((int)1);
        var tC = tA - tB;
    }

    [Fact]
    public void TestBinaryShift()
    {
        var tA = OrtTensorFromScalar((uint)1);
        var tB = OrtKI.LeftShift(tA, OrtTensorFromScalar((uint)2));
        var tC = OrtKI.RightShift(tA, OrtTensorFromScalar((uint)2));

        var a = (Const)(uint)1;
        var b = (Const)(uint)2;

        Assert.Equal(
            (uint)1 << 2,
            IR.F.Math.LeftShift(a, b).Evaluate().AsTensor().ToScalar<uint>());

        Assert.Equal(
            (uint)1 >> 2,
            IR.F.Math.RightShift(a, b).Evaluate().AsTensor().ToScalar<uint>());
    }

    [Fact]
    public void TestBinaryShift2()
    {
        var a = (Const)(uint)1;
        var b = (Const)(uint)2;

        Assert.Equal(
            (int)((uint)1 << 2) - 1,
             (IR.F.Tensors.Cast(IR.F.Math.LeftShift(a, b), DataTypes.Int32) - 1).Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestCompare()
    {
        Assert.True(CompilerServices.Evaluate((Expr)5 <= (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)5 <= (Expr)5).AsTensor().ToScalar<bool>());
        Assert.False(CompilerServices.Evaluate((Expr)(-1) <= (Expr)(-2)).AsTensor().ToScalar<bool>());

        Assert.False(CompilerServices.Evaluate((Expr)10 != (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)10 != (Expr)(-2)).AsTensor().ToScalar<bool>());

        Assert.True(CompilerServices.Evaluate((Expr)10 == (Expr)10).AsTensor().ToScalar<bool>());
        Assert.False(CompilerServices.Evaluate((Expr)10 == (Expr)2).AsTensor().ToScalar<bool>());

        Assert.False(CompilerServices.Evaluate((Expr)1 > (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)1 > (Expr)0).AsTensor().ToScalar<bool>());
    }

    [Fact]
    public void TestConcat()
    {
        var a = Const.FromTensor(Tensor.FromSpan<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 })));
        var b = Const.FromTensor(Tensor.FromSpan<int>(new int[12], new Shape(new[] { 1, 3, 4 })));
        var inputList = new Tuple(a, b);
        var expr = Tensors.Concat(inputList, 0);
        CompilerServices.InferenceType(expr);

        var tA = a.Value.ToOrtTensor();
        var tB = b.Value.ToOrtTensor();

        Assert.Equal(
            OrtKI.Concat(new[] { tA, tB }, 0),
            expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConcat2()
    {
        var a = Const.FromTensor(Tensor.FromSpan<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 })));
        var b = Const.FromTensor(Tensor.FromSpan<int>(new int[12], new Shape(new[] { 1, 3, 4 })));
        var inputList = new TupleConst(ImmutableArray.Create<Const>(a, b));
        var expr = Tensors.Concat(inputList, 0);
        CompilerServices.InferenceType(expr);

        var tA = a.Value.ToOrtTensor();
        var tB = b.Value.ToOrtTensor();

        Assert.Equal(
            OrtKI.Concat(new[] { tA, tB }, 0),
            expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestStack()
    {
        Expr a = 1;
        Expr b = 2;
        var inputList = new Tuple(a, b);
        var expr = Tensors.Stack(inputList, 0);
        CompilerServices.InferenceType(expr);

        Assert.Equal(new[] { 1, 2 }, expr.Evaluate().AsTensor().ToArray<int>());
    }

    [Fact]
    public void TestSlice()
    {
        var input = Tensor.FromSpan<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
        var begin = Tensor.FromSpan<int>(new[] { 0, 0, 0, 0 }, new Shape(new[] { 4 }));
        var end = Tensor.FromSpan<int>(new[] { 1, 1, 1, 5 }, new Shape(new[] { 4 }));
        var axes = Tensor.FromSpan<int>(new[] { 0, 1, 2, 3 }, new Shape(new[] { 4 }));
        var strides = Tensor.FromSpan<int>(new[] { 1, 1, 1, 1 }, new Shape(new[] { 4 }));
        var result = Const.FromTensor(Tensor.FromSpan<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] { 1, 1, 1, 5 })));
        var tResult = result.Value.ToOrtTensor();
        var expr = Tensors.Slice(input, begin, end, axes, strides);
        Assert.True(expr.InferenceType());
        Assert.Equal(
            tResult,
            expr.Evaluate().AsTensor().ToOrtTensor()
            );
    }

    [Fact]
    public void TestPad()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.FromSpan<int>(new[] { 0, 0, 0, 0, 1, 1, 2, 2 }, new Shape(new[] { 4, 2 }));
        var value = Tensor.FromScalar<float>(1.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new[] { 1, 1, 4, 7 }, result.Shape);
    }

    [Fact]
    public void TestPad2()
    {
        var tinput = OrtKI.Random(1, 1, 2, 3);
        var input = tinput.ToTensor();
        var pads = Tensor.FromSpan<long>(new long[] { 0, 0, 1, 2, 2, 4, 5, 6 }, new Shape(4, 2));
        var value = Tensor.FromScalar<float>(2.0f);
        var expr = NN.Pad(input, pads, Nncase.PadMode.Constant, value);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(new[] { 1, 4, 8, 14 }, result.Shape);
    }

    [Fact]
    public void TestStackAndCast()
    {
        var padh_before = Tensors.Cast(Tensor.FromSpan<float>(new[] { 1.0f }), Nncase.DataTypes.Int32);
        var padh_after = Tensors.Cast(Tensor.FromSpan<float>(new[] { 2.0f }), Nncase.DataTypes.Int32);
        var padw_before = Tensors.Cast(Tensor.FromSpan<float>(new[] { 3.0f }), Nncase.DataTypes.Int32);
        var padw_after = Tensors.Cast(Tensor.FromSpan<float>(new[] { 4.0f }), Nncase.DataTypes.Int32);

        var expr = Tensors.Stack(new Tuple(
          Tensors.Concat(new Tuple(padh_before, padh_after), 0),
          Tensors.Concat(new Tuple(padw_before, padw_after), 0)), 0);
        CompilerServices.InferenceType(expr);
        var result = expr.Evaluate().AsTensor().ToOrtTensor();
        Assert.Equal(MakeOrtTensor(new[] { 1, 2, 3, 4 }, new[] { 2, 2 }), result);
    }

    [Fact]
    public void TestConv2D()
    {
        // var weights = OrtKI.Random(8, 4, 3, 3);
        // var inputs = OrtKI.Random(1, 4, 5, 5);
        // var bias = OrtKI.Random(8);
        // var output = OrtKI.Conv(inputs, weights, bias, padding: new long[] { 1, 1 });
        //
        // var expr = Conv2D(inputs.ToTensor(), weights.ToTensor(), bias.ToTensor(),
        //     stride: new[] { 1, 1 }, padding: Tensor.FromSpan<int>(new int[] { 1, 1, 1, 1 }, new[] { 2, 2 }),
        //     dilation: new[] { 1, 1 }, Nncase.PadMode.Constant, 1);
        // Assert.True(expr.InferenceType());
        // Assert.Equal(output, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConv2D_1()
    {
        // var input = OrtKI.Random(1, 28, 28, 3).ToTensor();
        // var conv1 = Tensors.NCHWToNHWC(ReWriteTest.DummyOp.Conv2D(Tensors.NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
        // Assert.True(conv1.InferenceType());
        // Assert.Equal(new long[] { 1, 14, 14, 8 }, conv1.Evaluate().AsTensor().ToOrtTensor().shape);
    }

    [Fact]
    public void TestProd()
    {
        var input = Tensor.FromSpan<int>(new[] { 1, 2, 3, 4 });
        var prod = Tensors.Prod(input);
        prod.InferenceType();
        Assert.Equal(1 * 2 * 3 * 4, prod.Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestSize()
    {
        var input = OrtKI.Random(1, 3, 224, 224).ToTensor();
        var size = Tensors.SizeOf(input);
        size.InferenceType();
        Assert.Equal(1 * 3 * 224 * 224, size.Evaluate().AsTensor().ToScalar<int>());
    }

    private void AssertRangeOf(Expr input, float[] r)
    {
        Assert.Equal(r, RangeOf(input).Evaluate().AsTensor().ToArray<float>());
    }

    [Fact]
    public void TestRangeOf()
    {
        var input = Enumerable.Range(0, 32).Select(x => (float)x);
        var r = new[] { 0f, 31 };
        AssertRangeOf(input.ToArray(), r);
        var n1 = input.ToList();
        n1.Add(float.NaN);
        AssertRangeOf(n1.ToArray(), r);
        var n2 = input.ToList();
        n2.Add(float.PositiveInfinity);
        n2.Add(float.NegativeInfinity);
        AssertRangeOf(n2.ToArray(), r);
    }
}
