// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.Evaluator.Tensors;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Random;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorTensors : TestClassBase
{
    public static IEnumerable<object[]> TestExpandData =>
        new[] { new object[] { new long[] { 16, 16 } }, new object[] { new long[] { 1, 3, 16 } }, new object[] { new long[] { 1, 3, 16, 16 } }, };

    public static IEnumerable<object[]> TestTileData =>
        new[] { new object[] { new long[] { 1, 1, 1, 1 } }, new object[] { new long[] { 1, 2, 1, 1 } }, new object[] { new long[] { 1, 1, 2, 2 } }, };

    [Fact]
    public void TestBitcast()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var newShape = new long[] { 1, 3, 32, 8 };
        var input = OrtKI.Random(oldShape);
        var expect = OrtKI.Reshape(input, newShape, 0);

        var expr = IR.F.Tensors.Bitcast(DataTypes.Float32, input.ToTensor(), DataTypes.Float32, newShape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBroadcast()
    {
        var oldShape = new long[] { 16 };
        var newShape = new[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var expect = input + Tensor.Zeros<float>(new Shape(newShape)).ToOrtTensor();

        var expr = IR.F.Tensors.Broadcast(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestCast()
    {
        var shape = new[] { 1, 3, 16, 16 };
        var input = Tensor.Ones<float>(new Shape(shape));
        var expect = OrtKI.Cast(input.ToOrtTensor(), (long)OrtDataType.Int32);

        var expr = IR.F.Tensors.Cast(input, DataTypes.Int32);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConcat()
    {
        var a = Const.FromTensor(Tensor.From<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 })));
        var b = Const.FromTensor(Tensor.From<int>(new int[12], new Shape(new[] { 1, 3, 4 })));
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
        var a = Tensor.From<int>(Enumerable.Range(0, 12).ToArray(), new Shape(new[] { 1, 3, 4 }));
        var b = Tensor.From<int>(new int[12], new Shape(new[] { 1, 3, 4 }));
        var inputList = new TupleConst(Value.FromTensors(a, b));
        var expr = Tensors.Concat(inputList, 0);
        CompilerServices.InferenceType(expr);

        var tA = a.ToOrtTensor();
        var tB = b.ToOrtTensor();

        Assert.Equal(
            OrtKI.Concat(new[] { tA, tB }, 0),
            expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestConcat3()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var inputA = OrtKI.Random(shape);
        var inputB = OrtKI.Random(shape);

        for (long i = 0; i < shape.Length; i++)
        {
            var expect = OrtKI.Concat(new OrtKISharp.Tensor[] { inputA, inputB }, i);
            var expr = IR.F.Tensors.Concat(new Tuple(inputA.ToTensor(), inputB.ToTensor()), i);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        }
    }

    [Fact]
    public void TestConstantOfShape()
    {
        {
            var shape = new long[] { 1, 3, 16, 16 };
            var value = 1F;
            var expect = Tensor.Ones<float>(new Shape(shape));
            DoConstantOfShape(shape, value, expect);
        }

        {
            var shape = new long[] { 1, 3, 16, 16 };
            var value = 1L;
            var expect = Tensor.Ones<long>(new Shape(shape));
            DoConstantOfShape(shape, value, expect);
        }

        {
            var shape = new long[] { 1, 3, 16, 16 };
            var value = new float[1];
            var expect = Tensor.Zeros<float>(new Shape(shape));
            DoConstantOfShape(shape, value, expect);
        }

        {
            var shape = new long[] { 0 };
            var value = 0F;
            var expect = Tensor.Zeros<float>(new Shape(shape));
            DoConstantOfShape(shape, value, expect);
        }

        {
            var s = new long[] { 1, 3, 16, 16 };
            var shape = new Var(new TensorType(DataTypes.Int64, new int[] { s.Length }));
            var value = new float[1];
            var expect = Tensor.Zeros<float>(new Shape(s));

            var expr = IR.F.Tensors.ConstantOfShape(shape, value);
            CompilerServices.InferenceType(expr);
            var d = new Dictionary<Var, IValue>() { { shape, Value.FromTensor(Tensor.From<long>(s)) } };
            Assert.Equal(expect, expr.Evaluate(d).AsTensor());
        }
    }

    [Theory]
    [MemberData(nameof(TestExpandData))]
    public void TestExpand(long[] newShape)
    {
        var oldShape = new long[] { 1, 16 };
        var input = OrtKI.Random(oldShape);
        var expect = OrtKI.Expand(input, newShape);

        var expr = IR.F.Tensors.Expand(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestFlatten()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var rank = shape.Length;
        var input = OrtKI.Random(shape);

        for (int axis = -rank; axis < rank; axis++)
        {
            var expect = OrtKI.Flatten(input, axis);
            var expr = IR.F.Tensors.Flatten(input.ToTensor(), axis);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        }
    }

    [Fact]
    public void TestProd()
    {
        var input = Tensor.From<int>(new[] { 1, 2, 3, 4 });
        var prod = Tensors.Prod(input);
        prod.InferenceType();
        Assert.Equal(1 * 2 * 3 * 4, prod.Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestProd2()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);

        var expect = OrtKI.ReduceProd(input, new long[] { 0, 1, 2, 3 }, 0);
        var expr = IR.F.Tensors.Prod(input.ToTensor());
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestRange1()
    {
        var begin = 0F;
        var end = 100F;
        var step = 2F;
        var expect = OrtKI.Range(begin, end, step);

        var expr = IR.F.Tensors.Range(begin, end, step);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestRange2()
    {
        var begin = 0F;
        var end = 100F;
        var step = 2;

        var expr = IR.F.Tensors.Range(begin, end, step);
        CompilerServices.InferenceType(expr);
        Assert.IsType<InvalidType>(expr.CheckedType);
    }

    [Fact]
    public void TestRange3()
    {
        var begin = 0F;
        var end = 100F;
        var step = 2F;
        var start = new Var(new TensorType(DataTypes.Float32, new int[] { 1 }));
        var expect = OrtKI.Range(begin, end, step);

        var expr = IR.F.Tensors.Range(start, end, step);
        CompilerServices.InferenceType(expr);
        var d = new Dictionary<Var, IValue>() { { start, Value.FromTensor(Tensor.FromScalar(begin)) } };
        Assert.Equal(expect, expr.Evaluate(d).AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReshape()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var newShape = new long[] { oldShape[0] * oldShape[1], oldShape[2] * oldShape[3] };
        var expect = OrtKI.Reshape(input, newShape, 0);

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReshape2()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var newShape = new long[] { oldShape[0] * oldShape[1], oldShape[2] };

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.IsType<InvalidType>(expr.CheckedType);
    }

    [Fact]
    public void TestReshape3()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var newShape = new long[] { oldShape[0], -1, oldShape[2], -1 };

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.IsType<InvalidType>(expr.CheckedType);
    }

    [Fact]
    public void TestReshape4()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var newShape = new long[] { oldShape[0], -1, 5 };

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.IsType<InvalidType>(expr.CheckedType);
    }

    [Fact]
    public void TestReshape5()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(oldShape);
        var newShape = new long[] { oldShape[0], -1, oldShape[3] };
        var expect = OrtKI.Reshape(input, newShape, 0);

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), newShape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReshape6()
    {
        var oldShape = new long[] { 1, 3, 16, 16 };
        var newShape = new long[] { 3, 256 };
        var dims = new Var(new TensorType(DataTypes.Int64, new int[] { newShape.Length }));
        var input = OrtKI.Random(oldShape);
        var expect = OrtKI.Reshape(input, newShape, 0);

        var expr = IR.F.Tensors.Reshape(input.ToTensor(), dims);
        CompilerServices.InferenceType(expr);
        var d = new Dictionary<Var, IValue>() { { dims, Value.FromTensor(Tensor.From<long>(newShape)) } };
        Assert.Equal(expect, expr.Evaluate(d).AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSizeOf()
    {
        var shape = new Shape(new[] { 1, 3, 16, 16 });
        var input = OrtKI.Random(1, 3, 16, 16).ToTensor();
        var expr = IR.F.Tensors.SizeOf(input);
        CompilerServices.InferenceType(expr);
        Assert.Equal(shape.Size, expr.Evaluate().AsTensor().ToScalar<int>());
    }

    [Fact]
    public void TestSlice()
    {
        var input = Tensor.From<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
        var begin = Tensor.From<int>(new[] { 0, 0, 0, 0 }, new Shape(new[] { 4 }));
        var end = Tensor.From<int>(new[] { 1, 1, 1, 5 }, new Shape(new[] { 4 }));
        var axes = Tensor.From<int>(new[] { 0, 1, 2, 3 }, new Shape(new[] { 4 }));
        var strides = Tensor.From<int>(new[] { 1, 1, 1, 1 }, new Shape(new[] { 4 }));
        var result = Const.FromTensor(Tensor.From<int>(Enumerable.Range(0, 5).ToArray(), new Shape(new[] { 1, 1, 1, 5 })));
        var tResult = result.Value.ToOrtTensor();
        var expr = Tensors.Slice(input, begin, end, axes, strides);
        Assert.True(expr.InferenceType());
        Assert.Equal(
            tResult,
            expr.Evaluate().AsTensor().ToOrtTensor());
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensors.Slice(input, begin, end, 0));
    }

    [Fact]
    public void TestSliceIndex()
    {
        var input = Tensor.From<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
        var expr = Tensors.SliceIndex(input, 1);
        var expect = Slice(input, new[] { 1 }, new[] { 2 }, 1);
        Assert.True(expr.InferenceType());
        Assert.Equal(expect, expr);
    }

    [Fact]
    public void TestNHWCToWNCH()
    {
        var input = Tensor.From<int>(Enumerable.Range(0, 120).ToArray(), new Shape(new[] { 2, 3, 4, 5 }));
        var expr = NHWCToWNCH(input);
        var expect = Transpose(input, new[] { 2, 0, 3, 1 });
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestSlice2()
    {
        var v0 = Slice(new long[3] { 4, 8, 8 }, new[] { 0 }, new[] { 1 }, new[] { 0 }, new[] { 1 }); // i64[1]
        CompilerServices.InferenceType(v0);
        Assert.Equal(1, v0.CheckedShape.Rank);
        var ret = CompilerServices.Evaluate(v0).AsTensor();
        Assert.Equal(1, ret.Shape.Rank);
    }

    [Fact]
    public void TestSqueeze()
    {
        var shape = new long[] { 1, 3, 1, 16 };
        var input = OrtKI.Random(shape);
        var axes = new long[] { 0, 2 };
        var expect = OrtKI.Squeeze(input, axes);

        var expr = IR.F.Tensors.Squeeze(input.ToTensor(), axes);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSqueeze2()
    {
        var shape = new long[] { 1, 3, 1, 16 };
        var axes = new long[] { -2, -4 };
        var dims = new Var(new TensorType(DataTypes.Int64, new int[] { axes.Length }));

        var input = OrtKI.Random(shape);
        var expect = OrtKI.Squeeze(input, axes);

        var expr = IR.F.Tensors.Squeeze(input.ToTensor(), dims);
        CompilerServices.InferenceType(expr);
        var d = new Dictionary<Var, IValue>() { { dims, Value.FromTensor(Tensor.From<long>(axes)) } };
        Assert.Equal(expect, expr.Evaluate(d).AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestSplit1()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);
        var axis = 1L;
        var sections = new long[] { 1, 2 };
        var expect = OrtKI.Split(input, sections, axis);

        var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect.Select(t => t.ToTensor()).ToArray(), expr.Evaluate().AsTensors());
    }

    [Fact]
    public void TestSplit2()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);
        var axis = 1L;
        var sections = new long[] { 1 };
        var expect = OrtKI.Split(input, sections, axis);

        var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect.Select(t => t.ToTensor()).ToArray(), expr.Evaluate().AsTensors());
    }

    [Fact]
    public void TestSplit3()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);
        var axis = 1L;
        {
            var sections = new long[] { 2 };
            var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }

        {
            var sections = new long[] { 1, 2, 1 };
            var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }
    }

    [Fact]
    public void TestSplit4()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var dim = 1L;
        var axis = new Var(new TensorType(DataTypes.Int64, new int[] { 1 }));
        var sections = new long[] { 1, 2 };

        var input = OrtKI.Random(shape);
        var expect = OrtKI.Split(input, sections, dim);

        var expr = IR.F.Tensors.Split(input.ToTensor(), axis, sections);
        CompilerServices.InferenceType(expr);
        var d = new Dictionary<Var, IValue>() { { axis, Value.FromTensor(Tensor.FromScalar(dim)) } };
        Assert.Equal(expect.Select(t => t.ToTensor()).ToArray(), expr.Evaluate(d).AsTensors());
    }

    [Fact]
    public void TestStack()
    {
        Expr a = 1;
        Expr b = 2;
        var inputList = new Tuple(a, b);
        var expr = Tensors.Stack(inputList, 0);
        CompilerServices.InferenceType(expr);
        var ret = expr.Evaluate().AsTensor().ToArray<int>();
        Assert.Equal(new[] { 1, 2 }, ret);
    }

    [Fact]
    public void TestStack2()
    {
        Expr a = 2;
        var inputList = new Tuple(a);
        var expr = Tensors.Stack(inputList, 0);
        CompilerServices.InferenceType(expr);
        var ret = expr.Evaluate().AsTensor().ToArray<int>();
        Assert.Equal(new[] { 2 }, ret);
    }

    [Fact]
    public void TestStack3()
    {
        Expr a = IR.F.Random.Normal(DataTypes.Float32, 1, 1, 1, new[] { 2, 2 });
        {
            var inputList = new Tuple(a);
            var expr = Tensors.Stack(inputList, 0);
            CompilerServices.InferenceType(expr);
            var ret = expr.Evaluate().AsTensor();
            Assert.Equal(new[] { 1, 2, 2 }, ret.Shape.ToValueArray());
        }

        {
            var inputList = new Tuple(a);
            var expr = Tensors.Stack(inputList, 1);
            CompilerServices.InferenceType(expr);
            var ret = expr.Evaluate().AsTensor();
            Assert.Equal(new[] { 2, 1, 2 }, ret.Shape.ToValueArray());
        }

        {
            var inputList = new Tuple(a);
            var expr = Tensors.Stack(inputList, 2);
            CompilerServices.InferenceType(expr);
            var ret = expr.Evaluate().AsTensor();
            Assert.Equal(new[] { 2, 2, 1 }, ret.Shape.ToValueArray());
        }
    }

    [Fact]
    public void TestStack4()
    {
        {
            var a = OrtKI.Random(new long[] { 1, 3, 16, 16 }).ToTensor();
            var b = OrtKI.Random(new long[] { 1, 2, 8, 8 }).ToTensor();

            var inputs = new Tuple(a, b);
            var expr = Tensors.Stack(inputs, 1);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }

        {
            Expr a = 2;
            var input = new Tuple(a);
            var expr = Tensors.Stack(input, 1);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }
    }

    [Theory]
    [MemberData(nameof(TestTileData))]
    public void TestTile1(long[] repeats)
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var input = OrtKI.Random(shape);
        var expect = OrtKI.Tile(input, repeats);

        var expr = IR.F.Tensors.Tile(input.ToTensor(), repeats);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestTile2()
    {
        var shape = new long[] { 1, 3, 16, 16 };
        var a = new long[] { 1, 1, 2, 2 };
        var repeats = new Var(new TensorType(DataTypes.Int64, new int[] { a.Length }));
        var input = OrtKI.Random(shape);
        var expect = OrtKI.Tile(input, a);

        var expr = IR.F.Tensors.Tile(input.ToTensor(), repeats);
        CompilerServices.InferenceType(expr);
        var d = new Dictionary<Var, IValue>() { { repeats, Value.FromTensor(Tensor.From<long>(a)) } };
        Assert.Equal(expect, expr.Evaluate(d).AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestGatherND()
    {
        var shape = new[] { 2, 2 };
        var input = new Tensor<int>(new[] { 0, 1, 2, 3 }, shape);
        var indices = new Tensor<long>(new[] { 0L, 0L, 1L, 1L }, shape);
        long batchDims = 0L;
        var expect = OrtKI.GatherND(input.ToOrtTensor(), indices.ToOrtTensor(), batchDims);

        var expr = IR.F.Tensors.GatherND(input, batchDims, indices);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestScatterND()
    {
        var shape = new[] { 2, 1, 10 };
        var input = Tensor.FromScalar(0f, shape);
        var indices = new Tensor<long>(new[] { 0L, 0L, 1L, 1L, 0L, 1L }, new[] { 2, 1, 1, 3 });
        var updates = new Tensor<float>(new[] { 5f, 10f }, new[] { 2, 1, 1 });

        // var expect = OrtKI.ScatterND(input.ToOrtTensor(), indices.ToOrtTensor(), updates.ToOrtTensor(), "none");
        var expect = Tensor.FromScalar(0f, shape);
        expect[0, 0, 1] = 5f;
        expect[1, 0, 1] = 10f;

        var expr = IR.F.Tensors.ScatterND(input, indices, updates);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestGather()
    {
        var shape = new[] { 2, 2 };
        var input = new Tensor<int>(new[] { 0, 1, 2, 3 }, shape);
        var indices = new Tensor<long>(new[] { 0L, 0L, 1L, 1L }, shape);
        long batchDims = 0L;
        var expect = OrtKI.Gather(input.ToOrtTensor(), indices.ToOrtTensor(), batchDims);

        var expr = IR.F.Tensors.Gather(input, batchDims, indices);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestReverseSequence()
    {
        var shape = new long[] { 4, 4 };
        var input = OrtKI.Random(shape);
        var seqLens = Tensor.From<long>(new long[] { 1, 2, 3, 4 });
        var batchAxis = 1L;
        var timeAxis = 0L;
        var expect = OrtKI.ReverseSequence(input, seqLens.ToOrtTensor(), batchAxis, timeAxis);

        var expr = IR.F.Tensors.ReverseSequence(input.ToTensor(), seqLens, batchAxis, timeAxis);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestTopK()
    {
        var shape = new long[] { 1, 2, 4, 8 };
        var x = OrtKI.Random(shape);
        var k = 1L;
        var axis = -1;
        var largest = 1;
        var sorted = 1;
        var expect = OrtKI.TopK(x, k, axis, largest, sorted);

        var expr = IR.F.Tensors.TopK(x.ToTensor(), k, axis, largest, sorted);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect.Select(n => n.ToValue()).ToArray(), expr.Evaluate());
    }

    [Fact]
    public void TestWhere()
    {
        var shape = new long[] { 2, 2 };
        var con = new Tensor<bool>(new[] { true, false, true, true }, new[] { 2, 2 });
        var x = OrtKI.Random(shape);
        var y = OrtKI.Random(shape);
        var expect = OrtKI.Where(con.ToOrtTensor(), x, y);

        var expr = IR.F.Tensors.Where(con, x.ToTensor(), y.ToTensor());
        var expr1 = IR.F.Tensors.Where(con, x.ToTensor(), y.ToTensor(), true);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        Assert.Throws<NotImplementedException>(() => expr1.Evaluate());

        var conTF = Tensor.From(new[] { true, false });
        var xTF = OrtKI.Random(2);
        var yTF = OrtKI.Random(2);
        var exprTF = IR.F.Tensors.Where(conTF, xTF.ToTensor(), yTF.ToTensor(), true);
        CompilerServices.InferenceType(exprTF);
        var result = conTF.Select((b, i) => (b, i)).Where(t => t.b).Select(t => (long)t.i).ToArray();
        var expectTF = Tensor.From<long>(result, new Shape(result.Length, conTF.Rank));
        Assert.Equal(expectTF, exprTF.Evaluate().AsTensor());
    }

    [Fact]
    public void TestReduceMean()
    {
        long axis = 0L;
        long keepDims = 0L;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        var expr = IR.F.Tensors.ReduceMean(expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expr);
        var expect = Reduce(ReduceOp.Mean, expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestReduceMin()
    {
        long axis = 0L;
        long keepDims = 0L;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        var expr = IR.F.Tensors.ReduceMin(expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expr);
        var expect = Reduce(ReduceOp.Min, expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestReduceMax()
    {
        long axis = 0L;
        long keepDims = 0L;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        var expr = IR.F.Tensors.ReduceMax(expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expr);
        var expect = Reduce(ReduceOp.Max, expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    [Fact]
    public void TestReduceSum()
    {
        long axis = 0L;
        long keepDims = 0L;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        var expr = IR.F.Tensors.ReduceSum(expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expr);
        var expect = Reduce(ReduceOp.Sum, expr_a, axis, 0f, keepDims);
        CompilerServices.InferenceType(expect);
        Assert.Equal(expr, expect);
    }

    private void DoConstantOfShape(long[] shape, Expr value, Tensor expect)
    {
        var expr = IR.F.Tensors.ConstantOfShape(shape, value);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }
}
