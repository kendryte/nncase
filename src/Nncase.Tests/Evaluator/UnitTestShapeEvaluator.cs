// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestShapeEvaluator : TestClassBase
{
    private readonly int _defaultDim = 4;

    public static IEnumerable<object[]> RangeData => new[]
    {
        new object[] { 1, 7, 1 },
        new object[] { 1, 7, 2 },
    };

    [Fact]
    public void TestConstant1()
    {
        var t = Tensor.From(new[] { 1, 3, 24, 24 });
        var tensor = (Expr)t;
        var shape = tensor.EvaluateShapeExpr();
        Assert.Equal(shape, t.Shape.ToValueArray());
    }

    [Fact]
    public void TestConstant2()
    {
        var t = Testing.Rand(DataTypes.Float32, 1, 3, 24, 24);
        var tensor = (Expr)t;
        var shape = tensor.EvaluateShapeExpr();
        Assert.Equal(shape, t.Shape.ToValueArray());
    }

    [Fact]
    public void TestWithVar()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 6 }));
        var dimVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var newShape = new Expr[] { 1, 3, dimVar, 6 };
        var varMap = new Dictionary<Var, Expr[]> { { input, newShape } };
        Assert.Equal(Stack(new IR.Tuple(newShape), 0), input.EvaluateShapeExpr(varMap));
    }

    [Fact]
    public void TestWithTuple()
    {
        var i1 = new[] { 1, 3, 24, 24 };
        var i2 = new[] { 1, 3, 24 };
        var i3 = new[] { 1, 3 };
        var tuple = new Tuple(i1, i2, i3);
        var result = tuple.EvaluateShapeExpr();
        Assert.Equal(result, new Tuple(new[] { 4 }, new[] { 3 }, new[] { 2 }));
    }

    [Fact]
    public void TestUnary()
    {
        TestOpShapeEval(input => Unary(UnaryOp.Abs, input));
    }

    [Fact]
    public void TestBinary()
    {
        TestOpShapeEval(input => input / Testing.Rand<float>(1, 1, 1, 24));
    }

    [Fact]
    public void UnitTestReduceMean()
    {
        TestOpShapeEval(input => Reduce(ReduceOp.Mean, input, new[] { 1, 2 }, 0, true));
    }

    [Fact]
    public void UnitTestUnsqueeze()
    {
        TestOpShapeEval(input => Unsqueeze(input, new[] { 4 }));
        TestOpShapeEval(input => Unsqueeze(input, new[] { -1 }));
    }

    [Fact]
    public void UnitTestTile()
    {
        TestOpShapeEval(input => Tile(input, new long[] { 4, 3, 2, 1 }));
    }

    [Fact]
    public void UnitTestCumSum()
    {
        TestOpShapeEval(input => CumSum(input, 1, 0, 0));
    }

    [Fact]
    public void UnitTestSlice()
    {
        TestOpShapeEval(input => Slice(input, new[] { 1, 1, 1, 1 }, new[] { 2, 3, 3, 3 }, 4));
    }

    [Fact]
    public void UnitTestConcat()
    {
        TestOpShapeEval(input =>
        {
            var c = Concat(new IR.Tuple(input, input), 0);
            return c;
        });
    }

    [Fact]
    public void UnitTestConstantOfShape()
    {
        var input = new Var(new TensorType(DataTypes.Int32, new[] { 4 }));
        var expr = ConstantOfShape(input, 0);
        var shape = expr.EvaluateShapeExpr();
        var fixedShape = new[] { 1, 3, 24, 24 };
        var varValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(fixedShape) } };
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        Assert.Equal(shapeValue, fixedShape);
    }

    [Fact]
    public void UnitTestGather()
    {
        TestOpShapeEval(input => Gather(input, 2, new[] { 0, 1 }));
    }

    [Fact]
    public void UnitTestExpand()
    {
        TestOpShapeEval(input => Expand(input, new[] { 2, 3, _defaultDim, 24 }));
    }

    [Fact]
    public void UnitTestWhere()
    {
        TestOpShapeEval(input => Where(Testing.Rand<bool>(2, 3, 4, 24), input, input));
    }

    [Fact]
    public void UnitTestMatMul()
    {
        TestOpShapeEval(input => IR.F.Math.MatMul(input, Testing.Rand<float>(1, 3, 24, 24)));
    }

    [Fact]
    public void UnitTestShapeOf()
    {
        TestOpShapeEval(input => ShapeOf(input));
    }

    [Fact]
    public void UnitTestTranspose()
    {
        TestOpShapeEval(input => Transpose(input, new[] { 0, 2, 3, 1 }));
        TestOpShapeEval(input => Transpose(input, new[] { 0, 2, 1, 3 }));
    }

    [Fact]
    public void UnitTestReshape()
    {
        TestOpShapeEval(input => Reshape(input, new[] { 1, 3, 12, -1 }));
        TestOpShapeEval(input => Reshape(input, new[] { 1, -1, 12, 3 }));
    }

    [Fact]
    public void UnitTestGetItem()
    {
        var dimVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var input = new Var(new TensorType(DataTypes.Int32, new[] { Dimension.Unknown }));
        var expr = input[1];
        var dict = new Dictionary<Var, Expr[]> { { input, new[] { dimVar } } };
        var shape = expr.EvaluateShapeExpr(dict);
        var varValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(new[] { 4 }) } };
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        var evalShape = expr
            .Evaluate(new Dictionary<Var, IValue> { { input, Value.FromTensor(new[] { 2, 3, 4, 5 }) } })
            .AsTensor()
            .Shape;
        var fixedShape = evalShape.ToValueArray();
        Assert.Equal(fixedShape, shapeValue);
    }

    [Fact]
    public void UnitTestGetItemSingle()
    {
        var dimVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var input = new Var(new TensorType(DataTypes.Int32, new[] { Dimension.Unknown }));
        var expr = input[0];
        var dict = new Dictionary<Var, Expr[]> { { input, new[] { dimVar } } };
        var shape = expr.EvaluateShapeExpr(dict);
        var varValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(new[] { 1 }) } };
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        var evalShape = expr
            .Evaluate(new Dictionary<Var, IValue> { { input, Value.FromTensor(new[] { 2 }) } })
            .AsTensor()
            .Shape;
        var fixedShape = evalShape.ToValueArray();
        Assert.Equal(fixedShape, shapeValue);
    }

    [Fact]
    public void UnitTestEqual()
    {
        TestOpShapeEval(input => Equal(input, input));
    }

    [Fact]
    public void UnitTestLogSoftmax()
    {
        TestOpShapeEval(input => LogSoftmax(input, 0));
    }

    [Fact]
    public void UnitTestSoftmax()
    {
        TestOpShapeEval(input => Softmax(input, 0));
    }

    [Fact]
    public void UnitTestCast()
    {
        TestOpShapeEval(input => Cast(input, DataTypes.Int32));
    }

    [Fact]
    public void UnitTestErf()
    {
        TestOpShapeEval(input => Erf(input));
    }

    [Fact]
    public void UniTestReshape()
    {
        TestOpShapeEval(input => Reshape(input, new long[] { 1, 3, 1, -1 }));
    }

    [Fact]
    public void UnitTestPad()
    {
        TestOpShapeEval(input => Pad(input, new[,] { { 1, 2 }, { 1, 3 }, { 2, 4 }, { 6, 1 } }, PadMode.Constant, 0f));
    }

    [Fact]
    public void TestSpaceTobatch()
    {
        var dimVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var input = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown, 192 }));
        var paddings = Tensor.From(new[] { 0, 1 }, new[] { 1, 2 });
        var expr = SpaceToBatch(input, new[] { 3 }, paddings);
        var dict = new Dictionary<Var, Expr[]> { { input, new Expr[] { 1, dimVar, 192 } } };
        var shape = expr.EvaluateShapeExpr(dict);
        var varValues = new Dictionary<Var, IValue> { { dimVar, Value.FromTensor(8) } };
        Dumpper.DumpIR(shape, "Shape");
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        var evalShape = expr
            .Evaluate(new Dictionary<Var, IValue> { { input, Value.FromTensor(Testing.Rand<float>(1, 8, 192)) } })
            .AsTensor()
            .Shape;
        var fixedShape = evalShape.ToValueArray();
        Assert.Equal(fixedShape, shapeValue);
    }

    [Fact]
    public void TestBatchToSpace()
    {
        var dimVar = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var input = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 69, 192 }));
        var paddings = Tensor.From(new[] { 0, 1 }, new[] { 1, 2 });
        var expr = BatchToSpace(input, new[] { 3 }, paddings);
        var dict = new Dictionary<Var, Expr[]> { { input, new Expr[] { dimVar, 69, 192 } } };
        var shape = expr.EvaluateShapeExpr(dict);
        var varValues = new Dictionary<Var, IValue> { { dimVar, Value.FromTensor(3) } };
        Dumpper.DumpIR(shape, "Shape");
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        var evalShape = expr
            .Evaluate(new Dictionary<Var, IValue> { { input, Value.FromTensor(Testing.Rand<float>(3, 69, 192)) } })
            .AsTensor()
            .Shape;
        var fixedShape = evalShape.ToValueArray();
        Assert.Equal(fixedShape, shapeValue);
    }

    [Fact]
    public void UnitTestSqueeze()
    {
        TestOpShapeEval(input => Squeeze(input, new[] { 0 }));
        TestOpShapeEval(input => Squeeze(input, new[] { -4 }));
    }

    [Theory]
    [MemberData(nameof(RangeData))]
    public void UnitTestRange(int beginV, int endV, int stepV)
    {
        var begin = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var end = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var step = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var expr = Range(begin, end, step);
        var shape = expr.EvaluateShapeExpr();
        var varValues = new Dictionary<Var, IValue>
        {
            { begin, Value.FromTensor(beginV) },
            { end, Value.FromTensor(endV) },
            { step, Value.FromTensor(stepV) },
        };

        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        var fixedShape = expr.Evaluate(varValues).AsTensor().Shape.ToValueArray();
        Assert.Equal(fixedShape, shapeValue);
    }

    [Fact]
    public void TestShapeExprCache()
    {
        var cache = new ShapeExprCache(new Dictionary<Var, Expr[]>());
        var dimVar = new Var("dimVar", new TensorType(DataTypes.Int32, Shape.Scalar));
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new Shape(1, Dimension.Unknown, 24, 24)));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new Shape(1, Dimension.Unknown, 24, 24)));
        var m = IR.F.Math.MatMul(lhs, rhs);
        var dimVarDict = new Dictionary<Var, IValue> { { dimVar, Value.FromTensor(3) } };
        var lhsShape = new Expr[] { 1, dimVar, 24, 24 };
        var rhsShape = new Expr[] { 1, dimVar, 24, 24 };
        var lhsFixedShape = new[] { 1, 3, 24, 24 };
        var rhsFixedShape = new[] { 1, 3, 24, 24 };
        var shapeDict = new Dictionary<Var, Expr[]> { { lhs, lhsShape }, { rhs, rhsShape } };

        var mShapeExpr = m.EvaluateShapeExpr(shapeDict);
        cache.Add(m, mShapeExpr);
        var tr = Transpose(m, new[] { 0, 2, 3, 1 });
        var trShapeExpr = tr.EvaluateShapeExpr(cache + shapeDict);
        var fixedShape = trShapeExpr.Evaluate(dimVarDict).AsTensor().ToArray<int>();
        var originShape = tr.Evaluate(new Dictionary<Var, IValue>
        {
            { lhs, Value.FromTensor(Testing.Rand<float>(lhsFixedShape)) },
            { rhs, Value.FromTensor(Testing.Rand<float>(rhsFixedShape)) },
        }).AsTensor().Shape.ToValueArray();
        Assert.Equal(originShape, fixedShape);
    }

    private Expr MakeDim() => new Var(new TensorType(DataTypes.Int32, Shape.Scalar));

    private (Var Var, Expr[] NewShape) MakeInput(Dimension[] shape)
    {
        var input = new Var(new TensorType(DataTypes.Float32, shape));
        var newShape = shape.Select(x => x.IsFixed ? x.FixedValue : MakeDim()).ToArray();
        return (input, newShape);
    }

    private void TestOpShapeEval(Func<Expr, Expr> exprCtor, Var input, Expr[] newShape)
    {
        var varMap = new Dictionary<Var, Expr[]> { { input, newShape } };
        var expr = exprCtor(input);
        var shape = expr.EvaluateShapeExpr(varMap);
        var varValues = newShape.Where(x => x is Var).ToDictionary(x => (Var)x, _ => (IValue)Value.FromTensor(_defaultDim));
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();

        var fixedShape = newShape.Select(x =>
        {
            return x switch
            {
                Var => 4,
                TensorConst t => t.Value.ToScalar<int>(),
                _ => 4,
            };
        }).ToArray();
        var varsValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(Testing.Rand<float>(fixedShape)) } };
        var fixedShapeResult = expr.Evaluate(varsValues).AsTensor().Shape.ToValueArray();
        Assert.Equal(fixedShapeResult, shapeValue);
    }

    private void TestOpShapeEval(Func<Expr, Expr> exprCtor)
    {
        var (input, newShape) = MakeInput(new[] { 1, 3, Dimension.Unknown, 24 });
        TestOpShapeEval(exprCtor, input, newShape);
    }
}
