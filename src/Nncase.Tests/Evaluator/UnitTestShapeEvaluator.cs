using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestShapeEvaluator : TestClassBase
{
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

    private (Var, Expr[]) MakeInput(Dimension[] shape)
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
        var varValues = newShape.Where(x => x is Var).ToDictionary(x => (Var)x, _ => (IValue)Value.FromTensor(DefaultDim));
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();

        var fixedShape = newShape.Select(x =>
        {
            return x switch
            {
                Var => 4,
                TensorConst t => t.Value.ToScalar<int>()
            };
        }).ToArray();
        var varsValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(Testing.Rand<float>(fixedShape)) } };
        var fixedShapeResult = expr.Evaluate(varsValues).AsTensor().Shape.ToValueArray();
        Assert.Equal(fixedShapeResult, shapeValue);
    }

    private void TestOpShapeEval(Func<Expr, Expr> exprCtor)
    {
        var (input, newShape) = MakeInput(new[] { 2, 3, Dimension.Unknown, 24 });
        TestOpShapeEval(exprCtor, input, newShape);
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
        TestOpShapeEval(input => Unsqueeze(input, new[]{1}));
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
        var fixedShape = new[]{1, 3, 24, 24};
        var varValues = new Dictionary<Var, IValue> { { input, Value.FromTensor(fixedShape) } };
        var shapeValue = shape.Evaluate(varValues).AsTensor().ToArray<int>();
        Assert.Equal(shapeValue, fixedShape);
    }

    [Fact]
    public void UnitTestGather()
    {
        TestOpShapeEval(input => Gather(input, 2, new[]{0, 1}));
    }

    [Fact]
    public void UnitTestExpand()
    {
        TestOpShapeEval(input => Expand(input, new[]{2, 3, DefaultDim, 24}));
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
        TestOpShapeEval(input => Transpose(input, new[]{0, 2, 3, 1}));
    }

    [Fact]
    public void UnitTestReshape()
    {
        TestOpShapeEval(input => Reshape(input, new[]{1, 3, 12, -1}));
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

    private Expr MakeDim() => new Var(new TensorType(DataTypes.Int32, Shape.Scalar));

    private int DefaultDim = 4;

    [Fact]
    public void UnitTestShapeExprSaveInMeta()
    {
        var (input, newShape) = MakeInput(new[] { 1, 3, Dimension.Unknown, 24 });
        var expr = Softmax(Abs(input), 0);
        var varMap = new Dictionary<Var, Expr[]> { { input, newShape } };
        expr.EvaluateShapeExpr(varMap);
        Assert.NotEqual(expr.Metadata.ShapeExpr , null);
        Assert.NotEqual(expr.Arguments[0].Metadata.ShapeExpr , null);
    }
}
