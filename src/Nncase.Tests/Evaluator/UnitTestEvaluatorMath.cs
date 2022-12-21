using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Autofac;
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
using static Nncase.Utilities.DumpUtility;
using RangeOf = Nncase.IR.Math.RangeOf;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorMath : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestBinaryScalarScalar()
    {
        BinaryOp[] ops = new BinaryOp[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };

        // bool
        foreach(var op in ops)
        {
            var a = false;
            var b = true;
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // uint
        foreach(var op in ops)
        {
            var a = 1U;
            var b = 2U;
            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = 1f;
            var b = 2f;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = 1;
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = 1L;
            var b = 2L;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }
    }

    [Fact]
    public void TestBinaryScalarTensor()
    {
        BinaryOp[] ops = new BinaryOp[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };

        // bool
        foreach(var op in ops)
        {
            var a = true;
            var b = new bool[] { true, false, false, true, true, false, false, true };
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // uint
        foreach(var op in ops)
        {
            var a = 2U;
            var b = new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = 2f;
            var b = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = 2;
            var b = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = 2L;
            var b = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }
    }

    [Fact]
    public void TestBinaryTensorScalar()
    {
        BinaryOp[] ops = new BinaryOp[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };

        // bool
        foreach(var op in ops)
        {
            var a = new bool[] { true, false, false, true, true, false, false, true };
            var b = true;
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // uint
        foreach(var op in ops)
        {
            var a = new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2U;
            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2f;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2L;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }
    }

    [Fact]
    public void TestBinaryTensorTensor()
    {
        BinaryOp[] ops = new BinaryOp[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };

        // bool
        foreach(var op in ops)
        {
            var a = new bool[] { true, false, false, true, true, false, false, true };
            var b = new bool[] { true, false, true, false, true, false, false, true };
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // uint
        foreach(var op in ops)
        {
            var a = new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new uint[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new float[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new int[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new long[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }
    }

    private void TestBinaryRunNormal(BinaryOp op, OrtKISharp.Tensor ort_a, OrtKISharp.Tensor ort_b, Expr exp_a, Expr exp_b)
    {
        OrtKISharp.Tensor expect = ort_a;
        switch (op)
        {
            case BinaryOp.Add:
            {
                expect = ort_a + ort_b;
                break;
            }
            case BinaryOp.Sub:
            {
                expect = ort_a - ort_b;
                break;
            }

            case BinaryOp.Mul:
            {
                expect = ort_a * ort_b;
                break;
            }
            case BinaryOp.Div:
            {
                expect = ort_a / ort_b;
                break;
            }
            case BinaryOp.Mod:
            {
                if (DataTypes.IsFloat(ort_a.DataType.ToDataType()) && DataTypes.IsFloat(ort_b.DataType.ToDataType()))
                {
                    expect = OrtKI.Mod(ort_a, ort_b, 1);
                }
                else
                {
                    expect = OrtKI.Mod(ort_a, ort_b, 0);
                }
                break;
            }
            case BinaryOp.Min:
            {
                expect = OrtKI.Min(new[] { ort_a, ort_b });
                break;
            }
            case BinaryOp.Max:
            {
                expect = OrtKI.Max(new[] { ort_a, ort_b });
                break;
            }
            case BinaryOp.Pow:
            {
                expect = OrtKI.Pow(ort_a, ort_b);
                break;
            }
            case BinaryOp.LogicalAnd:
            {
                expect = OrtKI.And(ort_a, ort_b);
                break;
            }
            case BinaryOp.LogicalOr:
            {
                expect = OrtKI.Or(ort_a, ort_b);
                break;
            }
            case BinaryOp.LogicalXor:
            {
                expect = OrtKI.Xor(ort_a, ort_b);
                break;
            }
            case BinaryOp.LeftShift:
            {
                expect = OrtKI.LeftShift(ort_a, ort_b);
                break;
            }
            case BinaryOp.RightShift:
            {
                expect = OrtKI.RightShift(ort_a, ort_b);
                break;
            }
            default:
            {
                throw new ArgumentOutOfRangeException(nameof(op));
            }
        }

        var expr = IR.F.Math.Binary(op, exp_a, exp_b);
        CompilerServices.InferenceType(expr);

        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestBinaryRunInvalidType()
    {
        {
            BinaryOp[] ops = new BinaryOp[] { BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };
            foreach(var op in ops)
            {
                var expr = IR.F.Math.Binary(op, 1f, 2f);
                CompilerServices.InferenceType(expr);
                Assert.IsType<InvalidType>(expr.CheckedType);
            }
        }
    }

    [Fact]
    public void TestClamp()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var min = 3f;
        var max = 6f;
        var expr = IR.F.Math.Clamp(Tensor.From(input, new[] { 2, 4 }), min, max);
        CompilerServices.InferenceType(expr);

        var result = new float[] { 3, 3, 3, 4, 5, 6, 6, 6};
        var expect = Tensor.From(result, new[] { 2, 4 });
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestClampInvalidType()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var min = 3U;
        var max = 6L;
        var expr = IR.F.Math.Clamp(Tensor.From(input, new[] { 2, 4 }), min, max);
        CompilerServices.InferenceType(expr);
        Assert.IsType<InvalidType>(expr.CheckedType);
    }

    [Fact]
    public void TestCompare()
    {
        Assert.True(CompilerServices.Evaluate((Expr)5 < (Expr)10).AsTensor().ToScalar<bool>());
        Assert.False(CompilerServices.Evaluate((Expr)10 < (Expr)5).AsTensor().ToScalar<bool>());

        Assert.True(CompilerServices.Evaluate((Expr)5 <= (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)5 <= (Expr)5).AsTensor().ToScalar<bool>());
        Assert.False(CompilerServices.Evaluate((Expr)(-1) <= (Expr)(-2)).AsTensor().ToScalar<bool>());

        Assert.False(CompilerServices.Evaluate((Expr)1 > (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)1 > (Expr)0).AsTensor().ToScalar<bool>());

        Assert.False(CompilerServices.Evaluate((Expr)1 >= (Expr)10).AsTensor().ToScalar<bool>());
        Assert.True(CompilerServices.Evaluate((Expr)1 >= (Expr)0).AsTensor().ToScalar<bool>());

        {
            var ort_a = OrtKISharp.Tensor.FromScalar<int>(10);
            var ort_b = OrtKISharp.Tensor.FromScalar<int>(-2);
            var expect1 = OrtKI.Equal(ort_a, ort_a);
            var expect2 = OrtKI.Not(OrtKI.Equal(ort_a, ort_b));

            var expr_a = Tensor.FromScalar<int>(10);
            var expr_b = Tensor.FromScalar<int>(-2);

            var expr1 = IR.F.Math.Compare(CompareOp.Equal, expr_a, expr_a);
            CompilerServices.InferenceType(expr1);
            Assert.Equal(expect1, expr1.Evaluate().AsTensor().ToOrtTensor());

            var expr2 = IR.F.Math.Compare(CompareOp.NotEqual, expr_a, expr_b);
            CompilerServices.InferenceType(expr2);
            Assert.Equal(expect2, expr2.Evaluate().AsTensor().ToOrtTensor());
        }

        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new float[] { 4, 4, 4, 4, 4, 4, 4, 4 };
            bool []result = { false, false, false, false, false, false, false, false };

            var ort_a = OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 });
            var ort_b = OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 });

            var expr_a = Tensor.From(a, new[] { 2, 4 });
            var expr_b = Tensor.From(b, new[] { 2, 4 });
            var expect = Tensor.From(result, new[] { 2, 4 }).ToOrtTensor();

            CompareOp []ops = new CompareOp[] { CompareOp.Equal, CompareOp.NotEqual, CompareOp.LowerThan, CompareOp.LowerOrEqual,
                CompareOp.GreaterThan, CompareOp.GreaterOrEqual };

            foreach(var op in ops)
            {
                switch (op)
                {
                    case CompareOp.NotEqual:
                    {
                        expect = OrtKI.Not(OrtKI.Equal(ort_a, ort_b));
                        break;
                    }
                    case CompareOp.Equal:
                    {
                        expect = OrtKI.Equal(ort_a, ort_b);
                        break;
                    }
                    case CompareOp.LowerThan:
                    {
                        expect = OrtKI.Less(ort_a, ort_b);
                        break;
                    }
                    case CompareOp.LowerOrEqual:
                    {
                        expect = OrtKI.LessOrEqual(ort_a, ort_b);
                        break;
                    }
                    case CompareOp.GreaterThan:
                    {
                        expect = OrtKI.Greater(ort_a, ort_b);
                        break;
                    }
                    case CompareOp.GreaterOrEqual:
                    {
                        expect = OrtKI.GreaterOrEqual(ort_a, ort_b);
                        break;
                    }
                }

                var expr = IR.F.Math.Compare(op, expr_a, expr_b);
                CompilerServices.InferenceType(expr);

                Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
            }
        }
    }

    [Fact]
    public void TestCondition()
    {
        var expect = Tensor.FromRange(1, 8);
        var expr = IR.F.Math.Condition((Expr)10 > (Expr)9, expect);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestCumsum()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var axis = 0;
        var exclusive = false;
        var reverse = false;

        var input1 = OrtKISharp.Tensor.MakeTensor(input, new long[] { 2, 4 });
        var expect = OrtKI.CumSum(input1, axis, exclusive ? 1L : 0L, reverse ? 1L : 0L);

        var input2 = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Tensors.CumSum(input2, axis, exclusive, reverse);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
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

    [Fact]
    public void TestSelect()
    {
        var expect1 = Tensor.FromRange(1, 8);
        var expect2 = Tensor.FromRange(9, 8);
        {
            var expr = IR.F.Math.Select(true, expect1, expect2);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect1, expr.Evaluate().AsTensor());
        }
        {
            var expr = IR.F.Math.Select(false, expect1, expect2);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect2, expr.Evaluate().AsTensor());
        }
    }

    [Fact]
    public void TestUnary()
    {
        var a = (Const)7f;
        var tA = OrtKISharp.Tensor.FromScalar(7f);
        var expr = -a;
        CompilerServices.InferenceType(expr);
        Assert.Equal(
            -tA,
            expr.Evaluate().AsTensor().ToOrtTensor());
    }
}
