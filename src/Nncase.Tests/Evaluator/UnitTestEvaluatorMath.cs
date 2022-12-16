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
    public void TestBinary_scalar_scalar()
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
                TestBinary_run(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
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
                TestBinary_run(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = 1f;
            var b = 2f;
            if (op != BinaryOp.Mod && op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = 1;
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            {
                TestBinary_run(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = 1L;
            var b = 2L;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }
    }

    [Fact]
    public void TestBinary_tensor_scalar()
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2f;
            if (op != BinaryOp.Mod && op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // int
        foreach(var op in ops)
        {
            var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // long
        foreach(var op in ops)
        {
            var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2L;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }
    }

    [Fact]
    public void TestBinary_tensor_tensor()
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // float
        foreach(var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new float[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.Mod && op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
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
                TestBinary_run(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }),
                    Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }
    }

    private void TestBinary_run(BinaryOp op, OrtKISharp.Tensor ort_a, OrtKISharp.Tensor ort_b, Expr exp_a, Expr exp_b)
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
                expect = ort_a % ort_b;
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
    public void TestBinaryShift()
    {
        var tA = OrtKISharp.Tensor.FromScalar(1U);
        var tB = OrtKI.LeftShift(tA, OrtKISharp.Tensor.FromScalar(2U));
        var tC = OrtKI.RightShift(tA, OrtKISharp.Tensor.FromScalar(2U));

        var a = (Const)1U;
        var b = (Const)2U;

        Assert.Equal(1U << 2, IR.F.Math.LeftShift(a, b).Evaluate().AsTensor().ToScalar<uint>());
        Assert.Equal(1U >> 2, IR.F.Math.RightShift(a, b).Evaluate().AsTensor().ToScalar<uint>());
    }

    [Fact]
    public void TestBinaryShift2()
    {
        var a = (Const)1U;
        var b = (Const)2U;

        Assert.Equal(
            (int)(1U << 2) - 1,
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
