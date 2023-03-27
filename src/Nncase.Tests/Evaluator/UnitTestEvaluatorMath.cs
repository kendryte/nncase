// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
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

public class UnitTestEvaluatorMath : TestClassBase
{
    public static readonly TheoryData<int[], int[], int[]> ClampInvalidTypeData = new()
    {
        { new[] { 1, 2, 3, 4 }, new[] { 8 }, new[] { 8 } },
        { new[] { 1, 2, 3, 4 }, new[] { 4 }, new[] { 8 } },
        { new[] { 1, 2, 3, 4 }, new[] { 4 }, new[] { 1 } },
    };

    [Fact]
    public void TestBinaryScalarScalar()
    {
        var ops = new BinaryOp[]
        {
            BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift,
        };

        // bool
        foreach (var op in ops)
        {
            var a = false;
            var b = true;
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // uint
        foreach (var op in ops)
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
        foreach (var op in ops)
        {
            var a = 1f;
            var b = 2f;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // int
        foreach (var op in ops)
        {
            var a = 1;
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.FromScalar(b), a, b);
            }
        }

        // long
        foreach (var op in ops)
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
        var ops = new BinaryOp[]
        {
            BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift,
        };

        // bool
        foreach (var op in ops)
        {
            var a = true;
            var b = new bool[] { true, false, false, true, true, false, false, true };
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // uint
        foreach (var op in ops)
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
        foreach (var op in ops)
        {
            var a = 2f;
            var b = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // int
        foreach (var op in ops)
        {
            var a = 2;
            var b = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.FromScalar(a), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), a, Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // long
        foreach (var op in ops)
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
        var ops = new BinaryOp[]
        {
            BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift,
        };

        // bool
        foreach (var op in ops)
        {
            var a = new bool[] { true, false, false, true, true, false, false, true };
            var b = true;
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // uint
        foreach (var op in ops)
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
        foreach (var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2f;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // int
        foreach (var op in ops)
        {
            var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = 2;
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.FromScalar(b), Tensor.From(a, new[] { 2, 4 }), b);
            }
        }

        // long
        foreach (var op in ops)
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
        var ops = new BinaryOp[]
        {
            BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Mod, BinaryOp.Min, BinaryOp.Max, BinaryOp.Pow,
            BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift,
        };

        // bool
        foreach (var op in ops)
        {
            var a = new bool[] { true, false, false, true, true, false, false, true };
            var b = new bool[] { true, false, true, false, true, false, false, true };
            if (op == BinaryOp.LogicalAnd || op == BinaryOp.LogicalOr || op == BinaryOp.LogicalXor)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // uint
        foreach (var op in ops)
        {
            var a = new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new uint[] { 1, 1, 2, 2, 3, 3, 4, 4 };

            // if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor)
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // float
        foreach (var op in ops)
        {
            var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new float[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // int
        foreach (var op in ops)
        {
            var a = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new int[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }

        // long
        foreach (var op in ops)
        {
            var a = new long[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var b = new long[] { 1, 1, 2, 2, 3, 3, 4, 4 };
            if (op != BinaryOp.LogicalAnd && op != BinaryOp.LogicalOr && op != BinaryOp.LogicalXor && op != BinaryOp.LeftShift && op != BinaryOp.RightShift)
            {
                TestBinaryRunNormal(op, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 }), OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 }), Tensor.From(a, new[] { 2, 4 }), Tensor.From(b, new[] { 2, 4 }));
            }
        }
    }

    [Fact]
    public void TestBinaryRunInvalidType()
    {
        {
            var ops = new BinaryOp[] { BinaryOp.LogicalAnd, BinaryOp.LogicalOr, BinaryOp.LogicalXor, BinaryOp.LeftShift, BinaryOp.RightShift };
            foreach (var op in ops)
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

        var result = new float[] { 3, 3, 3, 4, 5, 6, 6, 6 };
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

    [Theory]
    [MemberData(nameof(ClampInvalidTypeData))]
    public void TestClampInvalidType2(int[] inputShape, int[] minShape, int[] maxShape)
    {
        var input = Tensor.FromScalar<float>(3.3f, inputShape);
        var min = Tensor.FromScalar<float>(0.0f, minShape);
        var max = Tensor.FromScalar<float>(6.0f, maxShape);
        var expr = IR.F.Math.Clamp(input, min, max);
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
            bool[] result = { false, false, false, false, false, false, false, false };

            var ort_a = OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 });
            var ort_b = OrtKISharp.Tensor.MakeTensor(b, new long[] { 2, 4 });

            var expr_a = Tensor.From(a, new[] { 2, 4 });
            var expr_b = Tensor.From(b, new[] { 2, 4 });
            _ = Tensor.From(result, new[] { 2, 4 }).ToOrtTensor();

            var ops = new CompareOp[]
            {
                CompareOp.Equal, CompareOp.NotEqual, CompareOp.LowerThan, CompareOp.LowerOrEqual,
                CompareOp.GreaterThan, CompareOp.GreaterOrEqual,
            };

            foreach (var op in ops)
            {
                var expect = op switch
                {
                    CompareOp.NotEqual => OrtKI.Not(OrtKI.Equal(ort_a, ort_b)),
                    CompareOp.Equal => OrtKI.Equal(ort_a, ort_b),
                    CompareOp.LowerThan => OrtKI.Less(ort_a, ort_b),
                    CompareOp.LowerOrEqual => OrtKI.LessOrEqual(ort_a, ort_b),
                    CompareOp.GreaterThan => OrtKI.Greater(ort_a, ort_b),
                    CompareOp.GreaterOrEqual => OrtKI.GreaterOrEqual(ort_a, ort_b),
                    _ => throw new ArgumentOutOfRangeException(nameof(op)),
                };

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

    [Fact]
    public void TestDequantize()
    {
        var input = new byte[] { 127, 128, 150, 160, 170, 180, 200, 205 };
        var axis = 0;
        byte zero_point = 127;
        var scale = 0.01F;

        var input1 = OrtKISharp.Tensor.MakeTensor(input, new long[] { 2, 4 });
        var expect = OrtKI.DequantizeLinear(input1, scale, zero_point, axis);

        var quant_param = new QuantParam(zero_point, scale);
        var input2 = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Math.Dequantize(input2, quant_param, DataTypes.Float32);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestQuantize()
    {
        var input = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
        var axis = 0;
        byte zero_point = 127;
        var scale = 0.05F;

        var input1 = OrtKISharp.Tensor.MakeTensor(input, new long[] { 2, 4 });
        var expect = OrtKI.QuantizeLinear(input1, scale, zero_point, axis);

        var quantParam = new QuantParam(zero_point, scale);
        var input2 = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Math.Quantize(input2, quantParam, DataTypes.UInt8);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestInt16Quantize()
    {
        var input = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
        var axis = 0;
        sbyte zeroPoint = 62;
        var scale = 0.05F;

        var input1 = OrtKISharp.Tensor.MakeTensor(input, new long[] { 2, 4 });

        // onnxruntime does not support quantize to i16, result of kernel is i8
        var expect = OrtKI.Cast(
            OrtKI.QuantizeLinear(input1, scale, zeroPoint, axis),
            (int)DataTypes.Int16.ToOrtType());

        var quantParam = new QuantParam(zeroPoint, scale);
        var input2 = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Math.Quantize(input2, quantParam, DataTypes.Int16);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    [Fact]
    public void TestFakeDequantize()
    {
        var input = new byte[] { 127, 128, 150, 160, 170, 180, 200, 205 };
        byte zero_point = 127;
        var scale = 0.01F;

        var expect = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Math.FakeDequantize(
            Tensor.From(input, new[] { 2, 4 }),
            new QuantParam(zero_point, scale),
            DataTypes.Float32);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestFakeQuantize()
    {
        var input = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
        byte zero_point = 127;
        var scale = 0.05F;

        var expect = Tensor.From(input, new[] { 2, 4 });
        var expr = IR.F.Math.FakeQuantize(
            Tensor.From(input, new[] { 2, 4 }),
            new QuantParam(zero_point, scale),
            DataTypes.UInt8);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
    }

    [Fact]
    public void TestMatmul()
    {
        {
            var input = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            var m1_ort = OrtKISharp.Tensor.MakeTensor(input, new long[] { 2, 4 });
            var m2_ort = OrtKISharp.Tensor.MakeTensor(input, new long[] { 4, 2 });

            var m1 = Tensor.From(input, new[] { 2, 4 });
            var m2 = Tensor.From(input, new[] { 4, 2 });
            var expect = OrtKI.MatMul(m1_ort, m2_ort);

            var expr = IR.F.Math.MatMul(m1, m2);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        }

        {
            var input1 = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F, 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            var input2 = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };

            var m1_ort = OrtKISharp.Tensor.MakeTensor(input1, new long[] { 2, 2, 4 });
            var m2_ort = OrtKISharp.Tensor.MakeTensor(input2, new long[] { 4, 2 });

            var m1 = Tensor.From(input1, new[] { 2, 2, 4 });
            var m2 = Tensor.From(input2, new[] { 4, 2 });
            var expect = OrtKI.MatMul(m1_ort, m2_ort);

            var expr = IR.F.Math.MatMul(m1, m2);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        }
    }

    [Fact]
    public void TestMatmulInvalidType()
    {
        {
            var input1 = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            var input2 = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var m1 = Tensor.From(input1, new[] { 2, 4 });
            var m2 = Tensor.From(input2, new[] { 4, 2 });

            var expr = IR.F.Math.MatMul(m1, m2);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }

        {
            var input1 = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            var input2 = new float[] { 1.0F, 1.2F, 1.4F, 1.5F, 1.6F, 1.8F, 1.9F, 2.0F };
            var m1 = Tensor.From(input1, new[] { 2, 4 });
            var m2 = Tensor.From(input2, new[] { 1, 8 });

            var expr = IR.F.Math.MatMul(m1, m2);
            CompilerServices.InferenceType(expr);
            Assert.IsType<InvalidType>(expr.CheckedType);
        }
    }

    [Fact]
    public void TestQuantParamOf()
    {
        float[] range = new float[] { 0F, 1F };
        QuantMode mode = QuantMode.UnsignedMode;
        int bits = 8;

        var expect = Tensor.FromScalar(QuantUtility.GetQuantParam((range[0], range[1]), bits, mode));
        var expr = IR.F.Math.QuantParamOf(mode, range, bits);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
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
    public void TestReduce()
    {
        long[] axes = { 0 };
        float initValue = 0F;
        long keepDims = 1;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var result = new float[] { 5, 6, 7, 8 };
        var ort_a = OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 });
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        _ = Tensor.From(result, new[] { 1, 4 }).ToOrtTensor();

        var ops = new ReduceOp[] { ReduceOp.Max, ReduceOp.Min, ReduceOp.Mean, ReduceOp.Prod, ReduceOp.Sum };

        foreach (var op in ops)
        {
            var expect = op switch
            {
                ReduceOp.Max => OrtKI.ReduceMax(ort_a, axes, keepDims),
                ReduceOp.Min => OrtKI.ReduceMin(ort_a, axes, keepDims),
                ReduceOp.Mean => OrtKI.ReduceMean(ort_a, axes, keepDims),
                ReduceOp.Prod => OrtKI.ReduceProd(ort_a, axes, keepDims),
                ReduceOp.Sum => OrtKI.ReduceSum(ort_a, axes, keepDims, 0L),
                _ => throw new ArgumentOutOfRangeException(nameof(op)),
            };

            var expr = IR.F.Tensors.Reduce(op, expr_a, Tensor.From(axes, new[] { 1 }), initValue, keepDims);
            CompilerServices.InferenceType(expr);
            Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
        }
    }

    [Fact]
    public void TestReduceArg()
    {
        long axis = 0L;
        long[] keepDims = { 0L, 1L };
        long select_last_idx = 0L;
        var a = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var result = new int[] { 5, 6, 7, 8 };
        var ort_a = OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 4 });
        var expr_a = Tensor.From(a, new[] { 2, 4 });
        _ = Tensor.From(result, new[] { 1, 4 }).ToOrtTensor();

        var ops = new ReduceArgOp[] { ReduceArgOp.ArgMax, ReduceArgOp.ArgMin };

        foreach (var keepdims in keepDims)
        {
            foreach (var op in ops)
            {
                var expect = op switch
                {
                    ReduceArgOp.ArgMax => OrtKI.ArgMax(ort_a, axis, keepdims, select_last_idx),
                    ReduceArgOp.ArgMin => OrtKI.ArgMin(ort_a, axis, keepdims, select_last_idx),
                    _ => throw new ArgumentOutOfRangeException(nameof(op)),
                };

                var expr = IR.F.Tensors.ReduceArg(op, DataTypes.Int64, expr_a, axis, keepdims, select_last_idx);
                CompilerServices.InferenceType(expr);
                Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
            }
        }
    }

    [Fact]
    public void TestRequire()
    {
        var expect = Tensor.FromRange(1, 8);
        var expr = IR.F.Math.Require((Expr)10 > (Expr)9, expect);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor());
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
        var ops = new UnaryOp[]
        {
            UnaryOp.Abs, UnaryOp.Acos, UnaryOp.Acosh, UnaryOp.Asin,
            UnaryOp.Asinh, UnaryOp.Ceil, UnaryOp.Cos, UnaryOp.Cosh, UnaryOp.Exp, UnaryOp.Floor,
            UnaryOp.Log, UnaryOp.Neg, UnaryOp.Round, UnaryOp.Rsqrt, UnaryOp.Sign, UnaryOp.Sin,
            UnaryOp.Sinh, UnaryOp.Sqrt, UnaryOp.Square, UnaryOp.Tanh,
        };
        {
            var f = 1F;
            foreach (var op in ops)
            {
                TestUnaryNormal(op, OrtKISharp.Tensor.FromScalar(f), Tensor.FromScalar(f));
            }
        }

        {
            var f = 123;
            foreach (var op in new UnaryOp[] { UnaryOp.Neg, UnaryOp.Abs, UnaryOp.Square })
            {
                TestUnaryNormal(op, OrtKISharp.Tensor.FromScalar(f), Tensor.FromScalar(f));
            }
        }

        {
            var f = new float[] { 1F, 1.1F, 1.2F, 1.3F };
            foreach (var op in ops)
            {
                TestUnaryNormal(op, OrtKISharp.Tensor.MakeTensor(f, new long[] { 2, 2 }), Tensor.From(f, new[] { 2, 2 }));
            }
        }

        {
            bool[] a = new bool[] { true, false, false, true };
            TestUnaryNormal(UnaryOp.LogicalNot, OrtKISharp.Tensor.MakeTensor(a, new long[] { 2, 2 }), Tensor.From(a, new[] { 2, 2 }));
        }
    }

    private void AssertRangeOf(Expr input, float[] r)
    {
        Assert.Equal(r, RangeOf(input).Evaluate().AsTensor().ToArray<float>());
    }

    private void TestUnaryNormal(UnaryOp op, OrtKISharp.Tensor ort, Expr e)
    {
        var expect = op switch
        {
            UnaryOp.Abs => OrtKI.Abs(ort),
            UnaryOp.Acos => OrtKI.Acos(ort),
            UnaryOp.Acosh => OrtKI.Acosh(ort),
            UnaryOp.Asin => OrtKI.Asin(ort),
            UnaryOp.Asinh => OrtKI.Asinh(ort),
            UnaryOp.Cos => OrtKI.Cos(ort),
            UnaryOp.Cosh => OrtKI.Cosh(ort),
            UnaryOp.Ceil => OrtKI.Ceil(ort),
            UnaryOp.Exp => OrtKI.Exp(ort),
            UnaryOp.Floor => OrtKI.Floor(ort),
            UnaryOp.Log => OrtKI.Log(ort),
            UnaryOp.LogicalNot => OrtKI.Not(ort),
            UnaryOp.Neg => OrtKI.Neg(ort),
            UnaryOp.Round => OrtKI.Round(ort),
            UnaryOp.Rsqrt => OrtKI.Rsqrt(ort),
            UnaryOp.Sign => OrtKI.Sign(ort),
            UnaryOp.Sin => OrtKI.Sin(ort),
            UnaryOp.Sinh => OrtKI.Sinh(ort),
            UnaryOp.Sqrt => OrtKI.Sqrt(ort),
            UnaryOp.Square => OrtKI.Square(ort),
            UnaryOp.Tanh => OrtKI.Tanh(ort),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };

        var expr = IR.F.Math.Unary(op, e);
        CompilerServices.InferenceType(expr);
        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }

    private void TestBinaryRunNormal(BinaryOp op, OrtKISharp.Tensor ort_a, OrtKISharp.Tensor ort_b, Expr exp_a, Expr exp_b)
    {
        static OrtKISharp.Tensor Mod(OrtKISharp.Tensor a, OrtKISharp.Tensor b)
        {
            var fmod = DataTypes.IsFloat(a.DataType.ToDataType()) && DataTypes.IsFloat(b.DataType.ToDataType()) ? 1L : 0L;
            return OrtKI.Mod(a, b, fmod);
        }

        OrtKISharp.Tensor expect = op switch
        {
            BinaryOp.Add => ort_a + ort_b,
            BinaryOp.Sub => ort_a - ort_b,
            BinaryOp.Mul => ort_a * ort_b,
            BinaryOp.Div => ort_a / ort_b,
            BinaryOp.Mod => Mod(ort_a, ort_b),
            BinaryOp.Min => OrtKI.Min(new[] { ort_a, ort_b }),
            BinaryOp.Max => OrtKI.Max(new[] { ort_a, ort_b }),
            BinaryOp.Pow => OrtKI.Pow(ort_a, ort_b),
            BinaryOp.LogicalAnd => OrtKI.And(ort_a, ort_b),
            BinaryOp.LogicalOr => OrtKI.Or(ort_a, ort_b),
            BinaryOp.LogicalXor => OrtKI.Xor(ort_a, ort_b),
            BinaryOp.LeftShift => OrtKI.LeftShift(ort_a, ort_b),
            BinaryOp.RightShift => OrtKI.RightShift(ort_a, ort_b),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };

        var expr = IR.F.Math.Binary(op, exp_a, exp_b);
        CompilerServices.InferenceType(expr);

        Assert.Equal(expect, expr.Evaluate().AsTensor().ToOrtTensor());
    }
}
