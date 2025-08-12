// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldPostOps : TransformTestBase
{
    public static IEnumerable<object[]> TestFoldCastPostOpsPositiveData =>
        new[]
        {
            new object[] { new long[] { 2, 3 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Add, 5.0f, false },
            new object[] { new long[] { 1, 5 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Mul, 2.0f, false },
            new object[] { new long[] { 3, 4 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Add, 10.0f, false },
            new object[] { new long[] { 4, 1 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Mul, 3.0f, false },
            new object[] { new long[] { 2, 3 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Add, 5.0f, true },
            new object[] { new long[] { 1, 5 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Mul, 2.0f, true },
            new object[] { new long[] { 3, 4 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Add, 10.0f, true },
            new object[] { new long[] { 4, 1 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Mul, 3.0f, true },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldCastPostOpsNegativeData =>
        new[]
        {
            new object[] { new long[] { 2, 3 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Sub, 5.0f, false },
            new object[] { new long[] { 1, 5 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Div, 2.0f, false },
            new object[] { new long[] { 3, 4 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Max, 10.0f, false },
            new object[] { new long[] { 4, 1 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Min, 3.0f, false },
            new object[] { new long[] { 2, 3 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Sub, 5.0f, true },
            new object[] { new long[] { 1, 5 }, DataTypes.Float32, DataTypes.Int32, BinaryOp.Div, 2.0f, true },
            new object[] { new long[] { 3, 4 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Max, 10.0f, true },
            new object[] { new long[] { 4, 1 }, DataTypes.Int32, DataTypes.Float32, BinaryOp.Min, 3.0f, true },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldBinaryPostOpsPositiveData =>
        new[]
        {
            new object[] { new long[] { 2, 3 }, new long[] { 2, 3 }, BinaryOp.Add, BinaryOp.Add, 5.0f, false },
            new object[] { new long[] { 1, 5 }, new long[] { 1, 5 }, BinaryOp.Sub, BinaryOp.Mul, 2.0f, false },
            new object[] { new long[] { 3, 4 }, new long[] { 3, 4 }, BinaryOp.Mul, BinaryOp.Add, 10.0f, false },
            new object[] { new long[] { 4, 1 }, new long[] { 4, 1 }, BinaryOp.Div, BinaryOp.Mul, 3.0f, false },
            new object[] { new long[] { 2, 3 }, new long[] { 2, 3 }, BinaryOp.Add, BinaryOp.Add, 5.0f, true },
            new object[] { new long[] { 1, 5 }, new long[] { 1, 5 }, BinaryOp.Sub, BinaryOp.Mul, 2.0f, true },
            new object[] { new long[] { 3, 4 }, new long[] { 3, 4 }, BinaryOp.Mul, BinaryOp.Add, 10.0f, true },
            new object[] { new long[] { 4, 1 }, new long[] { 4, 1 }, BinaryOp.Div, BinaryOp.Mul, 3.0f, true },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    public static IEnumerable<object[]> TestFoldBinaryPostOpsNegativeData =>
        new[]
        {
            new object[] { new long[] { 2, 3 }, new long[] { 2, 3 }, BinaryOp.Add, BinaryOp.Sub, 5.0f, false },
            new object[] { new long[] { 1, 5 }, new long[] { 1, 5 }, BinaryOp.Sub, BinaryOp.Div, 2.0f, false },
            new object[] { new long[] { 3, 4 }, new long[] { 3, 4 }, BinaryOp.Mul, BinaryOp.Max, 10.0f, false },
            new object[] { new long[] { 4, 1 }, new long[] { 4, 1 }, BinaryOp.Div, BinaryOp.Min, 3.0f, false },
            new object[] { new long[] { 2, 3 }, new long[] { 2, 3 }, BinaryOp.Add, BinaryOp.Sub, 5.0f, true },
            new object[] { new long[] { 1, 5 }, new long[] { 1, 5 }, BinaryOp.Sub, BinaryOp.Div, 2.0f, true },
            new object[] { new long[] { 3, 4 }, new long[] { 3, 4 }, BinaryOp.Mul, BinaryOp.Max, 10.0f, true },
            new object[] { new long[] { 4, 1 }, new long[] { 4, 1 }, BinaryOp.Div, BinaryOp.Min, 3.0f, true },
        }.Select((o, i) => o.Concat(new object[] { i }).ToArray());

    [Theory]
    [MemberData(nameof(TestFoldCastPostOpsPositiveData))]
    public void TestFoldCastPostOpsPositive(long[] shape, DataType sourceType, DataType targetType, BinaryOp op, float scalarValue, bool useShapeOneConst, int index)
    {
        var input = Random.Normal(sourceType, 0, 1, 0, shape);
        var cast = IR.F.Tensors.Cast(input, targetType);

        Expr scalar;
        if (useShapeOneConst)
        {
            scalar = Tensor.From(new[] { scalarValue }).CastElementTo(targetType);
        }
        else
        {
            scalar = Tensor.FromScalar(scalarValue).CastElementTo(targetType);
        }

        var rootPre = op switch
        {
            BinaryOp.Add => Math.Add(cast, scalar),
            BinaryOp.Mul => Math.Mul(cast, scalar),
            _ => throw new NotSupportedException($"Operation {op} not supported in test"),
        };

        TestMatched<FoldCastPostOps>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldCastPostOpsNegativeData))]
    public void TestFoldCastPostOpsNegative(long[] shape, DataType sourceType, DataType targetType, BinaryOp op, float scalarValue, bool useShapeOneConst, int index)
    {
        var input = Random.Normal(sourceType, 0, 1, 0, shape);
        var cast = IR.F.Tensors.Cast(input, targetType);

        Expr scalar;
        if (useShapeOneConst)
        {
            scalar = Tensor.From(new[] { scalarValue }).CastElementTo(targetType);
        }
        else
        {
            scalar = Tensor.FromScalar(scalarValue).CastElementTo(targetType);
        }

        var rootPre = op switch
        {
            BinaryOp.Sub => Math.Sub(cast, scalar),
            BinaryOp.Div => Math.Div(cast, scalar),
            BinaryOp.Max => Math.Max(cast, scalar),
            BinaryOp.Min => Math.Min(cast, scalar),
            _ => throw new NotSupportedException($"Operation {op} not supported in test"),
        };

        TestNotMatch<FoldCastPostOps>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldBinaryPostOpsPositiveData))]
    public void TestFoldBinaryPostOpsPositive(long[] lhsShape, long[] rhsShape, BinaryOp binary1Op, BinaryOp binary2Op, float scalarValue, bool useShapeOneConst, int index)
    {
        var lhs = Random.Normal(DataTypes.Float32, 0, 1, 0, lhsShape);
        var rhs = Random.Normal(DataTypes.Float32, 0, 1, 0, rhsShape);
        var binary1 = GetBinaryOp(binary1Op, lhs, rhs);

        Expr scalar;
        if (useShapeOneConst)
        {
            scalar = Tensor.From(new[] { scalarValue });
        }
        else
        {
            scalar = scalarValue;
        }

        var rootPre = binary2Op switch
        {
            BinaryOp.Add => Math.Add(binary1, scalar),
            BinaryOp.Mul => Math.Mul(binary1, scalar),
            _ => throw new NotSupportedException($"Operation {binary2Op} not supported in test"),
        };

        TestMatched<FoldBinaryPostOps>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFoldBinaryPostOpsNegativeData))]
    public void TestFoldBinaryPostOpsNegative(long[] lhsShape, long[] rhsShape, BinaryOp binary1Op, BinaryOp binary2Op, float scalarValue, bool useShapeOneConst, int index)
    {
        var lhs = Random.Normal(DataTypes.Float32, 0, 1, 0, lhsShape);
        var rhs = Random.Normal(DataTypes.Float32, 0, 1, 0, rhsShape);
        var binary1 = GetBinaryOp(binary1Op, lhs, rhs);

        Expr scalar;
        if (useShapeOneConst)
        {
            scalar = Tensor.From(new[] { scalarValue });
        }
        else
        {
            scalar = scalarValue;
        }

        var rootPre = binary2Op switch
        {
            BinaryOp.Sub => Math.Sub(binary1, scalar),
            BinaryOp.Div => Math.Div(binary1, scalar),
            BinaryOp.Max => Math.Max(binary1, scalar),
            BinaryOp.Min => Math.Min(binary1, scalar),
            _ => throw new NotSupportedException($"Operation {binary2Op} not supported in test"),
        };

        TestNotMatch<FoldBinaryPostOps>(rootPre);
    }

    private Expr GetBinaryOp(BinaryOp op, Expr lhs, Expr rhs)
    {
        return op switch
        {
            BinaryOp.Add => Math.Add(lhs, rhs),
            BinaryOp.Sub => Math.Sub(lhs, rhs),
            BinaryOp.Mul => Math.Mul(lhs, rhs),
            BinaryOp.Div => Math.Div(lhs, rhs),
            _ => throw new NotSupportedException($"Operation {op} not supported in test"),
        };
    }
}
