// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using Nncase;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestDistTypeInfer : TestClassBase
{
    public static TheoryData<UnaryOp, DistributedType, IRType> InferUnaryData => new() {
        { UnaryOp.Exp,
          new(new(DataTypes.Float32, new[] { 48, 32 }), new[] { SBP.B, SBP.B }, new(Placement.DeviceKind.CPU, new[] { 2, 4 }, "bt")),
          new DistributedType(new(DataTypes.Float32, new[] { 48, 32 }), new[] { SBP.B, SBP.B }, new(Placement.DeviceKind.CPU, new[] { 2, 4 }, "bt"))
        },
        { UnaryOp.Exp,
          new(new(DataTypes.Float32, new[] { 384, 128 }), new SBP[] { SBP.S(0), SBP.P }, new(Placement.DeviceKind.CPU, new[] { 2, 4 }, "bt")),
          new InvalidType(string.Empty)
        },
    };

    [Theory]
    [MemberData(nameof(InferUnaryData))]
    public void TestInferUnary(UnaryOp unaryOp, DistributedType inputType, IRType outType)
    {
        var input = new Var(inputType);
        var unary = IR.F.Math.Unary(unaryOp, input);
        CompilerServices.InferenceType(unary);
        if (outType is InvalidType && unary.CheckedType is InvalidType)
        {
        }
        else
        {
            Assert.Equal(outType, unary.CheckedType);
        }
    }

    [Theory]
    [ClassData(typeof(InferMatmulData))]
    public void TestInferMatmul(DistributedType lhsType, DistributedType rhsType, IRType outType)
    {
        var lhs = new Var(lhsType);
        var rhs = new Var(rhsType);
        var matmul = IR.F.Math.MatMul(lhs, rhs);
        CompilerServices.InferenceType(matmul);
        if (outType is InvalidType && matmul.CheckedType is InvalidType)
        {
        }
        else
        {
            Assert.Equal(outType, matmul.CheckedType);
        }
    }

    [Fact]
    public void TestInferMatmulProduct()
    {
        var placement = new Placement(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt");
        var lhsShape = new[] { 1, 64, 384, 8192 };
        var rhsShape = new[] { 1, 64, 8192, 384 };
        var lhsType = new TensorType(DataTypes.Float32, lhsShape);
        var rhsType = new TensorType(DataTypes.Float32, rhsShape);
        var candiateArgs = new[] {
            DistributedUtilities.GetLeafCandidateNDSBPs(lhsType, placement).Select(ndsbp => new DistributedType(lhsType, ndsbp, placement)),
            DistributedUtilities.GetLeafCandidateNDSBPs(rhsType, placement).Select(ndsbp => new DistributedType(rhsType, ndsbp, placement)),
            }.
            CartesianProduct().
            Select(argTypes => argTypes.Select(type => new Var(type)).ToArray());

        int count = 0;
        using (var f = Dumpper.OpenFile("types.txt"))
        {
            using var wr = new StreamWriter(f);
            foreach (var args in candiateArgs)
            {
                var matmul = IR.F.Math.MatMul(args[0], args[1]);
                matmul.InferenceType();
                if (matmul.CheckedType is not InvalidType)
                {
                    count++;
                }

                wr.WriteLine($"{args[0].CheckedType} {args[1].CheckedType} => {matmul.CheckedType}");
            }
        }

        Assert.Equal(64, count);
    }
}

internal sealed class InferMatmulData : TheoryData<DistributedType, DistributedType, IRType>
{
    public InferMatmulData()
    {
        var placement = new Placement(Placement.DeviceKind.CPU, new[] { 8, 4 }, "bt");
        var lhs = new[] { 1, 64, 384, 8192 };
        var rhs = new[] { 1, 64, 8192, 384 };
        var o = new[] { 1, 64, 384, 384 };

        Add(
            new(new(DataTypes.Float32, new[] { 1, 1, 384, 8192 }), new[] { SBP.S(2), SBP.S(2) }, placement),
            new(new(DataTypes.Float32, new[] { 1, 64, 8192, 128 }), new SBP[] { SBP.S(2), SBP.S(2) }, placement),
            new InvalidType(string.Empty));
        Add(
            new(new(DataTypes.Float32, lhs), new[] { SBP.S(1), SBP.S(2) }, placement),
            new(new(DataTypes.Float32, rhs), new SBP[] { SBP.S(1), SBP.B }, placement),
            new DistributedType(new(DataTypes.Float32, o), new[] { SBP.S(1), SBP.S(2) }, placement));
        Add(
            new(new(DataTypes.Float32, lhs), new[] { SBP.S(1), SBP.S(3) }, placement),
            new(new(DataTypes.Float32, rhs), new SBP[] { SBP.S(1), SBP.S(2) }, placement),
            new DistributedType(new(DataTypes.Float32, o), new SBP[] { SBP.S(1), SBP.P }, placement));
    }
}
