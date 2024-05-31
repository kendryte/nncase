// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestCombineBinary
{
    public static readonly TheoryData<int[], Tensor<float>, Tensor<float>, Tensor<float>> CombineClampBinaryPositiveData2 = new()
    {
        {
            new[] { 1, 4, 3, 3 },
            Tensor.From(new float[] { 0.23068452f, 0.8913302f, 0.36510944f, -2.4865444f }),
            Tensor.From(new float[] { 0.64266944f, -0.61224914f, -0.61040306f, 0.16890381f }),
            Tensor.From(new float[] { float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity })
        },
    };

    public static IEnumerable<object[]> CombineClampBinaryPositiveData
    {
        get
        {
            var inputShapes = new object[]
            {
              new[] { 1, 32, 24, 24 },
            };
            var constTensor = new object[]
            {
              IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 32 }).Evaluate().AsTensor().Cast<float>(),
              IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, new[] { 24, 24, 32 }).Evaluate().AsTensor().Cast<float>(),
            };
            var clampShapes = new object[]
            {
              Array.Empty<int>(),
              new[] { 32 },
              new[] { 24, 24, 32 },
            };
            var mins = new object[]
            {
              0.0f, float.NegativeInfinity,
            };
            var maxs = new object[]
            {
              1.0f, float.PositiveInfinity,
            };
            return LinqExtensions.CartesianProduct(new[] { inputShapes, constTensor, clampShapes, mins, maxs }).
              Select(p => p.ToArray()).
              Select(p => new object[]
                {
                p[0],
                p[1],
                Tensor.FromScalar<float>((float)p[3],  (int[])p[2]), // min
                Tensor.FromScalar<float>((float)p[4],  (int[])p[2]), // max
                });
        }
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampAddPositive(int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        var (input, rootPre) = GetCombineClampBinaryCase(BinaryOp.Add, inputShape, constTensor, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate() },
        };

        CompilerServices.InferenceType(rootPre);
        var preHashCode = rootPre.GetHashCode();
        var preValue = CompilerServices.Evaluate(rootPre, feedDict);
        var rootPost = CompilerServices.Rewrite(
            rootPre,
            new IRewriteRule[]
            {
                new Passes.Rules.Neutral.CombineClampAdd(),
            },
            new());

        Assert.NotEqual(preHashCode, rootPost.GetHashCode());
        Assert.True(Comparator.Compare(preValue, CompilerServices.Evaluate(rootPost, feedDict)));
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    [MemberData(nameof(CombineClampBinaryPositiveData2))]
    public void TestCombineClampMulPositive(int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        var (input, rootPre) = GetCombineClampBinaryCase(BinaryOp.Mul, inputShape, constTensor, min, max);

        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate() },
        };

        CompilerServices.InferenceType(rootPre);
        var preHashCode = rootPre.GetHashCode();
        var preValue = CompilerServices.Evaluate(rootPre, feedDict);
        var rootPost = CompilerServices.Rewrite(
            rootPre,
            new IRewriteRule[]
            {
                new Passes.Rules.Neutral.CombineClampMul(),
            },
            new());

        Assert.NotEqual(preHashCode, rootPost.GetHashCode());
        Assert.True(Comparator.Compare(preValue, CompilerServices.Evaluate(rootPost, feedDict)));
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampAddNegative(int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        var (input, constInput, rootPre) = GetCombineClampBinaryNegativeCase(BinaryOp.Add, inputShape, constTensor, min, max);
        _ = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, inputShape).Evaluate() },
          { constInput, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, constTensor.Shape).Evaluate() },
        };

        CompilerServices.InferenceType(rootPre);
        var preHashCode = rootPre.GetHashCode();
        var rootPost = CompilerServices.Rewrite(
            rootPre,
            new IRewriteRule[]
            {
                new Passes.Rules.Neutral.CombineClampAdd(),
            },
            new());

        Assert.Equal(preHashCode, rootPost.GetHashCode());
    }

    [Theory]
    [MemberData(nameof(CombineClampBinaryPositiveData))]
    public void TestCombineClampMulNegative(int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        var (input, constInput, rootPre) = GetCombineClampBinaryNegativeCase(BinaryOp.Mul, inputShape, constTensor, min, max);
        _ = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance)
        {
          { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 1, inputShape).Evaluate() },
          { constInput, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 2, constTensor.Shape).Evaluate() },
        };

        CompilerServices.InferenceType(rootPre);
        var preHashCode = rootPre.GetHashCode();
        var rootPost = CompilerServices.Rewrite(
            rootPre,
            new IRewriteRule[]
            {
                new Passes.Rules.Neutral.CombineClampMul(),
            },
            new());

        Assert.Equal(preHashCode, rootPost.GetHashCode());
    }

    private (Var Input, Expr Root) GetCombineClampBinaryCase(BinaryOp op, int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        Expr rootPre;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        {
            var v0 = IR.F.Tensors.NCHWToNHWC(input);
            var v1 = IR.F.Math.Binary(op, v0, constTensor);
            var v2 = IR.F.Math.Clamp(v1, min, max);
            rootPre = v2;
        }

        return (input, rootPre);
    }

    private (Var Input, Var ConstInput, Expr Root) GetCombineClampBinaryNegativeCase(BinaryOp op, int[] inputShape, Tensor<float> constTensor, Tensor<float> min, Tensor<float> max)
    {
        Expr rootPre;
        var input = new Var("input", new TensorType(DataTypes.Float32, inputShape));
        var constInput = new Var("constInput", new TensorType(DataTypes.Float32, constTensor.Shape));
        {
            var v0 = IR.F.Tensors.NCHWToNHWC(input);
            var v1 = IR.F.Math.Binary(op, v0, constInput);
            var v2 = IR.F.Math.Clamp(v1, min, max);
            rootPre = v2;
        }

        return (input, constInput, rootPre);
    }
}
